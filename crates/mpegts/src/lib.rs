//! Reading an ATSC broadcast MPEG transport stream (an HDHomeRun tuner's output): the program's video and main audio as timestamped access units,
//! plus the captions the video carries (ATSC A/53 `cc_data` in the MPEG-2 picture user data). Fed bytes as they
//! arrive, in any chunking.

#![forbid(unsafe_code)]

pub mod h264;
pub mod mpeg2;

use std::collections::HashMap;

pub const PACKET: usize = 188;
/// The 33-bit PES clock wraps every ~26.5 hours at 90 kHz.
const PTS_WRAP: i64 = 1 << 33;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    Mpeg2,
    H264,
    Hevc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Ac3,
    Eac3,
    Aac,
}

/// Caption byte triplets: (cc_type, byte 1, byte 2).
pub type Triplets = Vec<(u8, u8, u8)>;

/// What the demuxer hands back.
#[derive(Debug, PartialEq)]
pub enum Unit {
    /// One video PES: normally one coded picture with its headers. Times in 90 kHz, unwrapped.
    Video { codec: VideoCodec, pts: i64, dts: i64, data: Vec<u8> },
    /// One audio PES: one or more audio frames.
    Audio { codec: AudioCodec, pts: i64, data: Vec<u8> },
    /// Caption byte triplets (cc_type, byte 1, byte 2) carried by the picture with this `pts`, in the order sent.
    Captions { pts: i64, triplets: Triplets },
    /// Data was lost on this stream (a continuity counter jumped): what was being assembled was dropped.
    Discontinuity { video: bool },
}

#[derive(Default)]
struct Pes {
    buf: Vec<u8>,
    cc: Option<u8>,
}

/// Unwraps one stream's 33-bit times into a continuous count.
#[derive(Default)]
struct Clock {
    last: Option<i64>,
    epoch: i64,
}

impl Clock {
    fn unwrap(&mut self, raw: i64) -> i64 {
        if let Some(last) = self.last {
            let prev = last.rem_euclid(PTS_WRAP);
            if raw < prev - PTS_WRAP / 2 {
                self.epoch += PTS_WRAP;
            } else if raw > prev + PTS_WRAP / 2 && self.epoch > 0 {
                self.epoch -= PTS_WRAP; // a little reordering across the wrap
            }
        }
        let t = raw + self.epoch;
        self.last = Some(t);
        t
    }
}

/// A transport stream reader for one program (the first in the PAT).
#[derive(Default)]
pub struct Demuxer {
    pending: Vec<u8>,
    pmt_pid: Option<u16>,
    video: Option<(u16, VideoCodec)>,
    audio: Option<(u16, AudioCodec)>,
    streams: HashMap<u16, Pes>,
    video_clock: Clock,
    audio_clock: Clock,
}

impl Demuxer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The video and audio found so far.
    #[cfg(test)]
    pub fn streams(&self) -> (Option<VideoCodec>, Option<AudioCodec>) {
        (self.video.map(|v| v.1), self.audio.map(|a| a.1))
    }

    /// Feed bytes; returns the units completed by them.
    pub fn push(&mut self, data: &[u8]) -> Vec<Unit> {
        let mut out = Vec::new();
        self.pending.extend_from_slice(data);
        let mut at = 0;
        while at + PACKET <= self.pending.len() {
            if self.pending[at] != 0x47 {
                at += 1; // resync
                continue;
            }
            let pkt: [u8; PACKET] = self.pending[at..at + PACKET].try_into().expect("sized");
            self.packet(&pkt, &mut out);
            at += PACKET;
        }
        self.pending.drain(..at);
        out
    }

    /// The units still being assembled (at the end of the stream).
    pub fn flush(&mut self) -> Vec<Unit> {
        let mut out = Vec::new();
        for pid in [self.video.map(|v| v.0), self.audio.map(|a| a.0)].into_iter().flatten() {
            if let Some(p) = self.streams.get_mut(&pid) {
                let buf = std::mem::take(&mut p.buf);
                self.emit(pid, &buf, &mut out);
            }
        }
        out
    }

    fn packet(&mut self, p: &[u8; PACKET], out: &mut Vec<Unit>) {
        let start = p[1] & 0x40 != 0;
        let pid = (u16::from(p[1] & 0x1f) << 8) | u16::from(p[2]);
        let afc = (p[3] >> 4) & 3;
        let cc = p[3] & 0x0f;
        if p[1] & 0x80 != 0 {
            return; // transport error
        }
        let mut at = 4;
        if afc & 2 != 0 {
            at += 1 + usize::from(p[4]);
        }
        if afc & 1 == 0 || at >= PACKET {
            return; // no payload
        }
        let payload = &p[at..];
        if pid == 0 {
            if start {
                self.pat(payload);
            }
            return;
        }
        if Some(pid) == self.pmt_pid {
            if start {
                self.pmt(payload);
            }
            return;
        }
        let is_video = self.video.map(|v| v.0) == Some(pid);
        if !is_video && self.audio.map(|a| a.0) != Some(pid) {
            return;
        }
        let stream = self.streams.entry(pid).or_default();
        let lost = stream.cc.is_some_and(|last| (last + 1) & 0x0f != cc && last != cc);
        stream.cc = Some(cc);
        if lost {
            stream.buf.clear();
            out.push(Unit::Discontinuity { video: is_video });
            if !start {
                return;
            }
        }
        if start {
            let done = std::mem::take(&mut stream.buf);
            stream.buf.extend_from_slice(payload);
            if !done.is_empty() {
                self.emit(pid, &done, out);
            }
        } else if !stream.buf.is_empty() {
            stream.buf.extend_from_slice(payload);
        }
    }

    /// Section payload after the pointer field, if it is long enough.
    fn section(payload: &[u8]) -> Option<&[u8]> {
        let skip = 1 + usize::from(*payload.first()?);
        let s = payload.get(skip..)?;
        let len = (usize::from(s.get(1)? & 0x0f) << 8) | usize::from(*s.get(2)?);
        s.get(..3 + len)
    }

    fn pat(&mut self, payload: &[u8]) {
        let Some(s) = Self::section(payload) else { return };
        // Programs from byte 8 to the CRC: (number, pid) pairs; number 0 is the network PID.
        for e in s.get(8..s.len().saturating_sub(4)).unwrap_or(&[]).as_chunks::<4>().0 {
            let number = u16::from_be_bytes([e[0], e[1]]);
            if number != 0 {
                self.pmt_pid = Some((u16::from(e[2] & 0x1f) << 8) | u16::from(e[3]));
                return;
            }
        }
    }

    fn pmt(&mut self, payload: &[u8]) {
        let Some(s) = Self::section(payload) else { return };
        if s.len() < 16 {
            return;
        }
        let info_len = (usize::from(s[10] & 0x0f) << 8) | usize::from(s[11]);
        let mut at = 12 + info_len;
        let end = s.len() - 4;
        while at + 5 <= end {
            let ty = s[at];
            let pid = (u16::from(s[at + 1] & 0x1f) << 8) | u16::from(s[at + 2]);
            let es_len = (usize::from(s[at + 3] & 0x0f) << 8) | usize::from(s[at + 4]);
            let descriptors = s.get(at + 5..at + 5 + es_len).unwrap_or(&[]);
            let has = |tag: u8| {
                let mut d = 0;
                while d + 2 <= descriptors.len() {
                    if descriptors[d] == tag {
                        return true;
                    }
                    d += 2 + usize::from(descriptors[d + 1]);
                }
                false
            };
            let video = match ty {
                0x02 => Some(VideoCodec::Mpeg2),
                0x1b => Some(VideoCodec::H264),
                0x24 => Some(VideoCodec::Hevc),
                _ => None,
            };
            let audio = match ty {
                0x81 => Some(AudioCodec::Ac3),
                0x87 => Some(AudioCodec::Eac3),
                0x0f => Some(AudioCodec::Aac),
                0x06 if has(0x6a) => Some(AudioCodec::Ac3),
                0x06 if has(0x7a) => Some(AudioCodec::Eac3),
                _ => None,
            };
            if let Some(v) = video
                && self.video.is_none()
            {
                self.video = Some((pid, v));
            }
            // The first audio stream: the main language (ATSC lists it first).
            if let Some(a) = audio
                && self.audio.is_none()
            {
                self.audio = Some((pid, a));
            }
            at += 5 + es_len;
        }
    }

    fn emit(&mut self, pid: u16, pes: &[u8], out: &mut Vec<Unit>) {
        if pes.len() < 9 || pes[..3] != [0, 0, 1] {
            return;
        }
        let flags = pes[7] >> 6;
        let header_end = 9 + usize::from(pes[8]);
        if header_end > pes.len() || flags & 2 == 0 {
            return; // every access unit we use carries a PTS
        }
        let pts = read_ts(&pes[9..14]);
        let dts = if flags == 3 && pes.len() >= 19 { read_ts(&pes[14..19]) } else { pts };
        // PES_packet_length (when set) bounds the payload; video often leaves it 0 (unbounded).
        let declared = usize::from(u16::from_be_bytes([pes[4], pes[5]]));
        let end = if declared > 0 { (6 + declared).min(pes.len()) } else { pes.len() };
        let data = pes[header_end..end].to_vec();
        if let Some((vpid, codec)) = self.video
            && pid == vpid
        {
            let unwrapped = self.video_clock.unwrap(pts);
            let (pts, dts) = (unwrapped, dts + (unwrapped - pts));
            if codec == VideoCodec::Mpeg2 || codec == VideoCodec::H264 {
                let triplets = if codec == VideoCodec::Mpeg2 { mpeg2::captions(&data) } else { h264::captions(&data) };
                if !triplets.is_empty() {
                    out.push(Unit::Captions { pts, triplets });
                }
            }
            out.push(Unit::Video { codec, pts, dts, data });
        } else if let Some((_, codec)) = self.audio {
            let pts = self.audio_clock.unwrap(pts);
            out.push(Unit::Audio { codec, pts, data });
        }
    }
}

/// `n:d` in lowest terms (`(0, 0)` stays as it is).
pub(crate) fn reduce(n: u64, d: u64) -> (u32, u32) {
    let (mut a, mut b) = (n, d);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let g = a.max(1);
    ((n / g) as u32, (d / g) as u32)
}

/// A 33-bit PES time from its 5 bytes.
fn read_ts(b: &[u8]) -> i64 {
    (i64::from(b[0] & 0x0e) << 29) | (i64::from(b[1]) << 22) | (i64::from(b[2] & 0xfe) << 14) | (i64::from(b[3]) << 7) | (i64::from(b[4]) >> 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 2000 packets of a CBS capture: 1080i MPEG-2, AC-3 English and Spanish, with captions.
    const CBS: &[u8] = include_bytes!("../tests/fixtures/cbs-head.ts");

    #[test]
    fn finds_the_program_video_main_audio_and_captions() {
        let mut d = Demuxer::new();
        // Fed in odd-sized pieces, as the network delivers it.
        let mut units = Vec::new();
        for chunk in CBS.chunks(1000) {
            units.extend(d.push(chunk));
        }
        units.extend(d.flush());
        assert_eq!(d.streams(), (Some(VideoCodec::Mpeg2), Some(AudioCodec::Ac3)));
        let video: Vec<_> =
            units.iter().filter_map(|u| if let Unit::Video { pts, dts, data, .. } = u { Some((*pts, *dts, data.len())) } else { None }).collect();
        let audio: Vec<_> = units.iter().filter_map(|u| if let Unit::Audio { pts, data, .. } = u { Some((*pts, data.len())) } else { None }).collect();
        // ffprobe finds 7 pictures in this slice. The audio is one PES of five AC-3 frames (this broadcaster packs
        // them; ffprobe counts frames), cut short at the end of the slice.
        assert!((6..=7).contains(&video.len()), "{video:?}");
        assert_eq!(audio.len(), 1, "{audio:?}");
        for (pts, dts, _) in &video {
            assert!(dts <= pts);
        }
        // AC-3 frames start with their sync word.
        let first_audio = units.iter().find_map(|u| if let Unit::Audio { data, .. } = u { Some(data) } else { None }).unwrap();
        assert_eq!(first_audio[..2], [0x0b, 0x77]);
        let captions: usize = units.iter().map(|u| if let Unit::Captions { triplets, .. } = u { triplets.len() } else { 0 }).sum();
        assert!(captions > 0, "CEA-608/708 data in the picture user data");
        assert!(!units.iter().any(|u| matches!(u, Unit::Discontinuity { .. })));
    }

    #[test]
    fn unwraps_the_33_bit_clock() {
        let mut c = Clock::default();
        assert_eq!(c.unwrap(PTS_WRAP - 3000), PTS_WRAP - 3000);
        assert_eq!(c.unwrap(1000), PTS_WRAP + 1000, "wrapped");
        assert_eq!(c.unwrap(PTS_WRAP - 500), PTS_WRAP - 500, "a late frame from before the wrap");
        assert_eq!(c.unwrap(4000), PTS_WRAP + 4000);
    }
}
