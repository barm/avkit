//! Reading fragmented MP4 of the kind HLS recordings use (one video track, HEVC; one audio track,
//! AAC): the init segment's track setup, and each fragment's samples with their times. Enough to decode a
//! recording natively (see `native`); not a general MP4 parser.

use anyhow::{Context, bail, ensure};

/// A track from the init segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub id: u32,
    /// Units per second of its sample times.
    pub timescale: u32,
    pub kind: TrackKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackKind {
    /// HEVC: the parameter sets (VPS, SPS, PPS, in that order) and the length of each NAL unit's size prefix.
    /// `hvcc` is the raw `hvcC` box body (for writing the track again).
    Hevc {
        width: u16,
        height: u16,
        parameter_sets: Vec<Vec<u8>>,
        nal_length_size: u8,
        hvcc: Vec<u8>,
    },
    /// AAC: channels, sample rate, and the `esds` payload (the decoder's "magic cookie").
    Aac {
        channels: u16,
        sample_rate: u32,
        esds: Vec<u8>,
    },
    Other,
}

/// What the init segment says: the tracks and their fragment defaults.
#[derive(Debug, Clone, Default)]
pub struct Init {
    pub tracks: Vec<Track>,
    /// `trex` defaults per track: (track id, sample duration, sample size, sample flags).
    defaults: Vec<(u32, u32, u32, u32)>,
}

impl Init {
    pub fn track(&self, pick: impl Fn(&TrackKind) -> bool) -> Option<&Track> {
        self.tracks.iter().find(|t| pick(&t.kind))
    }
}

/// One sample (a coded picture or an audio packet) of a fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub track: u32,
    /// Decode time, in the track's timescale.
    pub dts: i64,
    /// Presentation time, in the track's timescale.
    pub pts: i64,
    /// Where its bytes are, in the buffer the fragment was read from.
    pub offset: usize,
    pub size: usize,
    /// How long it lasts, in the track's timescale.
    pub duration: u32,
}

/// The boxes at the top level of `data` (or inside a box's body): (type, body start, body end).
pub(crate) fn boxes(data: &[u8], start: usize, end: usize) -> anyhow::Result<Vec<([u8; 4], usize, usize)>> {
    let mut out = Vec::new();
    let mut at = start;
    while at + 8 <= end {
        let size32 = u32::from_be_bytes(data[at..at + 4].try_into()?) as usize;
        let kind: [u8; 4] = data[at + 4..at + 8].try_into()?;
        let (header, size) = match size32 {
            0 => (8, end - at),
            1 => {
                ensure!(at + 16 <= end, "truncated box header");
                (16, u64::from_be_bytes(data[at + 8..at + 16].try_into()?) as usize)
            }
            n => (8, n),
        };
        ensure!(size >= header && at + size <= end, "box {:?} runs past its parent", String::from_utf8_lossy(&kind));
        out.push((kind, at + header, at + size));
        at += size;
    }
    Ok(out)
}

fn child<'a>(list: &'a [([u8; 4], usize, usize)], kind: &[u8; 4]) -> Option<&'a ([u8; 4], usize, usize)> {
    list.iter().find(|b| &b.0 == kind)
}

fn u16_at(d: &[u8], at: usize) -> anyhow::Result<u16> {
    Ok(u16::from_be_bytes(d.get(at..at + 2).context("truncated")?.try_into()?))
}
fn u32_at(d: &[u8], at: usize) -> anyhow::Result<u32> {
    Ok(u32::from_be_bytes(d.get(at..at + 4).context("truncated")?.try_into()?))
}
fn u64_at(d: &[u8], at: usize) -> anyhow::Result<u64> {
    Ok(u64::from_be_bytes(d.get(at..at + 8).context("truncated")?.try_into()?))
}

/// Read the init segment (`ftyp` + `moov`).
pub fn parse_init(data: &[u8]) -> anyhow::Result<Init> {
    let top = boxes(data, 0, data.len())?;
    let &(_, ms, me) = child(&top, b"moov").context("no moov box")?;
    let moov = boxes(data, ms, me)?;
    let mut init = Init::default();
    for &(kind, s, e) in &moov {
        if &kind == b"trak" {
            init.tracks.push(parse_trak(data, s, e)?);
        } else if &kind == b"mvex" {
            for &(k, s, _) in &boxes(data, s, e)? {
                if &k == b"trex" {
                    // version/flags, track id, sample description index, duration, size, flags
                    init.defaults.push((u32_at(data, s + 4)?, u32_at(data, s + 12)?, u32_at(data, s + 16)?, u32_at(data, s + 20)?));
                }
            }
        }
    }
    ensure!(!init.tracks.is_empty(), "no tracks");
    Ok(init)
}

fn parse_trak(d: &[u8], s: usize, e: usize) -> anyhow::Result<Track> {
    let trak = boxes(d, s, e)?;
    let &(_, ts, _) = child(&trak, b"tkhd").context("no tkhd")?;
    let id = if d[ts] == 1 { u32_at(d, ts + 20)? } else { u32_at(d, ts + 12)? };
    let &(_, ms, me) = child(&trak, b"mdia").context("no mdia")?;
    let mdia = boxes(d, ms, me)?;
    let &(_, hs, _) = child(&mdia, b"mdhd").context("no mdhd")?;
    let timescale = if d[hs] == 1 { u32_at(d, hs + 20)? } else { u32_at(d, hs + 12)? };
    let &(_, is, ie) = child(&mdia, b"minf").context("no minf")?;
    let minf = boxes(d, is, ie)?;
    let &(_, bs, be) = child(&minf, b"stbl").context("no stbl")?;
    let stbl = boxes(d, bs, be)?;
    let &(_, ds, de) = child(&stbl, b"stsd").context("no stsd")?;
    // stsd: version/flags, entry count, then the sample entries.
    let entries = boxes(d, ds + 8, de)?;
    let kind = match entries.first().map(|(k, s, e)| (k.as_slice(), *s, *e)) {
        Some((b"hvc1" | b"hev1", s, e)) => parse_hevc_entry(d, s, e)?,
        Some((b"mp4a", s, e)) => parse_aac_entry(d, s, e)?,
        _ => TrackKind::Other,
    };
    Ok(Track { id, timescale, kind })
}

/// A VisualSampleEntry: 78 bytes of fields (width at 24, height at 26), then boxes, `hvcC` among them.
fn parse_hevc_entry(d: &[u8], s: usize, e: usize) -> anyhow::Result<TrackKind> {
    let (width, height) = (u16_at(d, s + 24)?, u16_at(d, s + 26)?);
    let inner = boxes(d, s + 78, e)?;
    let &(_, cs, ce) = child(&inner, b"hvcC").context("no hvcC")?;
    let c = &d[cs..ce];
    ensure!(c.len() >= 23, "short hvcC");
    let nal_length_size = (c[21] & 3) + 1;
    let mut sets: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut at = 23;
    for _ in 0..c[22] {
        let kind = c.get(at).context("truncated hvcC")? & 0x3f;
        let count = u16_at(c, at + 1)?;
        at += 3;
        for _ in 0..count {
            let len = u16_at(c, at)? as usize;
            sets.push((kind, c.get(at + 2..at + 2 + len).context("truncated hvcC")?.to_vec()));
            at += 2 + len;
        }
    }
    // VideoToolbox wants them as VPS, SPS, PPS.
    let mut parameter_sets = Vec::new();
    for want in [32, 33, 34] {
        parameter_sets.extend(sets.iter().filter(|(k, _)| *k == want).map(|(_, v)| v.clone()));
    }
    ensure!(parameter_sets.len() >= 3, "hvcC lacks VPS, SPS or PPS");
    Ok(TrackKind::Hevc { width, height, parameter_sets, nal_length_size, hvcc: c.to_vec() })
}

/// An AudioSampleEntry: 28 bytes of fields (channels at 16, rate at 24 as 16.16), then boxes, `esds` among them.
fn parse_aac_entry(d: &[u8], s: usize, e: usize) -> anyhow::Result<TrackKind> {
    let channels = u16_at(d, s + 16)?;
    let sample_rate = u32_at(d, s + 24)? >> 16;
    let inner = boxes(d, s + 28, e)?;
    let &(_, es, ee) = child(&inner, b"esds").context("no esds")?;
    // The payload after version/flags: the ES descriptor.
    Ok(TrackKind::Aac { channels, sample_rate, esds: d[es + 4..ee].to_vec() })
}

/// The samples of every `moof` in `data` (a run of fragments, `moof` + `mdat` pairs), in file order.
pub fn parse_fragments(init: &Init, data: &[u8]) -> anyhow::Result<Vec<Sample>> {
    let mut out = Vec::new();
    for (kind, s, e) in boxes(data, 0, data.len())? {
        if &kind == b"moof" {
            let moof_start = s - 8;
            for (k, ts, te) in boxes(data, s, e)? {
                if &k == b"traf" {
                    parse_traf(init, data, moof_start, ts, te, &mut out)?;
                }
            }
        }
    }
    Ok(out)
}

fn parse_traf(init: &Init, d: &[u8], moof_start: usize, s: usize, e: usize, out: &mut Vec<Sample>) -> anyhow::Result<()> {
    let traf = boxes(d, s, e)?;
    let &(_, hs, _) = child(&traf, b"tfhd").context("no tfhd")?;
    let flags = u32_at(d, hs)? & 0xff_ffff;
    let track = u32_at(d, hs + 4)?;
    let (_, mut dur, mut size, mut sflags) = init.defaults.iter().copied().find(|t| t.0 == track).unwrap_or((track, 0, 0, 0));
    let mut at = hs + 8;
    let mut base = moof_start as u64; // default-base-is-moof (what ffmpeg writes), unless an explicit base is given
    if flags & 0x1 != 0 {
        base = u64_at(d, at)?;
        at += 8;
    }
    if flags & 0x2 != 0 {
        at += 4;
    }
    if flags & 0x8 != 0 {
        dur = u32_at(d, at)?;
        at += 4;
    }
    if flags & 0x10 != 0 {
        size = u32_at(d, at)?;
        at += 4;
    }
    if flags & 0x20 != 0 {
        sflags = u32_at(d, at)?;
    }
    let _ = sflags;
    let mut dts = match child(&traf, b"tfdt") {
        Some(&(_, fs, _)) if d[fs] == 1 => u64_at(d, fs + 4)? as i64,
        Some(&(_, fs, _)) => i64::from(u32_at(d, fs + 4)?),
        None => 0,
    };
    for &(k, rs, _) in &traf {
        if &k != b"trun" {
            continue;
        }
        let version = d[rs];
        let rflags = u32_at(d, rs)? & 0xff_ffff;
        let count = u32_at(d, rs + 4)?;
        let mut at = rs + 8;
        let mut offset = base as i64;
        if rflags & 0x1 != 0 {
            offset += i64::from(u32_at(d, at)? as i32);
            at += 4;
        }
        if rflags & 0x4 != 0 {
            at += 4; // first sample flags
        }
        for _ in 0..count {
            let mut sdur = dur;
            let mut ssize = size;
            let mut cts = 0i64;
            if rflags & 0x100 != 0 {
                sdur = u32_at(d, at)?;
                at += 4;
            }
            if rflags & 0x200 != 0 {
                ssize = u32_at(d, at)?;
                at += 4;
            }
            if rflags & 0x400 != 0 {
                at += 4;
            }
            if rflags & 0x800 != 0 {
                let raw = u32_at(d, at)?;
                cts = if version == 0 { i64::from(raw) } else { i64::from(raw as i32) };
                at += 4;
            }
            let start = usize::try_from(offset).context("negative data offset")?;
            if start + ssize as usize > d.len() {
                bail!("sample data runs past the fragment");
            }
            out.push(Sample { track, dts, pts: dts + cts, offset: start, size: ssize as usize, duration: sdur });
            offset += i64::from(ssize);
            dts += i64::from(sdur);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture: 2.5 s, 320×180 HEVC at 29.97 fps and 5.1 AAC at 48 kHz, written by ffmpeg the way the recorder
    /// writes recordings (single-file fMP4 HLS). Its init segment is bytes 0..1433, then three fragments.
    const MEDIA: &[u8] = include_bytes!("../tests/fixtures/media.mp4");

    #[test]
    fn reads_the_tracks_of_a_recording() {
        let init = parse_init(&MEDIA[..1433]).unwrap();
        let v = init.track(|k| matches!(k, TrackKind::Hevc { .. })).unwrap();
        let TrackKind::Hevc { width, height, parameter_sets, nal_length_size, .. } = &v.kind else { unreachable!() };
        assert_eq!((*width, *height, *nal_length_size), (320, 180, 4));
        assert_eq!(parameter_sets.len(), 3);
        assert_eq!(parameter_sets[0][0] >> 1 & 0x3f, 32, "VPS first");
        let a = init.track(|k| matches!(k, TrackKind::Aac { .. })).unwrap();
        assert_eq!(a.timescale, 48_000);
        let TrackKind::Aac { channels, sample_rate, esds } = &a.kind else { unreachable!() };
        assert_eq!((*channels, *sample_rate), (6, 48_000));
        assert_eq!(esds[0], 0x03, "starts with the ES descriptor tag");
    }

    #[test]
    fn reads_the_samples_of_fragments_with_their_times() {
        let init = parse_init(&MEDIA[..1433]).unwrap();
        let v = init.track(|k| matches!(k, TrackKind::Hevc { .. })).unwrap().clone();
        let a = init.track(|k| matches!(k, TrackKind::Aac { .. })).unwrap().clone();
        let samples = parse_fragments(&init, &MEDIA[1433..]).unwrap();
        let video: Vec<_> = samples.iter().filter(|s| s.track == v.id).collect();
        let audio: Vec<_> = samples.iter().filter(|s| s.track == a.id).collect();
        assert_eq!(video.len(), 75, "2.5 s at 29.97 fps");
        // ~2.5 s of 1024-sample AAC packets (the encoder may add one for priming).
        assert!((117..=120).contains(&audio.len()), "{}", audio.len());
        for w in audio.windows(2) {
            assert_eq!(w[1].dts - w[0].dts, 1024);
        }
        let last = video.iter().map(|s| s.pts).max().unwrap() as f64 / f64::from(v.timescale);
        assert!((2.3..2.6).contains(&last), "{last}");
        // Every sample lies inside the data it was read from, and the HEVC ones are length-prefixed NAL units.
        let data = &MEDIA[1433..];
        for s in &video {
            let len = u32::from_be_bytes(data[s.offset..s.offset + 4].try_into().unwrap()) as usize;
            assert!(len + 4 <= s.size);
        }
    }
}
