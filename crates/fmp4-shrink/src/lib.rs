//! A small copy of a fragmented MP4 recording (HEVC video, AAC sound), for keeping many recordings cheaply: a
//! training library, say. VideoToolbox decodes the pictures straight to the smaller size and encodes them again
//! (HEVC at a constant quality, a keyframe every `key_secs`, every picture kept, since a frame-accurate edge can be
//! a few pictures long); AudioToolbox decodes the sound, which is mixed to one channel and encoded again (AAC).
//!
//! The output is `media.mp4` (one fragmented MP4, a fragment per keyframe) and `index.m3u8`, a byte-range
//! playlist over it, which is what lets a browser seek in it.

#![forbid(unsafe_code)]

use anyhow::{Context, bail, ensure};
use apple_media::analysis::AacDecoder;
use apple_media::audio::{AAC_PACKET_SAMPLES, AacEncoder, RATE};
use apple_media::video::{Decoder, Encoded, Encoder, PixelBuffer, Rate};
use fmp4::hls::{HlsWriter, Layout};
use fmp4::read::{self as mp4, TrackKind};
use fmp4::write::{AudioTrack, Run, Sample, VideoTrack, init_segment};
use std::collections::VecDeque;
use std::path::Path;

/// How small.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Lines of picture (never more than the source has); the width follows the shape.
    pub height: u32,
    /// VideoToolbox's constant quality, 0.0–1.0.
    pub quality: f32,
    /// Bits a second for the one channel of sound.
    pub audio_bitrate: u32,
    /// Seconds between keyframes (and fragments).
    pub key_secs: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { height: 360, quality: 0.5, audio_bitrate: 48_000, key_secs: 2.0 }
    }
}

/// What was written.
#[derive(Clone, Debug)]
pub struct Summary {
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub bytes: u64,
}

/// Samples of silence the AAC encoder puts before the sound it is given.
const ENCODER_PRIMING: i64 = 2112;

/// About this many bytes of the input are decoded at a time (its pictures are held until the batch is done).
const BATCH: usize = 4 << 20;

/// Makes the small copy of `input` in `out_dir` (`media.mp4`, `index.m3u8`). `progress` hears the seconds done.
pub fn shrink(input: &Path, out_dir: &Path, settings: &Settings, progress: &mut dyn FnMut(f64)) -> anyhow::Result<Summary> {
    let (init_bytes, fragments) = fmp4::stream::open(input, BATCH)?;
    let init = mp4::parse_init(&init_bytes)?;
    let v = init.track(|k| matches!(k, TrackKind::Hevc { .. })).context("no HEVC video track")?.clone();
    let TrackKind::Hevc { width: sw, height: sh, parameter_sets, nal_length_size, .. } = &v.kind else { unreachable!() };
    let height = even(settings.height.min(u32::from(*sh)));
    let width = even((f64::from(*sw) * f64::from(height) / f64::from(*sh)).round() as u32);
    let mut decoder = Decoder::hevc(parameter_sets, *nal_length_size, v.timescale, Some((width, height)))?;
    decoder.asynchronous();
    let encoder = Encoder::new(width, height, Rate::Quality(settings.quality), 29.97, settings.key_secs, false, v.timescale)?;
    let audio = match init.track(|k| matches!(k, TrackKind::Aac { .. })) {
        Some(a) => {
            let TrackKind::Aac { channels, sample_rate, esds } = &a.kind else { unreachable!() };
            ensure!(*sample_rate == RATE, "sound at {sample_rate} Hz: only {RATE} Hz is handled");
            Some(Sound {
                track: a.id,
                timescale: a.timescale,
                decoder: AacDecoder::new(*channels, *sample_rate, esds)?,
                encoder: AacEncoder::mono(settings.audio_bitrate)?,
                fed_in: 0,
                came_out: 0,
                // The encoder's output runs its priming (2112 samples) behind its input: start the input that far in,
                // so the copy's sound sits where the source's does when both are decoded the same way. (The DVR's
                // `AAC_PRIMING` is one packet less, the one a player holds back: a constant of the decoder, which
                // cancels out between source and copy.)
                next_pos: ENCODER_PRIMING,
                packets: VecDeque::new(),
                written: 0,
            })
        }
        None => None,
    };
    std::fs::create_dir_all(out_dir)?;
    let mut s = Shrink {
        out_dir: out_dir.to_path_buf(),
        width,
        height,
        vid: v.id,
        vts: v.timescale,
        key_ticks: (settings.key_secs * f64::from(v.timescale)) as i64,
        decoder,
        encoder,
        audio,
        origin: None,
        held: None,
        next_key: 0,
        last_duration: i64::from(v.timescale) / 30,
        encoded: Vec::new(),
        writer: None,
    };
    for batch in fragments {
        let batch = batch?;
        let samples = mp4::parse_fragments(&init, &batch)?;
        s.batch(&batch, &samples)?;
        if let Some((pts, _)) = &s.held {
            progress(*pts as f64 / f64::from(s.vts));
        }
    }
    s.finish()
}

fn even(n: u32) -> u32 {
    (n / 2 * 2).max(2)
}

struct Sound {
    track: u32,
    timescale: u32,
    decoder: AacDecoder,
    encoder: AacEncoder,
    /// Packets given to the decoder and samples taken out, to place its (delayed) output in time.
    fed_in: i64,
    came_out: i64,
    /// The output timeline position (48 kHz samples from the first picture) the encoder's next input sample is for.
    next_pos: i64,
    /// Encoded packets waiting for their fragment, and how many have been written.
    packets: VecDeque<Vec<u8>>,
    written: u64,
}

struct Shrink {
    out_dir: std::path::PathBuf,
    width: u32,
    height: u32,
    vid: u32,
    vts: u32,
    key_ticks: i64,
    decoder: Decoder,
    encoder: Encoder,
    audio: Option<Sound>,
    /// The source time (video timescale) that is zero in the copy.
    origin: Option<i64>,
    /// The latest picture, held until the next one gives its duration.
    held: Option<(i64, PixelBuffer)>,
    next_key: i64,
    last_duration: i64,
    /// Encoded pictures of the fragment being gathered.
    encoded: Vec<Encoded>,
    writer: Option<HlsWriter>,
}

impl Shrink {
    fn batch(&mut self, bytes: &[u8], samples: &[mp4::Sample]) -> anyhow::Result<()> {
        if self.origin.is_none() {
            self.origin = samples.iter().filter(|s| s.track == self.vid).map(|s| s.pts).min();
        }
        let Some(origin) = self.origin else { return Ok(()) };
        let mut pictures = Vec::new();
        for s in samples {
            let data = &bytes[s.offset..s.offset + s.size];
            if s.track == self.vid {
                pictures.extend(self.decoder.decode(data, s.dts, s.pts)?);
            } else if let Some(a) = self.audio.as_mut()
                && s.track == a.track
            {
                a.decode(data, s.pts, origin, self.vts)?;
            }
        }
        pictures.extend(self.decoder.flush());
        pictures.sort_by_key(|p| p.pts);
        for p in pictures {
            let pts = p.pts - origin;
            if pts < 0 || self.held.as_ref().is_some_and(|(h, _)| pts <= *h) {
                continue;
            }
            if let Some((hpts, picture)) = self.held.take() {
                self.encode(hpts, &picture, pts - hpts)?;
            }
            self.held = Some((pts, p.picture));
        }
        Ok(())
    }

    fn encode(&mut self, pts: i64, picture: &PixelBuffer, duration: i64) -> anyhow::Result<()> {
        let key = pts >= self.next_key;
        if key {
            self.next_key = pts - pts % self.key_ticks.max(1) + self.key_ticks;
        }
        self.last_duration = duration;
        let out = self.encoder.encode(picture, pts, duration, key)?;
        self.take(out)
    }

    /// Gathers encoded pictures; a keyframe closes the fragment before it.
    fn take(&mut self, out: Vec<Encoded>) -> anyhow::Result<()> {
        for e in out {
            if e.keyframe && !self.encoded.is_empty() {
                let next = e.dts;
                self.fragment(Some(next))?;
            }
            self.encoded.push(e);
        }
        Ok(())
    }

    /// Writes the gathered pictures, and the sound up to where they end, as one fragment. `next` is the next
    /// picture's decode time (the last one's duration comes from it).
    fn fragment(&mut self, next: Option<i64>) -> anyhow::Result<()> {
        let pics = std::mem::take(&mut self.encoded);
        let Some(first) = pics.first() else { return Ok(()) };
        let first_dts = first.dts;
        let end = next.unwrap_or(pics[pics.len() - 1].dts + self.last_duration);
        let video: Vec<Sample> = pics
            .iter()
            .enumerate()
            .map(|(i, p)| Sample {
                duration: (pics.get(i + 1).map_or(end, |n| n.dts) - p.dts) as u32,
                cts_offset: (p.pts - p.dts) as i32,
                keyframe: p.keyframe,
                data: p.data.clone(),
            })
            .collect();
        let end_secs = end as f64 / f64::from(self.vts);
        let (audio_dts, audio) = match self.audio.as_mut() {
            Some(a) => {
                // The packets that start before the pictures end (all of them, at the end).
                let mut take = Vec::new();
                while !a.packets.is_empty() && (next.is_none() || (a.written as f64 * AAC_PACKET_SAMPLES as f64) / f64::from(RATE) < end_secs) {
                    let data = a.packets.pop_front().unwrap_or_default();
                    take.push(Sample { data, duration: AAC_PACKET_SAMPLES as u32, cts_offset: 0, keyframe: true });
                    a.written += 1;
                }
                ((a.written - take.len() as u64) * AAC_PACKET_SAMPLES as u64, take)
            }
            None => (0, vec![]),
        };
        if self.writer.is_none() {
            let hvcc = self.encoder.hvcc().context("the encoder gave no hvcC")?.to_vec();
            let vt = VideoTrack { width: self.width as u16, height: self.height as u16, timescale: self.vts, hvcc };
            let at = match &self.audio {
                Some(a) => Some(AudioTrack { channels: 1, sample_rate: RATE, esds: a.encoder.esds()? }),
                None => None,
            };
            self.writer = Some(HlsWriter::create(
                &self.out_dir,
                &init_segment(&vt, at.as_ref()),
                (self.key_ticks / i64::from(self.vts)).max(1) as u32 + 1,
                Layout::SingleFile,
            )?);
        }
        let w = self.writer.as_mut().expect("writer");
        let video_run = Run { dts: u64::try_from(first_dts).context("a picture before the start")?, samples: &video };
        let audio_run = Run { dts: audio_dts, samples: &audio };
        let secs = (end - first_dts) as f64 / f64::from(self.vts);
        w.segment(secs, &video_run, (self.audio.is_some() && !audio.is_empty()).then_some(&audio_run))?;
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<Summary> {
        if let Some((pts, picture)) = self.held.take() {
            let d = self.last_duration;
            self.encode(pts, &picture, d)?;
        }
        let out = self.encoder.finish()?;
        self.take(out)?;
        if let Some(a) = self.audio.as_mut() {
            let rest = a.encoder.finish()?;
            a.packets.extend(rest);
        }
        self.fragment(None)?;
        let Some(w) = self.writer else { bail!("no pictures in the recording") };
        let duration = w.finish()?;
        let bytes = std::fs::metadata(self.out_dir.join("media.mp4"))?.len();
        Ok(Summary { width: self.width, height: self.height, duration, bytes })
    }
}

impl Sound {
    /// Decodes one AAC packet presented at `pts` (its own timescale), mixes it to mono and feeds the encoder,
    /// placed on the copy's timeline (`origin` in the video timescale `vts` is zero).
    fn decode(&mut self, packet: &[u8], pts: i64, origin: i64, vts: u32) -> anyhow::Result<()> {
        let channels = self.decoder.channels();
        let pcm = self.decoder.decode_channels(packet)?;
        let mono: Vec<f32> = pcm.chunks_exact(channels).map(downmix).collect();
        // The decoder holds some back: what comes out now began `lag` samples before this packet ended.
        self.fed_in += AAC_PACKET_SAMPLES as i64;
        let lag = self.fed_in - self.came_out;
        self.came_out += mono.len() as i64;
        if mono.is_empty() {
            return Ok(());
        }
        let start = pts + AAC_PACKET_SAMPLES as i64 - lag;
        let secs = start as f64 / f64::from(self.timescale) - origin as f64 / f64::from(vts);
        let pos = (secs * f64::from(RATE)).round() as i64;
        // Continuous within a couple of samples of rounding; otherwise fill a gap with silence or skip an overlap.
        let mut input: &[f32] = &mono;
        let silence;
        let gap = pos - self.next_pos;
        if gap > 2 {
            silence = vec![0.0; gap as usize];
            let packets = self.encoder.push(&silence)?;
            self.packets.extend(packets);
            self.next_pos += gap;
        } else if gap < -2 {
            // What's left starts where the encoder is.
            input = &input[((-gap) as usize).min(input.len())..];
        }
        let packets = self.encoder.push(input)?;
        self.packets.extend(packets);
        self.next_pos += input.len() as i64;
        Ok(())
    }
}

/// One channel from a frame of interleaved channels. 5.1 (L R C LFE Ls Rs) as ITU-R BS.775 folds it to stereo,
/// then the two averaged, without the LFE; stereo averaged; anything else the mean.
fn downmix(f: &[f32]) -> f32 {
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let m = match f.len() {
        6 => 0.5 * (f[0] + f[1]) + H * f[2] + 0.5 * H * (f[4] + f[5]),
        2 => 0.5 * (f[0] + f[1]),
        1 => f[0],
        n => f.iter().sum::<f32>() / n as f32,
    };
    m.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixes_down() {
        assert!((downmix(&[0.0, 0.0, 0.5, 1.0, 0.0, 0.0]) - 0.3535).abs() < 1e-3, "centre at -3 dB, no LFE");
        assert!((downmix(&[0.2, 0.4]) - 0.3).abs() < 1e-6);
        assert_eq!(downmix(&[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]), 1.0, "clipped");
    }

    #[test]
    fn shrinks_the_fixture() {
        // The fixture: 1.5 s of a test pattern with a tone, then 1 s of black and silence.
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fmp4/tests/fixtures/media.mp4");
        let out = std::env::temp_dir().join(format!("fmp4-shrink-{}", std::process::id()));
        let settings = Settings { height: 120, ..Settings::default() };
        let summary = shrink(&input, &out, &settings, &mut |_| {}).unwrap();
        assert_eq!(summary.height, 120);
        assert!((summary.duration - 2.5).abs() < 0.2, "{summary:?}");
        // It reads back: the right size and tracks, and decodes to pictures and sound on the same clock.
        let (init, frags) = fmp4::stream::open(&out.join("media.mp4"), 1 << 20).unwrap();
        let i = mp4::parse_init(&init).unwrap();
        let TrackKind::Hevc { height, .. } = i.track(|k| matches!(k, TrackKind::Hevc { .. })).unwrap().kind else { panic!() };
        assert_eq!(height, 120);
        let TrackKind::Aac { channels, .. } = i.track(|k| matches!(k, TrackKind::Aac { .. })).unwrap().kind else { panic!() };
        assert_eq!(channels, 1);
        let samples: usize = frags.map(|b| mp4::parse_fragments(&i, &b.unwrap()).unwrap().len()).sum();
        assert!(samples > 60, "{samples} samples");
        let playlist = std::fs::read_to_string(out.join("index.m3u8")).unwrap();
        assert!(playlist.contains("#EXT-X-ENDLIST") && playlist.contains("BYTERANGE"));
        #[derive(Default)]
        struct Edges {
            dark: Option<f64>,
            quiet: Option<f64>,
        }
        impl fmp4_scan::Sink for Edges {
            fn picture(&mut self, pts: f64, _w: usize, _h: usize, luma: &[u8]) {
                let mean = luma.iter().map(|&v| f64::from(v)).sum::<f64>() / luma.len() as f64;
                if mean < 20.0 && self.dark.is_none() {
                    self.dark = Some(pts);
                }
            }
            fn sound(&mut self, pts: f64, rate: u32, s: &[f32]) {
                // In 10 ms blocks: the first block, after a loud one, below half the tone's level (0.125).
                let block = rate as usize / 100;
                for (i, b) in s.chunks(block).enumerate() {
                    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    if peak > 0.0625 {
                        self.quiet = None;
                    } else if self.quiet.is_none() {
                        self.quiet = Some(pts + (i * block) as f64 / f64::from(rate));
                    }
                }
            }
        }
        let mut src = Edges::default();
        fmp4_scan::scan_file(&input, 32, 18, 30.0, &mut src).unwrap();
        let mut e = Edges::default();
        fmp4_scan::scan_file(&out.join("media.mp4"), 32, 18, 30.0, &mut e).unwrap();
        // Pictures and sound change where they do in the source, decoded the same way (the sound to the 10 ms
        // block: AAC smears a tone's end a little).
        let (dark, quiet) = (e.dark.unwrap(), e.quiet.unwrap());
        assert!((dark - src.dark.unwrap()).abs() < 0.001 && (dark - 1.5).abs() < 0.01, "dark at {dark}, source {:?}", src.dark);
        assert!((quiet - src.quiet.unwrap()).abs() < 0.011, "quiet at {quiet}, source {:?}", src.quiet);
        std::fs::remove_dir_all(out).unwrap();
    }
}
