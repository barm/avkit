//! Decode a fragmented MP4 recording (HEVC video, AAC sound) into what analysis needs: small grey pictures at a
//! fixed rate and one channel of sound, with their times. VideoToolbox decodes and shrinks the pictures,
//! AudioToolbox the sound (`apple-media`); `fmp4` reads the file.
//!
//! Feed it fragments as they are written ([`Scan::feed`], placed on a timeline you give, as a recorder with a
//! playlist would), or a whole file ([`scan_file`], on the stream's own clock from zero).

#![forbid(unsafe_code)]

use anyhow::Context;
use apple_media::analysis::{AacDecoder, HevcDecoder, Picture};
use fmp4::read::{self as mp4, Init, TrackKind};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Where decoded pictures and sound go.
pub trait Sink {
    /// An 8-bit luma picture, `width`×`height`, shown at `pts` seconds.
    fn picture(&mut self, pts: f64, width: usize, height: usize, luma: &[u8]);
    /// Mono samples at `rate` Hz, the first one at `pts` seconds.
    fn sound(&mut self, pts: f64, rate: u32, samples: &[f32]);
    /// With [`Scan::with_color`]: one picture a second in colour, `size`×`size` planar RGB 0.0–1.0 (the whole
    /// picture squashed to a square), shown at `pts` seconds.
    fn color(&mut self, _pts: f64, _size: usize, _rgb: &[f32]) {}
}

/// A run of a recording's fragments: their bytes, and where they start on the recording's timeline (seconds).
pub struct Fragment<'a> {
    pub bytes: &'a [u8],
    pub start: f64,
}

/// Decodes a recording, fragment by fragment.
pub struct Scan {
    init: Init,
    video: (u32, u32),              // track id, timescale
    audio: Option<(u32, u32, u32)>, // track id, timescale, sample rate
    pictures: HevcDecoder,
    sound: Option<AacDecoder>,
    width: usize,
    height: usize,
    /// Samples given to the sound decoder and taken out of it: the difference is how far its output lags.
    sound_in: i64,
    sound_out: i64,
    /// For [`Scan::feed_stream`]: the stream time (in the video timescale) that is zero.
    origin: Option<i64>,
}

impl Scan {
    /// From the recording's init segment; pictures come out `width`×`height`, `fps` of them a second.
    pub fn new(init_segment: &[u8], width: usize, height: usize, fps: f64) -> anyhow::Result<Self> {
        let init = mp4::parse_init(init_segment)?;
        let v = init.track(|k| matches!(k, TrackKind::Hevc { .. })).context("no HEVC video track")?.clone();
        let TrackKind::Hevc { parameter_sets, nal_length_size, .. } = &v.kind else { unreachable!() };
        // One picture per 1/fps of the stream's time: the first one decoded in each slot (decoding every picture
        // is needed anyway, but only these are shrunk and kept).
        let slot = f64::from(v.timescale) / fps;
        let taken: Arc<Mutex<BTreeSet<i64>>> = Arc::default();
        let want = Box::new(move |pts: i64| {
            let s = (pts as f64 / slot).floor() as i64;
            let mut t = taken.lock().unwrap();
            if t.len() > 256 {
                let keep = t.split_off(&(s - 64));
                *t = keep;
            }
            t.insert(s)
        });
        let pictures = HevcDecoder::new(parameter_sets, *nal_length_size, v.timescale, width, height, want)?;
        let (audio, sound) = match init.track(|k| matches!(k, TrackKind::Aac { .. })) {
            Some(a) => {
                let TrackKind::Aac { channels, sample_rate, esds } = &a.kind else { unreachable!() };
                (Some((a.id, a.timescale, *sample_rate)), Some(AacDecoder::new(*channels, *sample_rate, esds)?))
            }
            None => (None, None),
        };
        Ok(Self { video: (v.id, v.timescale), audio, init, pictures, sound, width, height, sound_in: 0, sound_out: 0, origin: None })
    }

    /// Also hands the sink one picture a second in colour, `size`×`size` (the first decoded in each second).
    pub fn with_color(mut self, size: usize) -> Self {
        let ts = i64::from(self.video.1);
        let mut last: Option<i64> = None;
        self.pictures.set_color(
            size,
            Box::new(move |pts: i64| {
                let s = pts.div_euclid(ts);
                if last.is_some_and(|l| s <= l) {
                    return false;
                }
                last = Some(s);
                true
            }),
        );
        self
    }

    /// Decodes one run of fragments. Times are placed on the recording's timeline by the fragment's start: the
    /// first picture of the fragment is at `start`, the sound on the same clock.
    pub fn feed(&mut self, f: &Fragment, sink: &mut impl Sink) -> anyhow::Result<()> {
        let samples = mp4::parse_fragments(&self.init, f.bytes)?;
        let (vid, vts) = self.video;
        let Some(first) = samples.iter().filter(|s| s.track == vid).map(|s| s.pts).min() else { return Ok(()) };
        self.decode(f.bytes, &samples, f.start - first as f64 / f64::from(vts), sink)
    }

    /// Decodes one run of fragments on the stream's own clock, counted from the first picture fed.
    pub fn feed_stream(&mut self, bytes: &[u8], sink: &mut impl Sink) -> anyhow::Result<()> {
        let samples = mp4::parse_fragments(&self.init, bytes)?;
        let (vid, vts) = self.video;
        let Some(first) = samples.iter().filter(|s| s.track == vid).map(|s| s.pts).min() else { return Ok(()) };
        let origin = *self.origin.get_or_insert(first);
        self.decode(bytes, &samples, -(origin as f64) / f64::from(vts), sink)
    }

    /// `offset`: seconds of stream time → seconds of output time.
    fn decode(&mut self, bytes: &[u8], samples: &[mp4::Sample], offset: f64, sink: &mut impl Sink) -> anyhow::Result<()> {
        let (vid, vts) = self.video;
        let mut pictures: Vec<Picture> = Vec::new();
        for s in samples {
            let data = &bytes[s.offset..s.offset + s.size];
            if s.track == vid {
                pictures.extend(self.pictures.decode(data, s.dts, s.pts)?);
            } else if let (Some((aid, ats, rate)), Some(sound)) = (self.audio, self.sound.as_mut())
                && s.track == aid
            {
                let mono = sound.decode(data)?;
                // The decoder holds some back: what comes out now began `lag` samples before this packet ended.
                self.sound_in += 1024;
                let lag = self.sound_in - self.sound_out;
                self.sound_out += mono.len() as i64;
                if !mono.is_empty() {
                    let start = s.pts + 1024 - lag;
                    sink.sound(start as f64 / f64::from(ats) + offset, rate, &mono);
                }
            }
        }
        pictures.extend(self.pictures.flush());
        pictures.sort_by_key(|p| p.pts);
        for p in &pictures {
            let pts = p.pts as f64 / f64::from(vts) + offset;
            sink.picture(pts, self.width, self.height, &p.luma);
            if let Some(rgb) = &p.rgb {
                sink.color(pts, ((rgb.len() / 3) as f64).sqrt() as usize, rgb);
            }
        }
        Ok(())
    }
}

/// About this many bytes of fragments are decoded at a time by [`scan_file`].
const BATCH: usize = 32 << 20;

/// Decodes a whole fragmented MP4 file, read as a stream (never all in memory), on its own clock from zero.
pub fn scan_file(path: &Path, width: usize, height: usize, fps: f64, sink: &mut impl Sink) -> anyhow::Result<()> {
    let (init, fragments) = fmp4::stream::open(path, BATCH)?;
    let mut scan = Scan::new(&init, width, height, fps)?;
    for batch in fragments {
        scan.feed_stream(&batch?, sink)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEDIA: &[u8] = include_bytes!("../../fmp4/tests/fixtures/media.mp4");

    #[derive(Default)]
    struct Collect {
        pictures: Vec<(f64, f64)>, // pts, mean luma
        sound: Vec<(f64, f32)>,    // pts, peak
    }

    impl Sink for Collect {
        fn picture(&mut self, pts: f64, _w: usize, _h: usize, luma: &[u8]) {
            self.pictures.push((pts, luma.iter().map(|&v| f64::from(v)).sum::<f64>() / luma.len() as f64));
        }
        fn sound(&mut self, pts: f64, _rate: u32, s: &[f32]) {
            self.sound.push((pts, s.iter().fold(0f32, |a, &b| a.max(b.abs()))));
        }
    }

    /// The fixture: 1.5 s of a test pattern with a tone, then 1 s of black and silence; three fragments.
    fn check(c: &Collect) {
        assert!((23..=27).contains(&c.pictures.len()), "{} pictures", c.pictures.len());
        assert!(c.pictures[0].0.abs() < 0.11);
        let bright = |t0: f64, t1: f64| c.pictures.iter().filter(|p| p.0 >= t0 && p.0 < t1).all(|p| p.1 > 40.0);
        let dark = |t0: f64, t1: f64| c.pictures.iter().filter(|p| p.0 >= t0 && p.0 < t1).all(|p| p.1 < 20.0);
        assert!(bright(0.0, 1.4) && dark(1.6, 2.5), "{:?}", c.pictures);
        // The first chunk is the encoder ramping in.
        let loud = c.sound.iter().filter(|s| s.0 > 0.05 && s.0 < 1.4).all(|s| s.1 > 0.05);
        let quiet = c.sound.iter().filter(|s| s.0 > 1.6).all(|s| s.1 < 0.01);
        assert!(loud && quiet, "{:?}", c.sound);
    }

    #[test]
    fn fragments_on_a_given_timeline() {
        let mut scan = Scan::new(&MEDIA[..1433], 48, 27, 10.0).unwrap();
        let mut c = Collect::default();
        for (off, len, start) in [(1433usize, 57712usize, 0.0), (59145, 31272, 1.001), (90417, 1488, 2.002)] {
            scan.feed(&Fragment { bytes: &MEDIA[off..off + len], start }, &mut c).unwrap();
        }
        check(&c);
    }

    #[test]
    fn color_once_a_second() {
        #[derive(Default)]
        struct Colors(Vec<(f64, usize, f32)>);
        impl Sink for Colors {
            fn picture(&mut self, _: f64, _: usize, _: usize, _: &[u8]) {}
            fn sound(&mut self, _: f64, _: u32, _: &[f32]) {}
            fn color(&mut self, pts: f64, size: usize, rgb: &[f32]) {
                assert_eq!(rgb.len(), 3 * size * size);
                self.0.push((pts, size, rgb.iter().sum::<f32>() / rgb.len() as f32));
            }
        }
        let mut c = Colors::default();
        let mut scan = Scan::new(&MEDIA[..1433], 48, 27, 10.0).unwrap().with_color(16);
        for (off, len, start) in [(1433usize, 57712usize, 0.0), (59145, 31272, 1.001), (90417, 1488, 2.002)] {
            scan.feed(&Fragment { bytes: &MEDIA[off..off + len], start }, &mut c).unwrap();
        }
        // 2.5 s: seconds 0, 1 and 2; the test pattern is colourful, the last second black.
        assert_eq!(c.0.len(), 3, "{:?}", c.0);
        assert!(c.0.iter().all(|x| x.1 == 16));
        assert!(c.0[0].2 > 0.2 && c.0[2].2 < 0.05, "{:?}", c.0);
    }

    #[test]
    fn a_whole_file() {
        let mut c = Collect::default();
        scan_file(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../fmp4/tests/fixtures/media.mp4"), 48, 27, 10.0, &mut c).unwrap();
        check(&c);
    }
}
