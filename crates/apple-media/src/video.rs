//! Full-frame video with VideoToolbox: decoding broadcast video into pixel buffers (MPEG-2 has no hardware decoder
//! on Apple Silicon, but VideoToolbox's software one works) and encoding them to HEVC in hardware. Safe code over
//! the sessions in `vt`.

use crate::vt::{self, CODEC_HEVC, Compression, Decompression, PIXEL_FORMAT_420V, keys, no, time, yes};
use anyhow::anyhow;
use objc2_core_foundation::{CFNumber, CFRetained};
use objc2_core_media::CMFormatDescription;
use objc2_core_video::{CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetWidth};

pub use crate::vt::CODEC_MPEG2;

/// A decoded picture (a retained CVPixelBuffer), released on drop.
#[derive(Clone)]
pub struct PixelBuffer(CFRetained<CVPixelBuffer>);

// SAFETY: CoreVideo pixel buffers are reference-counted objects that may be passed between threads; we only read
// them once decoded, and writers (the deinterlacer) hold the only reference to a buffer they fill.
unsafe impl Send for PixelBuffer {}

impl PixelBuffer {
    pub fn new(buffer: CFRetained<CVPixelBuffer>) -> Self {
        Self(buffer)
    }
    pub fn cv(&self) -> &CVPixelBuffer {
        &self.0
    }
    pub fn size(&self) -> (usize, usize) {
        (CVPixelBufferGetWidth(&self.0), CVPixelBufferGetHeight(&self.0))
    }
}

/// A decoded picture and its presentation time (in the timescale it was given in).
pub struct Decoded {
    pub pts: i64,
    pub picture: PixelBuffer,
}

/// Decodes broadcast MPEG-2 (or another codec VideoToolbox knows by its four-character code) into pixel buffers.
pub struct Decoder {
    session: Decompression,
    format: CFRetained<CMFormatDescription>,
    timescale: u32,
}

impl Decoder {
    pub fn new(codec: u32, width: u32, height: u32, timescale: u32) -> anyhow::Result<Self> {
        Self::with_format(vt::video_format(codec, width, height)?, timescale)
    }

    /// An H.264 decoder for these parameter sets (NAL units without start codes); samples are then given as
    /// length-prefixed NAL units (4-byte lengths).
    pub fn h264(sps: &[u8], pps: &[u8], timescale: u32) -> anyhow::Result<Self> {
        Self::with_format(vt::h264_format(sps, pps)?, timescale)
    }

    fn with_format(format: CFRetained<CMFormatDescription>, timescale: u32) -> anyhow::Result<Self> {
        // 420v: what the encoder takes.
        let pixel = CFNumber::new_i32(PIXEL_FORMAT_420V as i32);
        let attributes = vt::dict(&[(keys().pixel_format, &pixel)]);
        let session = Decompression::new(&format, Some(attributes.as_ref()))?;
        Ok(Self { session, format, timescale })
    }

    /// Decode one coded picture; returns the pictures that came out (in decode order: sort by `pts`).
    pub fn decode(&mut self, data: &[u8], dts: i64, pts: i64) -> anyhow::Result<Vec<Decoded>> {
        let sample = vt::sample(data, &self.format, time(pts, self.timescale), time(dts, self.timescale))?;
        self.session.decode(&sample).map_err(|e| anyhow!("video decode: {e:#}"))?;
        self.session.drain().into_iter().map(|p| p.map(|(pts, picture)| Decoded { pts, picture }).map_err(|e| anyhow!("video decode: OSStatus {e}"))).collect()
    }

    pub fn flush(&mut self) -> Vec<Decoded> {
        self.session.wait();
        self.session.drain().into_iter().filter_map(Result::ok).map(|(pts, picture)| Decoded { pts, picture }).collect()
    }
}

/// An encoded picture: times in the encoder's timescale, data as length-prefixed NAL units.
pub struct Encoded {
    pub pts: i64,
    pub dts: i64,
    pub data: Vec<u8>,
    pub keyframe: bool,
}

/// How hard the encoder squeezes: an average bitrate, or a constant quality (0.0-1.0) that spends bits where the
/// picture needs them, so a static newsroom is small and a football game keeps its detail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    Bitrate(i64),
    Quality(f32),
}

/// HEVC in hardware: no B-frames (decode order is presentation order), a keyframe whenever asked and at least
/// every `max_key_secs`.
pub struct Encoder {
    session: Compression,
    /// The `hvcC` box body, from the first keyframe.
    hvcc: Option<Vec<u8>>,
    timescale: u32,
}

impl Encoder {
    pub fn new(width: u32, height: u32, rate: Rate, fps: f64, max_key_secs: f64, realtime: bool, timescale: u32) -> anyhow::Result<Self> {
        let k = keys();
        let spec = vt::dict(&[(k.hardware_encoder, yes())]);
        let mut session = Compression::new(width, height, CODEC_HEVC, Some(spec.as_ref()))?;
        session.set(k.real_time, if realtime { yes() } else { no() })?;
        session.set(k.profile_level, k.hevc_main_auto)?;
        session.set(k.allow_reordering, no())?;
        match rate {
            Rate::Bitrate(bps) => session.set(k.average_bit_rate, &CFNumber::new_i64(bps))?,
            Rate::Quality(q) => session.set(k.quality, &CFNumber::new_f64(f64::from(q.clamp(0.0, 1.0))))?,
        }
        session.set(k.max_key_interval, &CFNumber::new_f64(max_key_secs))?;
        session.set(k.expected_frame_rate, &CFNumber::new_f64(fps))?;
        session.prepare();
        Ok(Self { session, hvcc: None, timescale })
    }

    /// Encode a picture shown at `pts` for `duration`; `key` forces a keyframe. Returns what came out.
    pub fn encode(&mut self, picture: &PixelBuffer, pts: i64, duration: i64, key: bool) -> anyhow::Result<Vec<Encoded>> {
        // Made when needed (a keyframe every 2 s): CoreFoundation dictionaries can't be kept in a Send type.
        let force_key = key.then(|| vt::dict(&[(keys().force_key_frame, yes())]));
        self.session.encode(picture.cv(), time(pts, self.timescale), time(duration, self.timescale), force_key.as_deref().map(AsRef::as_ref))?;
        self.take()
    }

    /// What has come out: the encoded pictures, or the first error.
    fn take(&mut self) -> anyhow::Result<Vec<Encoded>> {
        let mut out = Vec::new();
        for frame in self.session.drain() {
            let s = frame.map_err(|e| anyhow!("HEVC encode: OSStatus {e}"))?;
            if self.hvcc.is_none() {
                self.hvcc = s.hvcc;
            }
            out.push(Encoded { pts: s.pts, dts: s.dts, data: s.data, keyframe: s.keyframe });
        }
        Ok(out)
    }

    /// Everything still in the encoder.
    pub fn finish(&mut self) -> anyhow::Result<Vec<Encoded>> {
        self.session.complete()?;
        self.take()
    }

    /// The `hvcC` box body, once the first picture is out.
    pub fn hvcc(&self) -> Option<&[u8]> {
        self.hvcc.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpegts::{Demuxer, Unit};

    /// Decode a whole broadcast capture (not in the repository: set `DVR_TS_SAMPLE` to a .ts file).
    #[test]
    fn decodes_broadcast_mpeg2() {
        let Ok(path) = std::env::var("DVR_TS_SAMPLE") else { return };
        let bytes = std::fs::read(path).unwrap();
        let mut d = Demuxer::new();
        let mut units = d.push(&bytes);
        units.extend(d.flush());
        let mut dec: Option<Decoder> = None;
        let mut pictures = Vec::new();
        let mut fed = 0;
        for u in &units {
            if let Unit::Video { pts, dts, data, .. } = u {
                if dec.is_none() {
                    let Some((w, h)) = mpegts::mpeg2::size(data) else { continue };
                    dec = Some(Decoder::new(vt::CODEC_MPEG2, w, h, 90_000).unwrap());
                    let _ = &dec;
                }
                pictures.extend(dec.as_mut().unwrap().decode(data, *dts, *pts).unwrap());
                fed += 1;
            }
        }
        pictures.extend(dec.as_mut().unwrap().flush());
        let size = pictures[0].picture.size();
        eprintln!("fed {fed} pictures, got {} of {size:?}", pictures.len());
        assert!(pictures.len() + 3 >= fed && !pictures.is_empty());

        // And encode them to HEVC, a keyframe every 2 s, at an average bitrate and at constant quality.
        pictures.sort_by_key(|p| p.pts);
        for rate in [Rate::Bitrate(4_000_000), Rate::Quality(0.6)] {
            let mut enc = Encoder::new(size.0 as u32, size.1 as u32, rate, 30.0, 2.0, false, 90_000).unwrap();
            let t0 = std::time::Instant::now();
            let origin = pictures[0].pts;
            let mut out = Vec::new();
            let mut next_key = 0;
            for (i, p) in pictures.iter().enumerate() {
                let pts = p.pts - origin;
                let dur = pictures.get(i + 1).map_or(1500, |n| n.pts - p.pts);
                let key = pts >= next_key;
                if key {
                    next_key += 180_000;
                }
                out.extend(enc.encode(&p.picture, pts, dur, key).unwrap());
            }
            out.extend(enc.finish().unwrap());
            let bytes: usize = out.iter().map(|e| e.data.len()).sum();
            let keys = out.iter().filter(|e| e.keyframe).count();
            eprintln!("{rate:?}: encoded {} pictures, {} keyframes, {:.1} Mbit/s, in {:?}", out.len(), keys, bytes as f64 * 8.0 / 10.0 / 1e6, t0.elapsed());
            assert_eq!(out.len(), pictures.len());
            assert!(enc.hvcc().is_some_and(|h| h.len() > 23));
            assert!(out.windows(2).all(|w| w[1].dts > w[0].dts && w[0].pts == w[0].dts), "no reordering");
        }
    }
}
