//! Decoding for analysis: HEVC pictures shrunk to small luma images (VideoToolbox, in hardware) and AAC folded
//! to one channel (AudioToolbox). Safe code over `vt` and `at`.

use anyhow::{anyhow, ensure};

use crate::at::{Converter, compressed, pcm_f32};
use crate::video::PixelBuffer;
use crate::vt::{self, Decompression, PIXEL_FORMAT_420F, PIXEL_FORMAT_420V, keys, time};
use objc2_audio_toolbox::kAudioConverterDecompressionMagicCookie;
use objc2_core_audio_types::kAudioFormatMPEG4AAC;
use objc2_core_foundation::{CFNumber, CFRetained};
use objc2_core_media::CMFormatDescription;
use objc2_core_video::CVPixelBufferGetPixelFormatType;

/// A decoded picture, reduced to what the break detector looks at: its time and its brightness (full-range
/// luma), `width`×`height`.
pub struct Picture {
    pub pts: i64,
    pub luma: Vec<u8>,
}

/// A decoded picture's luma, shrunk to `dw`×`dh` and stretched to full range.
fn luma(picture: &PixelBuffer, dw: usize, dh: usize) -> anyhow::Result<Vec<u8>> {
    let format = CVPixelBufferGetPixelFormatType(picture.cv());
    ensure!(format == PIXEL_FORMAT_420V || format == PIXEL_FORMAT_420F, "unexpected pixel format {format:#x}");
    vt::read_plane(picture.cv(), 0, |bytes, w, h, stride| shrink(bytes, w, h, stride, dw, dh, format == PIXEL_FORMAT_420V))
        .ok_or_else(|| anyhow!("cannot read the picture's luma"))
}

/// Average `src` (w×h, `stride` bytes a row) down to `dw`×`dh`, each output pixel the mean of the source pixels
/// it covers, and stretch video-range luma (16–235) to full range, as ffmpeg's `format=gray` does.
fn shrink(src: &[u8], w: usize, h: usize, stride: usize, dw: usize, dh: usize, video_range: bool) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh];
    for oy in 0..dh {
        let (y0, y1) = (oy * h / dh, ((oy + 1) * h / dh).max(oy * h / dh + 1));
        for ox in 0..dw {
            let (x0, x1) = (ox * w / dw, ((ox + 1) * w / dw).max(ox * w / dw + 1));
            let mut sum = 0u32;
            for y in y0..y1 {
                sum += src[y * stride + x0..y * stride + x1].iter().map(|&p| u32::from(p)).sum::<u32>();
            }
            let mean = sum / ((x1 - x0) * (y1 - y0)) as u32;
            out[oy * dw + ox] = if video_range { ((mean.saturating_sub(16) * 255 + 109) / 219).min(255) as u8 } else { mean as u8 };
        }
    }
    out
}

/// An HEVC decoder (hardware where the Mac has one) that hands back pictures shrunk to the scan's size.
pub struct HevcDecoder {
    session: Decompression,
    format: CFRetained<CMFormatDescription>,
    timescale: u32,
    width: usize,
    height: usize,
    /// Which pictures to keep (by presentation time, in the track's timescale), decided by the caller.
    want: Box<dyn FnMut(i64) -> bool + Send>,
}

impl HevcDecoder {
    /// A decoder for a track's parameter sets (VPS, SPS, PPS), handing back `width`×`height` pictures for the
    /// presentation times `want` accepts.
    pub fn new(
        parameter_sets: &[Vec<u8>],
        nal_length_size: u8,
        timescale: u32,
        width: usize,
        height: usize,
        want: Box<dyn FnMut(i64) -> bool + Send>,
    ) -> anyhow::Result<Self> {
        let format = vt::hevc_format(parameter_sets, nal_length_size)?;
        let pixel = CFNumber::new_i32(PIXEL_FORMAT_420V as i32);
        let attributes = vt::dict(&[(keys().pixel_format, &pixel)]);
        let session = Decompression::new(&format, Some(attributes.as_ref()))?;
        Ok(Self { session, format, timescale, width, height, want })
    }

    /// Decode one coded picture (length-prefixed NAL units) with its times; returns the pictures that came out.
    pub fn decode(&mut self, data: &[u8], dts: i64, pts: i64) -> anyhow::Result<Vec<Picture>> {
        let sample = vt::sample(data, &self.format, time(pts, self.timescale), time(dts, self.timescale))?;
        self.session.decode(&sample)?;
        self.take()
    }

    /// Pictures still inside the decoder.
    pub fn flush(&mut self) -> Vec<Picture> {
        self.session.wait();
        self.take().unwrap_or_default()
    }

    /// The pictures the callback has sent that the caller wants, shrunk; or the first error.
    fn take(&mut self) -> anyhow::Result<Vec<Picture>> {
        let mut out = Vec::new();
        for msg in self.session.drain() {
            let (pts, picture) = msg.map_err(|e| anyhow!("VideoToolbox picture: OSStatus {e}"))?;
            if (self.want)(pts) {
                out.push(Picture { pts, luma: luma(&picture, self.width, self.height)? });
            }
        }
        Ok(out)
    }
}

// ---- AAC -------------------------------------------------------------------------------------------------

const AAC_FRAMES_PER_PACKET: u32 = 1024;

/// An AAC decoder whose output is folded to one channel for the silence detector: each sample is the loudest of
/// the channels at that instant (a quiet moment is quiet on every channel; averaging would hide a voice that is
/// only in the centre channel).
pub struct AacDecoder {
    converter: Converter,
    channels: usize,
    buffer: Vec<f32>,
}

impl AacDecoder {
    pub fn new(channels: u16, sample_rate: u32, esds: &[u8]) -> anyhow::Result<Self> {
        let input = compressed(sample_rate, kAudioFormatMPEG4AAC, AAC_FRAMES_PER_PACKET, u32::from(channels));
        let mut converter = Converter::new(&input, &pcm_f32(sample_rate, u32::from(channels))).map_err(|e| e.context("AAC decoder"))?;
        converter.set_bytes(kAudioConverterDecompressionMagicCookie, esds).map_err(|e| e.context("AAC decoder config"))?;
        Ok(Self { converter, channels: usize::from(channels), buffer: vec![0.0; AAC_FRAMES_PER_PACKET as usize * 2 * usize::from(channels)] })
    }

    /// Decode one packet; returns the mono samples that came out (the decoder may hold some back at first).
    pub fn decode(&mut self, packet: &[u8]) -> anyhow::Result<Vec<f32>> {
        let frames = self.converter.decode_packet(packet, &mut self.buffer, self.channels)?;
        Ok(self.buffer[..frames * self.channels].chunks_exact(self.channels).map(|f| f.iter().fold(0f32, |m, s| m.max(s.abs()))).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrinking_averages_and_stretches_video_range() {
        // 4×2 video-range luma: black (16) on the left, white (235) on the right.
        let src = [16, 16, 235, 235, 16, 16, 235, 235];
        assert_eq!(shrink(&src, 4, 2, 4, 2, 1, true), vec![0, 255]);
        assert_eq!(shrink(&src, 4, 2, 4, 1, 1, false), vec![125]);
    }
}
