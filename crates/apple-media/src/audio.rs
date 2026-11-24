//! Broadcast sound to AAC with AudioToolbox: AC-3 frames split out of their PES, decoded to 5.1 PCM (a stereo
//! stretch upmixed into its front pair), and encoded as 5.1 AAC-LC.

use crate::at::{Converter, LayoutTag, compressed, pcm_f32};
use objc2_audio_toolbox::{
    kAudioConverterCompressionMagicCookie, kAudioConverterEncodeBitRate, kAudioConverterInputChannelLayout, kAudioConverterOutputChannelLayout,
    kAudioConverterPropertyMaximumOutputPacketSize,
};
use objc2_core_audio_types::{kAudioChannelLayoutTag_MPEG_5_1_A, kAudioFormatAC3, kAudioFormatEnhancedAC3, kAudioFormatMPEG4AAC};

/// Output channels: always 5.1 (L R C LFE Ls Rs), so the AAC track never changes shape mid-recording.
pub const CHANNELS: usize = 6;
pub const RATE: u32 = 48_000;
/// Samples in one AC-3 frame and in one AAC packet.
pub const AC3_FRAME_SAMPLES: usize = 1536;
pub const AAC_PACKET_SAMPLES: usize = 1024;
/// How far the AAC encoder's output runs behind its input, as a player decodes it from the file (no edit list):
/// the encoder's 2112 priming samples less the one packet a decoder itself holds back. Measured with a test
/// stream of flashes and beeps at the same instants: with this, sound and picture land within a few
/// milliseconds of each other (as with the ffmpeg pipeline); with 2112 the sound came ~21 ms early.
pub const AAC_PRIMING: usize = 2112 - 1024;

/// AC-3 frame lengths in 16-bit words at 48 kHz, by `frmsizecod / 2`.
const AC3_WORDS_48K: [usize; 19] = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280];

/// The AC-3 frames in a PES payload: (offset, length, channels including LFE). Stops at anything it can't read.
pub fn ac3_frames(data: &[u8]) -> Vec<(usize, usize, u32)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 7 <= data.len() {
        if data[at] != 0x0b || data[at + 1] != 0x77 {
            at += 1;
            continue;
        }
        let fscod = data[at + 4] >> 6;
        let code = usize::from(data[at + 4] & 0x3f);
        if fscod != 0 || code / 2 >= AC3_WORDS_48K.len() {
            break; // only 48 kHz (ATSC)
        }
        let len = AC3_WORDS_48K[code / 2] * 2;
        if at + len > data.len() {
            break;
        }
        // bsi: bsid(5) bsmod(3) acmod(3), then optional mix levels, then lfeon.
        let bits = u16::from_be_bytes([data[at + 6], data[at + 7.min(len - 1)]]);
        let acmod = (data[at + 6] >> 5) & 7;
        let mut pos = 3; // bits used after acmod, counted from bit 13 of `bits`
        if acmod & 1 != 0 && acmod != 1 {
            pos += 2;
        }
        if acmod & 4 != 0 {
            pos += 2;
        }
        if acmod == 2 {
            pos += 2;
        }
        let lfe = (bits >> (15 - pos)) & 1;
        let channels = [2, 1, 2, 3, 3, 4, 4, 5][usize::from(acmod)] + u32::from(lfe);
        out.push((at, len, channels));
        at += len;
    }
    out
}

/// The independent E-AC-3 frames in a PES payload: (offset, length, channels including LFE, samples). Dependent
/// substreams (the extra channels of 7.1) are skipped: the main 5.1 is what is kept.
pub fn eac3_frames(data: &[u8]) -> Vec<(usize, usize, u32, usize)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 6 <= data.len() {
        if data[at] != 0x0b || data[at + 1] != 0x77 {
            at += 1;
            continue;
        }
        let (b2, b3, b4) = (data[at + 2], data[at + 3], data[at + 4]);
        let len = ((usize::from(b2 & 7) << 8) | usize::from(b3)) * 2 + 2;
        if at + len > data.len() || len < 6 {
            break;
        }
        let dependent = b2 >> 6 == 1;
        let fscod = b4 >> 6;
        let blocks = if fscod == 3 { 6 } else { [1, 2, 3, 6][usize::from((b4 >> 4) & 3)] };
        let acmod = (b4 >> 1) & 7;
        let channels = [2, 1, 2, 3, 3, 4, 4, 5][usize::from(acmod)] + u32::from(b4 & 1);
        if !dependent && fscod != 3 {
            out.push((at, len, channels, 256 * blocks));
        }
        at += len;
    }
    out
}

/// Decodes AC-3 to interleaved 5.1 float PCM. AudioToolbox maps whatever the frame carries (2.0, 3/2, 5.1) into
/// the 5.1 layout; a change of channels mid-stream gets a fresh decoder.
pub struct Ac3Decoder {
    converter: Converter,
    channels: u32,
    out: Vec<f32>,
}

impl Ac3Decoder {
    pub fn new(channels: u32) -> anyhow::Result<Self> {
        Self::with_format(kAudioFormatAC3, channels, AC3_FRAME_SAMPLES as u32)
    }

    /// An E-AC-3 decoder for frames of `samples` samples.
    pub fn eac3(channels: u32, samples: u32) -> anyhow::Result<Self> {
        Self::with_format(kAudioFormatEnhancedAC3, channels, samples)
    }

    fn with_format(format: u32, channels: u32, samples: u32) -> anyhow::Result<Self> {
        let mut converter = Converter::new(&compressed(RATE, format, samples, channels), &pcm_f32(RATE, CHANNELS as u32))
            .map_err(|e| e.context(format!("AC-3/E-AC-3 decoder ({channels} channels)")))?;
        converter.set(kAudioConverterOutputChannelLayout, &LayoutTag::new(kAudioChannelLayoutTag_MPEG_5_1_A))?;
        Ok(Self { converter, channels, out: vec![0.0; AC3_FRAME_SAMPLES * 2 * CHANNELS] })
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Decode one AC-3 frame into interleaved 5.1 samples.
    pub fn decode(&mut self, frame: &[u8]) -> anyhow::Result<&[f32]> {
        let frames = self.converter.decode_packet(frame, &mut self.out, CHANNELS)?;
        Ok(&self.out[..frames * CHANNELS])
    }
}

/// 5.1 AAC-LC at a set bitrate, from interleaved 5.1 float PCM.
pub struct AacEncoder {
    converter: Converter,
    /// PCM waiting for the encoder (interleaved).
    samples: Vec<f32>,
    max_packet: usize,
}

impl AacEncoder {
    pub fn new(bitrate: u32) -> anyhow::Result<Self> {
        let output = compressed(RATE, kAudioFormatMPEG4AAC, AAC_PACKET_SAMPLES as u32, CHANNELS as u32);
        let mut converter = Converter::new(&pcm_f32(RATE, CHANNELS as u32), &output).map_err(|e| e.context("AAC encoder"))?;
        converter.set(kAudioConverterInputChannelLayout, &LayoutTag::new(kAudioChannelLayoutTag_MPEG_5_1_A))?;
        converter.set(kAudioConverterEncodeBitRate, &bitrate)?;
        let max_packet = converter.get_u32(kAudioConverterPropertyMaximumOutputPacketSize)? as usize;
        Ok(Self { converter, samples: Vec::new(), max_packet })
    }

    /// The ES descriptor (the `esds` body after version and flags), for the init segment.
    pub fn esds(&self) -> anyhow::Result<Vec<u8>> {
        self.converter.get_bytes(kAudioConverterCompressionMagicCookie)
    }

    /// Add interleaved 5.1 samples; returns the AAC packets that are ready (each `AAC_PACKET_SAMPLES` long).
    pub fn push(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Vec<u8>>> {
        self.samples.extend_from_slice(samples);
        self.drain(false)
    }

    /// The rest, the last packet padded with silence.
    pub fn finish(&mut self) -> anyhow::Result<Vec<Vec<u8>>> {
        self.drain(true)
    }

    fn drain(&mut self, all: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        let mut buf = vec![0u8; self.max_packet];
        loop {
            if !all && self.samples.len() / CHANNELS < AAC_PACKET_SAMPLES {
                break;
            }
            let (taken, packet) = self.converter.encode_packet(&self.samples, CHANNELS, all, &mut buf)?;
            self.samples.drain(..taken);
            let Some(len) = packet else { break };
            packets.push(buf[..len].to_vec());
        }
        Ok(packets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mpegts::{Demuxer, Unit};

    const CBS: &[u8] = include_bytes!("../../mpegts/tests/fixtures/cbs-head.ts");

    #[test]
    fn splits_the_ac3_frames_of_a_pes() {
        let mut d = Demuxer::new();
        let mut units = d.push(CBS);
        units.extend(d.flush()); // the slice ends inside its one audio PES
        let Some(Unit::Audio { data, .. }) = units.iter().find(|u| matches!(u, Unit::Audio { .. })) else { panic!("no audio") };
        let frames = ac3_frames(data);
        assert_eq!(frames.len(), 4, "five frames a PES, the last cut off by the end of the slice");
        assert!(frames.iter().all(|&(_, len, ch)| len == 1536 && ch == 6), "{frames:?}");
    }

    #[test]
    fn decodes_ac3_and_encodes_5_1_aac() {
        let mut d = Demuxer::new();
        let mut units = d.push(CBS);
        units.extend(d.flush()); // the slice ends inside its one audio PES
        let Some(Unit::Audio { data, .. }) = units.iter().find(|u| matches!(u, Unit::Audio { .. })) else { panic!("no audio") };
        let mut dec = Ac3Decoder::new(6).unwrap();
        let mut enc = AacEncoder::new(384_000).unwrap();
        let mut pcm = Vec::new();
        for (off, len, _) in ac3_frames(data) {
            pcm.extend_from_slice(dec.decode(&data[off..off + len]).unwrap());
        }
        // The decoder holds back a few milliseconds (a constant latency).
        let samples = pcm.len() / CHANNELS;
        assert!((3 * AC3_FRAME_SAMPLES..=4 * AC3_FRAME_SAMPLES).contains(&samples), "{samples}");
        // The dialogue of a news show is in the centre channel (index 2), and the LFE (3) is quiet.
        let rms = |c: usize| (pcm.iter().skip(c).step_by(CHANNELS).map(|s| s * s).sum::<f32>() / (pcm.len() / CHANNELS) as f32).sqrt();
        let levels: Vec<f32> = (0..CHANNELS).map(rms).collect();
        assert!(levels[2] > levels[3], "{levels:?}");
        let mut packets = enc.push(&pcm).unwrap();
        packets.extend(enc.finish().unwrap());
        // 6144 samples plus the encoder's priming, in 1024-sample packets.
        assert!((7..=9).contains(&packets.len()), "{}", packets.len());
        let esds = enc.esds().unwrap();
        assert_eq!(esds[0], 0x03, "an ES descriptor");
    }
}
