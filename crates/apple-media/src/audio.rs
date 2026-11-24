//! Broadcast sound to AAC with AudioToolbox: AC-3 frames split out of their PES, decoded to 5.1 PCM (a stereo
//! stretch upmixed into its front pair), and encoded as 5.1 AAC-LC.

use crate::sys::*;
use anyhow::{anyhow, ensure};
use std::ffi::c_void;
use std::ptr::{null, null_mut};

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

/// One packet for a converter's input callback (decoding: one compressed frame at a time).
struct Packet<'a> {
    data: Option<&'a [u8]>,
    desc: AudioStreamPacketDescription,
}

const NO_MORE_INPUT: OSStatus = 0x6e6d_7264;

extern "C" fn feed_packet(_c: AudioConverterRef, n: *mut u32, list: *mut AudioBufferList, desc: *mut *mut AudioStreamPacketDescription, user: *mut c_void) -> OSStatus {
    // SAFETY: user is the Packet passed to FillComplexBuffer, alive for that call.
    unsafe {
        let p = &mut *(user as *mut Packet);
        let Some(bytes) = p.data.take() else {
            *n = 0;
            return NO_MORE_INPUT;
        };
        *n = 1;
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0].mData = bytes.as_ptr() as *mut c_void;
        (*list).mBuffers[0].mDataByteSize = bytes.len() as u32;
        p.desc = AudioStreamPacketDescription { mStartOffset: 0, mVariableFramesInPacket: 0, mDataByteSize: bytes.len() as u32 };
        if !desc.is_null() {
            *desc = &raw mut p.desc;
        }
        0
    }
}

fn layout(tag: u32) -> AudioChannelLayout {
    AudioChannelLayout { mChannelLayoutTag: tag, mChannelBitmap: 0, mNumberChannelDescriptions: 0, mChannelDescriptions: [0; 20] }
}

fn pcm(channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: f64::from(RATE),
        mFormatID: u32::from_be_bytes(*b"lpcm"),
        mFormatFlags: 1 | 8,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 32,
        mReserved: 0,
    }
}

/// Decodes AC-3 to interleaved 5.1 float PCM. AudioToolbox maps whatever the frame carries (2.0, 3/2, 5.1) into
/// the 5.1 layout; a change of channels mid-stream gets a fresh decoder.
pub struct Ac3Decoder {
    converter: AudioConverterRef,
    channels: u32,
    out: Vec<f32>,
}

unsafe impl Send for Ac3Decoder {}

impl Ac3Decoder {
    pub fn new(channels: u32) -> anyhow::Result<Self> {
        Self::with_format(FORMAT_AC3, channels, AC3_FRAME_SAMPLES as u32)
    }

    /// An E-AC-3 decoder for frames of `samples` samples.
    pub fn eac3(channels: u32, samples: u32) -> anyhow::Result<Self> {
        Self::with_format(FORMAT_EAC3, channels, samples)
    }

    fn with_format(format: u32, channels: u32, samples: u32) -> anyhow::Result<Self> {
        let input = AudioStreamBasicDescription { mSampleRate: f64::from(RATE), mFormatID: format, mFramesPerPacket: samples, mChannelsPerFrame: channels, ..Default::default() };
        let output = pcm(CHANNELS as u32);
        let mut converter: AudioConverterRef = null();
        // SAFETY: valid descriptions; out-parameter; the layout struct is copied by the converter.
        unsafe {
            let st = AudioConverterNew(&raw const input, &raw const output, &raw mut converter);
            ensure!(st == 0, "AC-3/E-AC-3 decoder ({channels} channels): OSStatus {st}");
            let l = layout(LAYOUT_MPEG_5_1_A);
            let st = AudioConverterSetProperty(converter, PROP_OUTPUT_LAYOUT, 12, (&raw const l).cast());
            if st != 0 {
                AudioConverterDispose(converter);
                return Err(anyhow!("AC-3 decoder layout: OSStatus {st}"));
            }
        }
        Ok(Self { converter, channels, out: vec![0.0; AC3_FRAME_SAMPLES * 2 * CHANNELS] })
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Decode one AC-3 frame into interleaved 5.1 samples.
    pub fn decode(&mut self, frame: &[u8]) -> anyhow::Result<&[f32]> {
        let mut packet = Packet { data: Some(frame), desc: AudioStreamPacketDescription::default() };
        let capacity = (self.out.len() / CHANNELS) as u32;
        let mut frames = capacity;
        let mut list = AudioBufferList { mNumberBuffers: 1, mBuffers: [AudioBuffer { mNumberChannels: CHANNELS as u32, mDataByteSize: (self.out.len() * 4) as u32, mData: self.out.as_mut_ptr().cast() }] };
        // SAFETY: the callback's data lives for the call; the output list points at our buffer of `frames` frames.
        let st = unsafe { AudioConverterFillComplexBuffer(self.converter, feed_packet, (&raw mut packet).cast(), &raw mut frames, &raw mut list, null_mut()) };
        ensure!(st == 0 || st == NO_MORE_INPUT, "AC-3 decode: OSStatus {st}");
        Ok(&self.out[..frames.min(capacity) as usize * CHANNELS])
    }
}

impl Drop for Ac3Decoder {
    fn drop(&mut self) {
        // SAFETY: created in `new`.
        unsafe { AudioConverterDispose(self.converter) };
    }
}

/// The PCM waiting for the AAC encoder, and the encoder's input callback over it.
struct Fifo {
    samples: Vec<f32>,
    /// Interleaved samples handed out in the last callback (removed after the call).
    taken: usize,
    /// No more input is coming: hand over what is left even if short (the encoder pads the last packet).
    draining: bool,
}

extern "C" fn feed_pcm(_c: AudioConverterRef, n: *mut u32, list: *mut AudioBufferList, _desc: *mut *mut AudioStreamPacketDescription, user: *mut c_void) -> OSStatus {
    // SAFETY: user is the encoder's Fifo, alive for the FillComplexBuffer call; the buffer handed out stays
    // valid until that call returns (nothing else touches the FIFO meanwhile).
    unsafe {
        let f = &mut *(user as *mut Fifo);
        let available = (f.samples.len() - f.taken) / CHANNELS;
        let want = *n as usize;
        if available == 0 && f.draining {
            *n = 0;
            return 0; // end of stream: the encoder flushes what it holds
        }
        if available == 0 || (available < want && !f.draining) {
            *n = 0;
            return NO_MORE_INPUT;
        }
        let give = available.min(want);
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0].mNumberChannels = CHANNELS as u32;
        (*list).mBuffers[0].mData = f.samples.as_mut_ptr().add(f.taken).cast();
        (*list).mBuffers[0].mDataByteSize = (give * CHANNELS * 4) as u32;
        f.taken += give * CHANNELS;
        *n = give as u32;
        0
    }
}

/// 5.1 AAC-LC at a set bitrate, from interleaved 5.1 float PCM.
pub struct AacEncoder {
    converter: AudioConverterRef,
    fifo: Box<Fifo>,
    max_packet: usize,
}

unsafe impl Send for AacEncoder {}

impl AacEncoder {
    pub fn new(bitrate: u32) -> anyhow::Result<Self> {
        let input = pcm(CHANNELS as u32);
        let output = AudioStreamBasicDescription { mSampleRate: f64::from(RATE), mFormatID: u32::from_be_bytes(*b"aac "), mFramesPerPacket: AAC_PACKET_SAMPLES as u32, mChannelsPerFrame: CHANNELS as u32, ..Default::default() };
        let mut converter: AudioConverterRef = null();
        // SAFETY: as Ac3Decoder::new.
        let max_packet = unsafe {
            let st = AudioConverterNew(&raw const input, &raw const output, &raw mut converter);
            ensure!(st == 0, "AAC encoder: OSStatus {st}");
            let l = layout(LAYOUT_MPEG_5_1_A);
            let rate = bitrate;
            let st1 = AudioConverterSetProperty(converter, PROP_INPUT_LAYOUT, 12, (&raw const l).cast());
            let st2 = AudioConverterSetProperty(converter, PROP_ENCODE_BITRATE, 4, (&raw const rate).cast());
            let mut max = 0u32;
            let mut size = 4u32;
            let st3 = AudioConverterGetProperty(converter, PROP_MAX_OUTPUT_PACKET, &raw mut size, (&raw mut max).cast());
            if let Some(st) = [st1, st2, st3].into_iter().find(|&s| s != 0) {
                AudioConverterDispose(converter);
                return Err(anyhow!("AAC encoder settings: OSStatus {st}"));
            }
            max as usize
        };
        Ok(Self { converter, fifo: Box::new(Fifo { samples: Vec::new(), taken: 0, draining: false }), max_packet })
    }

    /// The ES descriptor (the `esds` body after version and flags), for the init segment.
    pub fn esds(&self) -> anyhow::Result<Vec<u8>> {
        let mut size = 0u32;
        // SAFETY: size query, then a buffer of that size.
        unsafe {
            let st = AudioConverterGetPropertyInfo(self.converter, PROP_COMPRESSION_COOKIE, &raw mut size, null_mut());
            ensure!(st == 0, "AAC encoder cookie size: OSStatus {st}");
            let mut cookie = vec![0u8; size as usize];
            let st = AudioConverterGetProperty(self.converter, PROP_COMPRESSION_COOKIE, &raw mut size, cookie.as_mut_ptr().cast());
            ensure!(st == 0, "AAC encoder cookie: OSStatus {st}");
            cookie.truncate(size as usize);
            Ok(cookie)
        }
    }

    /// Add interleaved 5.1 samples; returns the AAC packets that are ready (each `AAC_PACKET_SAMPLES` long).
    pub fn push(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Vec<u8>>> {
        self.fifo.samples.extend_from_slice(samples);
        self.drain(false)
    }

    /// The rest, the last packet padded with silence.
    pub fn finish(&mut self) -> anyhow::Result<Vec<Vec<u8>>> {
        self.fifo.draining = true;
        self.drain(true)
    }

    fn drain(&mut self, all: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        let mut buf = vec![0u8; self.max_packet];
        loop {
            if !all && self.fifo.samples.len() / CHANNELS < AAC_PACKET_SAMPLES {
                break;
            }
            let mut n = 1u32;
            let mut desc = AudioStreamPacketDescription::default();
            let mut list = AudioBufferList { mNumberBuffers: 1, mBuffers: [AudioBuffer { mNumberChannels: CHANNELS as u32, mDataByteSize: buf.len() as u32, mData: buf.as_mut_ptr().cast() }] };
            self.fifo.taken = 0;
            // SAFETY: the FIFO is boxed and outlives the call; the output list points at `buf`.
            let st = unsafe { AudioConverterFillComplexBuffer(self.converter, feed_pcm, (&raw mut *self.fifo).cast(), &raw mut n, &raw mut list, &raw mut desc) };
            let taken = self.fifo.taken;
            self.fifo.samples.drain(..taken);
            ensure!(st == 0 || st == NO_MORE_INPUT, "AAC encode: OSStatus {st}");
            if n == 0 {
                break;
            }
            packets.push(buf[..list.mBuffers[0].mDataByteSize as usize].to_vec());
        }
        Ok(packets)
    }
}

impl Drop for AacEncoder {
    fn drop(&mut self) {
        // SAFETY: created in `new`.
        unsafe { AudioConverterDispose(self.converter) };
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
