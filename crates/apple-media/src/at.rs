//! The thin layer over AudioToolbox's converters (through `objc2-audio-toolbox`): every call into them that Rust
//! cannot check is made here, once, with its justification, behind the safe `Converter`. The AC-3/E-AC-3 and AAC
//! decoders and the AAC encoder (`audio`, `analysis`) are then plain safe Rust.
//!
//! A converter pulls its input through a callback during `AudioConverterFillComplexBuffer`, on the calling
//! thread. Our two kinds of input, one compressed packet (decoding) or interleaved PCM (encoding), each have a
//! callback here; callers just pass slices.

use anyhow::{Result, ensure};
use objc2_audio_toolbox::{
    AudioConverterDispose, AudioConverterFillComplexBuffer, AudioConverterGetProperty, AudioConverterGetPropertyInfo, AudioConverterNew,
    AudioConverterPropertyID, AudioConverterRef, AudioConverterSetProperty,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioStreamPacketDescription, kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked,
    kAudioFormatLinearPCM,
};
use std::ffi::c_void;
use std::ptr::{NonNull, null_mut};

pub type OSStatus = i32;

/// What our input callbacks answer when they have nothing more for now (any non-zero status the converter hands
/// back to its caller).
const NOT_NOW: OSStatus = 0x6e6d_7264; // 'nmrd'

/// Interleaved 32-bit float PCM.
pub fn pcm_f32(rate: u32, channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: f64::from(rate),
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 32,
        mReserved: 0,
    }
}

/// A compressed format with `frames` samples a packet.
pub fn compressed(rate: u32, format: u32, frames: u32, channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        mSampleRate: f64::from(rate),
        mFormatID: format,
        mFormatFlags: 0,
        mBytesPerPacket: 0,
        mFramesPerPacket: frames,
        mBytesPerFrame: 0,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 0,
        mReserved: 0,
    }
}

/// An `AudioChannelLayout` that is just a layout tag (no channel descriptions), as the converter's layout
/// properties take it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayoutTag {
    tag: u32,
    bitmap: u32,
    descriptions: u32,
}

impl LayoutTag {
    pub fn new(tag: u32) -> Self {
        Self { tag, bitmap: 0, descriptions: 0 }
    }
}

/// An AudioToolbox converter, disposed on drop.
pub struct Converter(AudioConverterRef);

// SAFETY: an audio converter may be used from any thread, one at a time (every use takes `&mut self` or happens
// within one call).
unsafe impl Send for Converter {}

/// One compressed packet for `feed_packet`.
struct PacketInput<'a> {
    data: Option<&'a [u8]>,
    description: AudioStreamPacketDescription,
}

/// Interleaved PCM for `feed_pcm`.
struct PcmInput<'a> {
    pcm: &'a [f32],
    channels: usize,
    /// Samples (interleaved) handed out so far.
    taken: usize,
    /// No more input is coming: hand over what is left even if short (the encoder pads the last packet).
    draining: bool,
}

extern "C-unwind" fn feed_packet(
    _c: AudioConverterRef,
    packets: NonNull<u32>,
    data: NonNull<AudioBufferList>,
    description: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> OSStatus {
    // SAFETY: `user` is the PacketInput `decode_packet` passes, alive for the fill call this runs within; the
    // other pointers are the converter's own, valid for this call. The packet is only read by the converter.
    unsafe {
        let input = &mut *(user as *mut PacketInput);
        let Some(bytes) = input.data.take() else {
            *packets.as_ptr() = 0;
            return NOT_NOW;
        };
        *packets.as_ptr() = 1;
        let list = data.as_ptr();
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0].mData = bytes.as_ptr().cast_mut().cast();
        (*list).mBuffers[0].mDataByteSize = bytes.len() as u32;
        input.description = AudioStreamPacketDescription { mStartOffset: 0, mVariableFramesInPacket: 0, mDataByteSize: bytes.len() as u32 };
        if !description.is_null() {
            *description = &raw mut input.description;
        }
        0
    }
}

extern "C-unwind" fn feed_pcm(
    _c: AudioConverterRef,
    frames: NonNull<u32>,
    data: NonNull<AudioBufferList>,
    _description: *mut *mut AudioStreamPacketDescription,
    user: *mut c_void,
) -> OSStatus {
    // SAFETY: `user` is the PcmInput `encode_packet` passes, alive for the fill call this runs within; the samples
    // handed out stay valid (and unchanged) until that call returns, and the converter only reads them.
    unsafe {
        let input = &mut *(user as *mut PcmInput);
        let available = (input.pcm.len() - input.taken) / input.channels;
        let want = *frames.as_ptr() as usize;
        if available == 0 && input.draining {
            *frames.as_ptr() = 0;
            return 0; // end of stream: the encoder flushes what it holds
        }
        if available == 0 || (available < want && !input.draining) {
            *frames.as_ptr() = 0;
            return NOT_NOW;
        }
        let give = available.min(want);
        let list = data.as_ptr();
        (*list).mNumberBuffers = 1;
        (*list).mBuffers[0].mNumberChannels = input.channels as u32;
        (*list).mBuffers[0].mData = input.pcm.as_ptr().add(input.taken).cast_mut().cast();
        (*list).mBuffers[0].mDataByteSize = (give * input.channels * 4) as u32;
        input.taken += give * input.channels;
        *frames.as_ptr() = give as u32;
        0
    }
}

impl Converter {
    pub fn new(input: &AudioStreamBasicDescription, output: &AudioStreamBasicDescription) -> Result<Self> {
        let mut raw: AudioConverterRef = null_mut();
        // SAFETY: the descriptions are read during the call; out-parameter.
        let st = unsafe { AudioConverterNew(NonNull::from(input), NonNull::from(output), NonNull::from(&mut raw)) };
        ensure!(st == 0 && !raw.is_null(), "audio converter: OSStatus {st}");
        Ok(Self(raw))
    }

    /// Set a property to a plain value (a number, or a `#[repr(C)]` struct such as `LayoutTag`).
    pub fn set<T: Copy>(&mut self, property: AudioConverterPropertyID, value: &T) -> Result<()> {
        // SAFETY: a live converter; the value is `size_of::<T>()` readable bytes for the call (it is copied).
        let st = unsafe { AudioConverterSetProperty(self.0, property, size_of::<T>() as u32, NonNull::from(value).cast()) };
        ensure!(st == 0, "audio converter setting {property:#x}: OSStatus {st}");
        Ok(())
    }

    /// Set a property to some bytes (e.g. a decoder's magic cookie).
    pub fn set_bytes(&mut self, property: AudioConverterPropertyID, bytes: &[u8]) -> Result<()> {
        ensure!(!bytes.is_empty(), "audio converter setting {property:#x}: no data");
        // SAFETY: a live converter; the slice is readable for its length during the call (it is copied).
        let st = unsafe { AudioConverterSetProperty(self.0, property, bytes.len() as u32, NonNull::from(bytes).cast()) };
        ensure!(st == 0, "audio converter setting {property:#x}: OSStatus {st}");
        Ok(())
    }

    pub fn get_u32(&self, property: AudioConverterPropertyID) -> Result<u32> {
        let (mut value, mut size) = (0u32, 4u32);
        // SAFETY: a live converter; a 4-byte out-buffer and its size.
        let st = unsafe { AudioConverterGetProperty(self.0, property, NonNull::from(&mut size), NonNull::from(&mut value).cast()) };
        ensure!(st == 0, "audio converter property {property:#x}: OSStatus {st}");
        Ok(value)
    }

    /// A property of variable size (e.g. an encoder's magic cookie).
    pub fn get_bytes(&self, property: AudioConverterPropertyID) -> Result<Vec<u8>> {
        let mut size = 0u32;
        // SAFETY: a live converter; a size out-parameter (writability not asked).
        let st = unsafe { AudioConverterGetPropertyInfo(self.0, property, &mut size, null_mut()) };
        ensure!(st == 0 && size > 0, "audio converter property {property:#x} size: OSStatus {st}");
        let mut bytes = vec![0u8; size as usize];
        // SAFETY: a live converter; an out-buffer of `size` bytes and its size.
        let st = unsafe { AudioConverterGetProperty(self.0, property, NonNull::from(&mut size), NonNull::from(bytes.as_mut_slice()).cast()) };
        ensure!(st == 0, "audio converter property {property:#x}: OSStatus {st}");
        bytes.truncate(size as usize);
        Ok(bytes)
    }

    /// Decode one compressed packet into `out` (interleaved, `channels` samples a frame). Returns the frames
    /// written (a decoder may hold some back at first).
    pub fn decode_packet(&mut self, packet: &[u8], out: &mut [f32], channels: usize) -> Result<usize> {
        let capacity = out.len() / channels;
        let mut frames = capacity as u32;
        let mut list = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer { mNumberChannels: channels as u32, mDataByteSize: (out.len() * 4) as u32, mData: out.as_mut_ptr().cast() }],
        };
        let mut input =
            PacketInput { data: Some(packet), description: AudioStreamPacketDescription { mStartOffset: 0, mVariableFramesInPacket: 0, mDataByteSize: 0 } };
        // SAFETY: a live converter; the callback's input and the output list (over `out`, `frames` frames) live
        // across the call, which is the only place the callback runs.
        let st = unsafe {
            AudioConverterFillComplexBuffer(
                self.0,
                Some(feed_packet),
                (&raw mut input).cast(),
                NonNull::from(&mut frames),
                NonNull::from(&mut list),
                null_mut(),
            )
        };
        ensure!(st == 0 || st == NOT_NOW, "audio decode: OSStatus {st}");
        Ok((frames as usize).min(capacity))
    }

    /// Encode from `pcm` (interleaved, `channels` samples a frame) into one packet in `out`. Returns how many
    /// samples (interleaved) were used, and the packet's length if one came out. `draining`: `pcm` is the rest of
    /// the stream (a short final packet is padded).
    pub fn encode_packet(&mut self, pcm: &[f32], channels: usize, draining: bool, out: &mut [u8]) -> Result<(usize, Option<usize>)> {
        let mut packets = 1u32;
        let mut description = AudioStreamPacketDescription { mStartOffset: 0, mVariableFramesInPacket: 0, mDataByteSize: 0 };
        let mut list = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer { mNumberChannels: channels as u32, mDataByteSize: out.len() as u32, mData: out.as_mut_ptr().cast() }],
        };
        let mut input = PcmInput { pcm, channels, taken: 0, draining };
        // SAFETY: a live converter; the callback's input, the output list (over `out`) and the packet description
        // live across the call, which is the only place the callback runs.
        let st = unsafe {
            AudioConverterFillComplexBuffer(
                self.0,
                Some(feed_pcm),
                (&raw mut input).cast(),
                NonNull::from(&mut packets),
                NonNull::from(&mut list),
                &mut description,
            )
        };
        ensure!(st == 0 || st == NOT_NOW, "audio encode: OSStatus {st}");
        let bytes = list.mBuffers[0].mDataByteSize as usize;
        Ok((input.taken, (packets > 0).then_some(bytes.min(out.len()))))
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: created in `new`, disposed once.
        unsafe { AudioConverterDispose(self.0) };
    }
}
