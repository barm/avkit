//! Decoding for analysis: HEVC pictures shrunk to small luma images (VideoToolbox, in hardware) and AAC folded
//! to one channel (AudioToolbox). Plain FFI to their C APIs, kept to the few calls these need; every `unsafe` block is here.


use anyhow::{anyhow, ensure};
use std::ffi::c_void;
use std::ptr::{null, null_mut};

use crate::sys::*;

/// 8-bit 4:2:0, Y in its own plane, video range (`420v`).
const PIXEL_FORMAT_420V: u32 = 0x3432_3076;
const PIXEL_FORMAT_420F: u32 = 0x3432_3066;
const CF_NUMBER_SINT32: isize = 3;

/// A decoded picture, reduced to what the break detector looks at: its time and its brightness (full-range
/// luma), `width`×`height`.
pub struct Picture {
    pub pts: i64,
    pub luma: Vec<u8>,
}

/// Where VideoToolbox's callback leaves its pictures (it runs during `decode`, on the decoding thread).
struct Sink {
    width: usize,
    height: usize,
    /// Which pictures to keep (by presentation time, in the track's timescale), decided by the caller.
    want: Box<dyn FnMut(i64) -> bool + Send>,
    out: Vec<Picture>,
    error: Option<OSStatus>,
}

/// An HEVC decoder (hardware where the Mac has one) that hands back pictures shrunk to the scan's size.
pub struct HevcDecoder {
    session: VTDecompressionSessionRef,
    format: CMFormatDescriptionRef,
    sink: Box<Sink>,
    timescale: u32,
}

// The session is used from one thread at a time (the scan's task); VideoToolbox sessions may move between threads.
unsafe impl Send for HevcDecoder {}

extern "C" fn on_picture(refcon: *mut c_void, _frame: *mut c_void, status: OSStatus, _flags: u32, image: CVImageBufferRef, pts: CMTime, _dur: CMTime) {
    // SAFETY: refcon is the decoder's boxed Sink, alive for the session's life; VideoToolbox calls this while
    // `decode` runs (synchronous decoding) or during the final wait, never concurrently with another use.
    let sink = unsafe { &mut *(refcon as *mut Sink) };
    if status != 0 || image.is_null() {
        if status != 0 {
            sink.error = Some(status);
        }
        return;
    }
    if !(sink.want)(pts.value) {
        return;
    }
    // SAFETY: image is a CVPixelBuffer valid for this call; it is locked read-only while its luma plane is read.
    unsafe {
        let format = CVPixelBufferGetPixelFormatType(image);
        if format != PIXEL_FORMAT_420V && format != PIXEL_FORMAT_420F {
            sink.error = Some(-1);
            return;
        }
        if CVPixelBufferLockBaseAddress(image, 1) != 0 {
            sink.error = Some(-2);
            return;
        }
        let base = CVPixelBufferGetBaseAddressOfPlane(image, 0);
        let stride = CVPixelBufferGetBytesPerRowOfPlane(image, 0);
        let (w, h) = (CVPixelBufferGetWidthOfPlane(image, 0), CVPixelBufferGetHeightOfPlane(image, 0));
        let plane = std::slice::from_raw_parts(base, stride * h);
        let luma = shrink(plane, w, h, stride, sink.width, sink.height, format == PIXEL_FORMAT_420V);
        CVPixelBufferUnlockBaseAddress(image, 1);
        sink.out.push(Picture { pts: pts.value, luma });
    }
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

impl HevcDecoder {
    /// A decoder for a track's parameter sets (VPS, SPS, PPS), handing back `width`×`height` pictures for the
    /// presentation times `want` accepts.
    pub fn new(parameter_sets: &[Vec<u8>], nal_length_size: u8, timescale: u32, width: usize, height: usize, want: Box<dyn FnMut(i64) -> bool + Send>) -> anyhow::Result<Self> {
        let ptrs: Vec<*const u8> = parameter_sets.iter().map(|p| p.as_ptr()).collect();
        let sizes: Vec<usize> = parameter_sets.iter().map(Vec::len).collect();
        let mut format: CMFormatDescriptionRef = null();
        // SAFETY: the pointer and size arrays describe live slices; format is an out-parameter.
        let st = unsafe {
            CMVideoFormatDescriptionCreateFromHEVCParameterSets(null(), ptrs.len(), ptrs.as_ptr(), sizes.as_ptr(), i32::from(nal_length_size), null(), &raw mut format)
        };
        ensure!(st == 0, "HEVC format description: OSStatus {st}");
        let mut sink = Box::new(Sink { width, height, want, out: Vec::new(), error: None });
        let record = VTDecompressionOutputCallbackRecord { callback: on_picture, refcon: (&raw mut *sink).cast() };
        let mut session: VTDecompressionSessionRef = null();
        // SAFETY: the attributes dictionary and number are created and released here; the callback record is
        // copied by VideoToolbox; session is an out-parameter.
        let st = unsafe {
            let pixel = PIXEL_FORMAT_420V as i32;
            let number = CFNumberCreate(null(), CF_NUMBER_SINT32, (&raw const pixel).cast());
            let keys = [kCVPixelBufferPixelFormatTypeKey];
            let values = [number];
            let attrs = CFDictionaryCreate(null(), keys.as_ptr(), values.as_ptr(), 1, &raw const kCFTypeDictionaryKeyCallBacks, &raw const kCFTypeDictionaryValueCallBacks);
            let st = VTDecompressionSessionCreate(null(), format, null(), attrs, &raw const record, &raw mut session);
            CFRelease(attrs);
            CFRelease(number);
            st
        };
        if st != 0 {
            // SAFETY: format was created above and is released once.
            unsafe { CFRelease(format) };
            return Err(anyhow!("VideoToolbox decoder: OSStatus {st}"));
        }
        Ok(Self { session, format, sink, timescale })
    }

    /// Decode one coded picture (length-prefixed NAL units) with its times; returns the pictures that came out.
    pub fn decode(&mut self, data: &[u8], dts: i64, pts: i64) -> anyhow::Result<Vec<Picture>> {
        // SAFETY: the block buffer is allocated by CoreMedia and filled from `data`; the sample buffer refers to
        // it and to the live format description; both are released after the synchronous decode returns.
        unsafe {
            let mut block: CMBlockBufferRef = null();
            let st = CMBlockBufferCreateWithMemoryBlock(null(), null_mut(), data.len(), null(), null(), 0, data.len(), 0, &raw mut block);
            ensure!(st == 0, "block buffer: OSStatus {st}");
            let st = CMBlockBufferReplaceDataBytes(data.as_ptr().cast(), block, 0, data.len());
            if st != 0 {
                CFRelease(block);
                return Err(anyhow!("block buffer copy: OSStatus {st}"));
            }
            let timing = CMSampleTimingInfo { duration: CMTime::INVALID, presentationTimeStamp: CMTime::new(pts, self.timescale), decodeTimeStamp: CMTime::new(dts, self.timescale) };
            let size = data.len();
            let mut sample: CMSampleBufferRef = null();
            let st = CMSampleBufferCreateReady(null(), block, self.format, 1, 1, &raw const timing, 1, &raw const size, &raw mut sample);
            CFRelease(block);
            ensure!(st == 0, "sample buffer: OSStatus {st}");
            let mut info = 0u32;
            let st = VTDecompressionSessionDecodeFrame(self.session, sample, 0, null_mut(), &raw mut info);
            CFRelease(sample);
            ensure!(st == 0, "VideoToolbox decode: OSStatus {st}");
        }
        if let Some(e) = self.sink.error.take() {
            return Err(anyhow!("VideoToolbox picture: OSStatus {e}"));
        }
        Ok(std::mem::take(&mut self.sink.out))
    }

    /// Pictures still inside the decoder.
    pub fn flush(&mut self) -> Vec<Picture> {
        // SAFETY: the session is live.
        unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        std::mem::take(&mut self.sink.out)
    }
}

impl Drop for HevcDecoder {
    fn drop(&mut self) {
        // SAFETY: both were created in `new` and are released exactly once; invalidating first stops callbacks.
        unsafe {
            VTDecompressionSessionInvalidate(self.session);
            CFRelease(self.session);
            CFRelease(self.format);
        }
    }
}

// ---- AAC -------------------------------------------------------------------------------------------------

const FORMAT_AAC: u32 = u32::from_be_bytes(*b"aac ");
const FORMAT_LPCM: u32 = u32::from_be_bytes(*b"lpcm");
const FLOAT_PACKED: u32 = 1 | 8; // kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
const PROP_MAGIC_COOKIE: u32 = u32::from_be_bytes(*b"dmgc");
/// Our input callback's "nothing more for now" (any non-zero status the converter hands back).
const NO_MORE_INPUT: OSStatus = 0x6e6d_7264; // 'nmrd'
const FRAMES_PER_PACKET: u32 = 1024;

/// One packet for the converter's input callback.
struct Pending<'a> {
    data: Option<&'a [u8]>,
    description: AudioStreamPacketDescription,
}

extern "C" fn feed(_c: AudioConverterRef, packets: *mut u32, data: *mut AudioBufferList, desc: *mut *mut AudioStreamPacketDescription, user: *mut c_void) -> OSStatus {
    // SAFETY: user is the Pending passed to FillComplexBuffer below, alive for that call; the other pointers are
    // the converter's own out-parameters for this call.
    unsafe {
        let p = &mut *(user as *mut Pending);
        let Some(bytes) = p.data.take() else {
            *packets = 0;
            return NO_MORE_INPUT;
        };
        *packets = 1;
        (*data).mNumberBuffers = 1;
        (*data).mBuffers[0].mData = bytes.as_ptr() as *mut c_void;
        (*data).mBuffers[0].mDataByteSize = bytes.len() as u32;
        p.description = AudioStreamPacketDescription { mStartOffset: 0, mVariableFramesInPacket: 0, mDataByteSize: bytes.len() as u32 };
        if !desc.is_null() {
            *desc = &raw mut p.description;
        }
        0
    }
}

/// An AAC decoder whose output is folded to one channel for the silence detector: each sample is the loudest of
/// the channels at that instant (a quiet moment is quiet on every channel; averaging would hide a voice that is
/// only in the centre channel).
pub struct AacDecoder {
    converter: AudioConverterRef,
    channels: usize,
    buffer: Vec<f32>,
}

unsafe impl Send for AacDecoder {}

impl AacDecoder {
    pub fn new(channels: u16, sample_rate: u32, esds: &[u8]) -> anyhow::Result<Self> {
        let input = AudioStreamBasicDescription {
            mSampleRate: f64::from(sample_rate),
            mFormatID: FORMAT_AAC,
            mFramesPerPacket: FRAMES_PER_PACKET,
            mChannelsPerFrame: u32::from(channels),
            ..Default::default()
        };
        let bytes = 4 * u32::from(channels);
        let output = AudioStreamBasicDescription {
            mSampleRate: f64::from(sample_rate),
            mFormatID: FORMAT_LPCM,
            mFormatFlags: FLOAT_PACKED,
            mBytesPerPacket: bytes,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytes,
            mChannelsPerFrame: u32::from(channels),
            mBitsPerChannel: 32,
            mReserved: 0,
        };
        let mut converter: AudioConverterRef = null();
        // SAFETY: both descriptions are valid for the call; converter is an out-parameter.
        let st = unsafe { AudioConverterNew(&raw const input, &raw const output, &raw mut converter) };
        ensure!(st == 0, "AAC decoder: OSStatus {st}");
        // SAFETY: the cookie slice is valid for the call; the converter copies it.
        let st = unsafe { AudioConverterSetProperty(converter, PROP_MAGIC_COOKIE, esds.len() as u32, esds.as_ptr().cast()) };
        if st != 0 {
            // SAFETY: created above, disposed once.
            unsafe { AudioConverterDispose(converter) };
            return Err(anyhow!("AAC decoder config: OSStatus {st}"));
        }
        Ok(Self { converter, channels: usize::from(channels), buffer: vec![0.0; FRAMES_PER_PACKET as usize * 2 * usize::from(channels)] })
    }

    /// Decode one packet; returns the mono samples that came out (the decoder may hold some back at first).
    pub fn decode(&mut self, packet: &[u8]) -> anyhow::Result<Vec<f32>> {
        let mut pending = Pending { data: Some(packet), description: AudioStreamPacketDescription::default() };
        let frames_capacity = (self.buffer.len() / self.channels) as u32;
        let mut frames = frames_capacity;
        let mut list = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer { mNumberChannels: self.channels as u32, mDataByteSize: (self.buffer.len() * 4) as u32, mData: self.buffer.as_mut_ptr().cast() }],
        };
        // SAFETY: the input callback gets `pending` (alive for the call); the output list points at our buffer,
        // sized to `frames` frames.
        let st = unsafe { AudioConverterFillComplexBuffer(self.converter, feed, (&raw mut pending).cast(), &raw mut frames, &raw mut list, null_mut()) };
        ensure!(st == 0 || st == NO_MORE_INPUT, "AAC decode: OSStatus {st}");
        let n = (frames as usize).min(frames_capacity as usize);
        Ok(self.buffer[..n * self.channels].chunks_exact(self.channels).map(|f| f.iter().fold(0f32, |m, s| m.max(s.abs()))).collect())
    }
}

impl Drop for AacDecoder {
    fn drop(&mut self) {
        // SAFETY: created in `new`, disposed once.
        unsafe { AudioConverterDispose(self.converter) };
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
