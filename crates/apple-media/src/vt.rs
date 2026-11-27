//! The thin layer over VideoToolbox, CoreMedia and CoreVideo (through the `objc2-*` bindings): every call into
//! them that Rust cannot check is made here, once, with its justification, behind a safe function or method. The
//! decoders and the encoder (`video`, `analysis`) are then plain safe Rust. Sessions and other objects are
//! `CFRetained`, so they are released by drop; sessions are invalidated first.

use crate::outbox::{Outbox, sender};
use crate::video::PixelBuffer;
use anyhow::{Result, anyhow, ensure};
use objc2_core_foundation::{CFArray, CFBoolean, CFData, CFDictionary, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags, CMVideoFormatDescriptionCreate,
    CMVideoFormatDescriptionCreateFromH264ParameterSets, CMVideoFormatDescriptionCreateFromHEVCParameterSets,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms, kCMSampleAttachmentKey_NotSync,
};
use objc2_core_video::{CVImageBuffer, CVPixelBuffer, kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey};
use objc2_video_toolbox::{
    VTCompressionOutputCallback, VTCompressionSession, VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallback,
    VTDecompressionOutputCallbackRecord, VTDecompressionSession, VTEncodeInfoFlags, VTSessionSetProperty, kVTCompressionPropertyKey_AllowFrameReordering,
    kVTCompressionPropertyKey_AverageBitRate, kVTCompressionPropertyKey_DataRateLimits, kVTCompressionPropertyKey_ExpectedFrameRate,
    kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, kVTCompressionPropertyKey_ProfileLevel, kVTCompressionPropertyKey_Quality,
    kVTCompressionPropertyKey_RealTime, kVTEncodeFrameOptionKey_ForceKeyFrame, kVTProfileLevel_HEVC_Main_AutoLevel,
    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
};
use std::ffi::c_void;
use std::ptr::{NonNull, null, null_mut};

pub type OSStatus = i32;

pub const CODEC_MPEG2: u32 = u32::from_be_bytes(*b"mp2v");
pub const CODEC_HEVC: u32 = u32::from_be_bytes(*b"hvc1");
/// 8-bit 4:2:0, Y in its own plane, video range (`420v`) and full range (`420f`).
pub const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
pub const PIXEL_FORMAT_420F: u32 = u32::from_be_bytes(*b"420f");

/// A time in `timescale` units.
pub fn time(value: i64, timescale: u32) -> CMTime {
    CMTime { value, timescale: timescale as i32, flags: CMTimeFlags::Valid, epoch: 0 }
}

/// "No time" (e.g. a sample's unknown duration, or "all frames").
pub const INVALID_TIME: CMTime = CMTime { value: 0, timescale: 0, flags: CMTimeFlags(0), epoch: 0 };

/// The constant keys and values VideoToolbox and CoreVideo define, read once.
pub struct Keys {
    pub pixel_format: &'static CFString,
    pub width: &'static CFString,
    pub height: &'static CFString,
    pub hardware_encoder: &'static CFString,
    pub real_time: &'static CFString,
    pub profile_level: &'static CFString,
    pub hevc_main_auto: &'static CFString,
    pub allow_reordering: &'static CFString,
    pub average_bit_rate: &'static CFString,
    pub data_rate_limits: &'static CFString,
    pub quality: &'static CFString,
    pub max_key_interval: &'static CFString,
    pub expected_frame_rate: &'static CFString,
    pub force_key_frame: &'static CFString,
}

pub fn keys() -> Keys {
    // SAFETY: Apple's exported constant strings, initialised before main and never changed.
    unsafe {
        Keys {
            pixel_format: kCVPixelBufferPixelFormatTypeKey,
            width: kCVPixelBufferWidthKey,
            height: kCVPixelBufferHeightKey,
            hardware_encoder: kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
            real_time: kVTCompressionPropertyKey_RealTime,
            profile_level: kVTCompressionPropertyKey_ProfileLevel,
            hevc_main_auto: kVTProfileLevel_HEVC_Main_AutoLevel,
            allow_reordering: kVTCompressionPropertyKey_AllowFrameReordering,
            average_bit_rate: kVTCompressionPropertyKey_AverageBitRate,
            data_rate_limits: kVTCompressionPropertyKey_DataRateLimits,
            quality: kVTCompressionPropertyKey_Quality,
            max_key_interval: kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration,
            expected_frame_rate: kVTCompressionPropertyKey_ExpectedFrameRate,
            force_key_frame: kVTEncodeFrameOptionKey_ForceKeyFrame,
        }
    }
}

/// A dictionary of settings (pass it on with `.as_ref()`).
pub fn dict(entries: &[(&CFString, &CFType)]) -> CFRetained<CFDictionary<CFString, CFType>> {
    let (keys, values): (Vec<&CFString>, Vec<&CFType>) = entries.iter().copied().unzip();
    CFDictionary::from_slices(&keys, &values)
}

/// `true`, as a CoreFoundation value.
pub fn yes() -> &'static CFType {
    CFBoolean::new(true)
}

/// `false`, as a CoreFoundation value.
pub fn no() -> &'static CFType {
    CFBoolean::new(false)
}

/// Take ownership of a +1 reference returned through an out-parameter, if one was.
fn adopt<T: objc2_core_foundation::Type>(st: OSStatus, out: *const T, what: &str) -> Result<CFRetained<T>> {
    ensure!(st == 0, "{what}: OSStatus {st}");
    let out = NonNull::new(out.cast_mut()).ok_or_else(|| anyhow!("{what}: nothing returned"))?;
    // SAFETY: the call succeeded and returned an object it created for us (+1), which we now own.
    Ok(unsafe { CFRetained::from_raw(out) })
}

/// A format description from just a codec and a size (MPEG-2).
pub fn video_format(codec: u32, width: u32, height: u32) -> Result<CFRetained<CMFormatDescription>> {
    let mut out: *const CMFormatDescription = null();
    // SAFETY: plain values and an out-parameter that lives across the call.
    let st = unsafe { CMVideoFormatDescriptionCreate(None, codec, width as i32, height as i32, None, NonNull::from(&mut out)) };
    adopt(st, out, "video format description")
}

/// Pointers and sizes of parameter sets, as CoreMedia wants them.
fn parameter_set_arrays(sets: &[&[u8]]) -> Result<(Vec<NonNull<u8>>, Vec<usize>)> {
    let ptrs = sets.iter().map(|s| NonNull::new(s.as_ptr().cast_mut()).ok_or_else(|| anyhow!("empty parameter set"))).collect::<Result<_>>()?;
    Ok((ptrs, sets.iter().map(|s| s.len()).collect()))
}

/// An H.264 format description from its SPS and PPS (NAL units without start codes); samples are then
/// length-prefixed NAL units with 4-byte lengths.
pub fn h264_format(sps: &[u8], pps: &[u8]) -> Result<CFRetained<CMFormatDescription>> {
    let (mut ptrs, mut sizes) = parameter_set_arrays(&[sps, pps])?;
    let mut out: *const CMFormatDescription = null();
    // SAFETY: the pointer and size arrays describe live slices for the length of the call; out-parameter.
    let st = unsafe {
        CMVideoFormatDescriptionCreateFromH264ParameterSets(None, 2, NonNull::from(&mut ptrs[0]), NonNull::from(&mut sizes[0]), 4, NonNull::from(&mut out))
    };
    adopt(st, out, "H.264 format description")
}

/// An HEVC format description from its parameter sets (VPS, SPS, PPS).
pub fn hevc_format(sets: &[Vec<u8>], nal_length_size: u8) -> Result<CFRetained<CMFormatDescription>> {
    let slices: Vec<&[u8]> = sets.iter().map(Vec::as_slice).collect();
    let (mut ptrs, mut sizes) = parameter_set_arrays(&slices)?;
    ensure!(!ptrs.is_empty(), "no HEVC parameter sets");
    let mut out: *const CMFormatDescription = null();
    // SAFETY: as in h264_format.
    let st = unsafe {
        CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            None,
            ptrs.len(),
            NonNull::from(&mut ptrs[0]),
            NonNull::from(&mut sizes[0]),
            i32::from(nal_length_size),
            None,
            NonNull::from(&mut out),
        )
    };
    adopt(st, out, "HEVC format description")
}

/// One coded picture as a sample buffer: `data` is copied into a block buffer CoreMedia owns.
pub fn sample(data: &[u8], format: &CMFormatDescription, pts: CMTime, dts: CMTime) -> Result<CFRetained<CMSampleBuffer>> {
    let mut block: *mut CMBlockBuffer = null_mut();
    // SAFETY: CoreMedia allocates the block (null memory block, default allocator); out-parameter.
    let st = unsafe { CMBlockBuffer::create_with_memory_block(None, null_mut(), data.len(), None, null(), 0, data.len(), 0, NonNull::from(&mut block)) };
    let block = adopt(st, block, "block buffer")?;
    // SAFETY: `data` is valid for its length; the block was created with exactly that length.
    let st = unsafe { CMBlockBuffer::replace_data_bytes(NonNull::from(data).cast(), &block, 0, data.len()) };
    ensure!(st == 0, "block buffer copy: OSStatus {st}");
    let timing = CMSampleTimingInfo { duration: INVALID_TIME, presentationTimeStamp: pts, decodeTimeStamp: dts };
    let size = data.len();
    let mut sample: *mut CMSampleBuffer = null_mut();
    // SAFETY: one sample: the timing and size arrays are single values that live across the call; out-parameter.
    let st = unsafe { CMSampleBuffer::create_ready(None, Some(&block), Some(format), 1, 1, &timing, 1, &size, NonNull::from(&mut sample)) };
    adopt(st, sample, "sample buffer")
}

/// A retained picture handed to a decoder callback, or None if there is none.
fn picture_from_callback(image: *mut CVImageBuffer) -> Option<CFRetained<CVPixelBuffer>> {
    let image = NonNull::new(image)?;
    // SAFETY: VideoToolbox hands the callback a valid image buffer for the duration of the call; retaining it
    // keeps it alive afterwards.
    Some(unsafe { CFRetained::retain(image) })
}

/// Read plane `plane` of a picture: `f` gets its bytes, width, height and bytes per row. None if it cannot be
/// read (the buffer is locked read-only around the call).
pub fn read_plane<R>(picture: &CVPixelBuffer, plane: usize, f: impl FnOnce(&[u8], usize, usize, usize) -> R) -> Option<R> {
    use objc2_core_video::{
        CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetWidthOfPlane,
        CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    };
    // SAFETY: a live buffer, locked read-only here and unlocked below on every path.
    if unsafe { CVPixelBufferLockBaseAddress(picture, CVPixelBufferLockFlags::ReadOnly) } != 0 {
        return None;
    }
    let base = CVPixelBufferGetBaseAddressOfPlane(picture, plane);
    let (stride, w, h) =
        (CVPixelBufferGetBytesPerRowOfPlane(picture, plane), CVPixelBufferGetWidthOfPlane(picture, plane), CVPixelBufferGetHeightOfPlane(picture, plane));
    let out = (!base.is_null()).then(|| {
        // SAFETY: while the buffer is locked, the plane's memory is `stride` bytes for each of its `h` rows.
        let bytes = unsafe { std::slice::from_raw_parts(base.cast::<u8>(), stride * h) };
        f(bytes, w, h, stride)
    });
    // SAFETY: locked above with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(picture, CVPixelBufferLockFlags::ReadOnly) };
    out
}

/// What an encoder callback received, copied out: the frame, and the `hvcC` box body on keyframes.
pub struct EncodedSample {
    pub pts: i64,
    pub dts: i64,
    pub data: Vec<u8>,
    pub keyframe: bool,
    pub hvcc: Option<Vec<u8>>,
}

/// Copy out what an encoder callback was handed. None for a dropped frame; an error status if it can't be read.
fn encoded_sample(sample: *mut CMSampleBuffer) -> Option<Result<EncodedSample, OSStatus>> {
    let sample = NonNull::new(sample)?;
    // SAFETY: VideoToolbox hands the callback a valid sample buffer for the duration of the call; everything read
    // from it is copied out before returning.
    Some(unsafe {
        let sample = sample.as_ref();
        (|| {
            let block = sample.data_buffer().ok_or(-3)?;
            let len = block.data_length();
            let mut data = vec![0u8; len];
            let st = block.copy_data_bytes(0, len, NonNull::new(data.as_mut_ptr()).ok_or(-3)?.cast());
            if st != 0 {
                return Err(st);
            }
            let (pts, dts) = (sample.presentation_time_stamp(), sample.decode_time_stamp());
            // The attachments are an array of dictionaries keyed by strings (CoreMedia's documented layout).
            let attachments = sample.sample_attachments_array(false).map(|a| CFRetained::cast_unchecked::<CFArray<CFDictionary<CFString, CFType>>>(a));
            let not_sync = attachments
                .and_then(|a| a.get(0))
                .and_then(|d| d.get(kCMSampleAttachmentKey_NotSync))
                .and_then(|v| v.downcast::<CFBoolean>().ok())
                .is_some_and(|b| b.as_bool());
            let hvcc = if not_sync {
                None
            } else {
                // The sample description's extension atoms: a dictionary from atom names to their bodies.
                sample
                    .format_description()
                    .and_then(|f| f.extension(kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms))
                    .and_then(|a| a.downcast::<CFDictionary>().ok())
                    .map(|a| CFRetained::cast_unchecked::<CFDictionary<CFString, CFType>>(a))
                    .and_then(|a| a.get(&CFString::from_static_str("hvcC")))
                    .and_then(|v| v.downcast::<CFData>().ok())
                    .map(|d| d.to_vec())
            };
            Ok(EncodedSample {
                pts: pts.value,
                dts: if dts.flags.contains(CMTimeFlags::Valid) { dts.value } else { pts.value },
                data,
                keyframe: !not_sync,
                hvcc,
            })
        })()
    })
}

/// What a decompression session's callback sends: a picture and its presentation time, or why one failed.
pub type Picture = Result<(i64, PixelBuffer), OSStatus>;

extern "C-unwind" fn on_picture(
    refcon: *mut c_void,
    _f: *mut c_void,
    status: OSStatus,
    _flags: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    pts: CMTime,
    _d: CMTime,
) {
    let msg = if status != 0 {
        Err(status)
    } else {
        let Some(picture) = picture_from_callback(image) else { return }; // nothing decoded (a dropped frame)
        Ok((pts.value, PixelBuffer::new(picture)))
    };
    // SAFETY: refcon is the context of the Decompression's own outbox (see `Decompression::new`), which outlives
    // the session.
    let _ = unsafe { sender::<Picture>(refcon) }.send(msg);
}

/// A decompression session with its callback and outbox, invalidated and released on drop.
pub struct Decompression {
    // Field order is drop order: the session (invalidated, so no more callbacks) before the outbox they send to.
    session: CFRetained<VTDecompressionSession>,
    outbox: Outbox<Picture>,
    flags: VTDecodeFrameFlags,
}

// SAFETY: a VideoToolbox session may be used from any thread, one at a time; `&mut self` methods ensure that.
unsafe impl Send for Decompression {}

impl Decompression {
    /// A session for `format`, with pictures in `attributes` (e.g. a pixel format), or VideoToolbox's choice.
    pub fn new(format: &CMFormatDescription, attributes: Option<&CFDictionary>) -> Result<Self> {
        let outbox = Outbox::new();
        let callback: VTDecompressionOutputCallback = Some(on_picture);
        let record = VTDecompressionOutputCallbackRecord { decompressionOutputCallback: callback, decompressionOutputRefCon: outbox.context() };
        let mut out: *mut VTDecompressionSession = null_mut();
        // SAFETY: the format and attributes are live; the callback record is copied; its context is our outbox,
        // which this struct keeps until after the session is invalidated.
        let st = unsafe { VTDecompressionSession::create(None, format, None, attributes, &record, NonNull::from(&mut out)) };
        Ok(Self { session: adopt(st, out, "video decoder")?, outbox, flags: VTDecodeFrameFlags(0) })
    }

    /// Let `decode` return before the picture is decoded, so the decoder works through several at once (for
    /// decoding faster than real time). Pictures then come out later: `wait` before the last `drain`.
    pub fn asynchronous(&mut self) {
        self.flags = VTDecodeFrameFlags::Frame_EnableAsynchronousDecompression;
    }

    /// Decode one sample: synchronously (then `drain` has its pictures), or, if `asynchronous`, queued.
    pub fn decode(&mut self, sample: &CMSampleBuffer) -> Result<()> {
        let mut info = VTDecodeInfoFlags(0);
        // SAFETY: a live session and sample; no per-frame context; out-parameter.
        let st = unsafe { self.session.decode_frame(sample, self.flags, null_mut(), &mut info) };
        ensure!(st == 0, "VideoToolbox decode: OSStatus {st}");
        Ok(())
    }

    /// Wait for every picture still in the decoder to come out.
    pub fn wait(&mut self) {
        // SAFETY: a live session.
        unsafe { self.session.wait_for_asynchronous_frames() };
    }

    /// The pictures (or errors) that have come out since the last call.
    pub fn drain(&mut self) -> Vec<Picture> {
        self.outbox.drain()
    }
}

impl Drop for Decompression {
    fn drop(&mut self) {
        // SAFETY: a live session; after this no callback runs, so the outbox may go.
        unsafe { self.session.invalidate() };
    }
}

/// What a compression session's callback sends: a frame, or why encoding failed.
pub type Frame = Result<EncodedSample, OSStatus>;

extern "C-unwind" fn on_frame(refcon: *mut c_void, _f: *mut c_void, status: OSStatus, _flags: VTEncodeInfoFlags, sample: *mut CMSampleBuffer) {
    let msg = if status != 0 {
        Err(status)
    } else {
        match encoded_sample(sample) {
            None => return, // dropped frame
            Some(r) => r,
        }
    };
    // SAFETY: refcon is the context of the Compression's own outbox (see `Compression::new`), which outlives the
    // session. This runs on VideoToolbox's thread; the outbox's channel is what makes that safe.
    let _ = unsafe { sender::<Frame>(refcon) }.send(msg);
}

/// A compression session with its callback and outbox, invalidated and released on drop.
pub struct Compression {
    // Field order is drop order: the session (invalidated, so no more callbacks) before the outbox they send to.
    session: CFRetained<VTCompressionSession>,
    outbox: Outbox<Frame>,
}

// SAFETY: as for Decompression.
unsafe impl Send for Compression {}

impl Compression {
    /// A `width`×`height` session for `codec`.
    pub fn new(width: u32, height: u32, codec: u32, specification: Option<&CFDictionary>) -> Result<Self> {
        let outbox = Outbox::new();
        let callback: VTCompressionOutputCallback = Some(on_frame);
        let mut out: *mut VTCompressionSession = null_mut();
        // SAFETY: plain values and a live specification; the context is our outbox, which this struct keeps until
        // after the session is invalidated.
        let st = unsafe {
            VTCompressionSession::create(
                None,
                width as i32,
                height as i32,
                codec,
                specification,
                None,
                None,
                callback,
                outbox.context(),
                NonNull::from(&mut out),
            )
        };
        Ok(Self { session: adopt(st, out, "video encoder")?, outbox })
    }

    pub fn set(&mut self, key: &CFString, value: &CFType) -> Result<()> {
        // SAFETY: a live session, key and value.
        let st = unsafe { VTSessionSetProperty(&self.session, key, Some(value)) };
        ensure!(st == 0, "encoder setting: OSStatus {st}");
        Ok(())
    }

    pub fn prepare(&mut self) {
        // SAFETY: a live session.
        unsafe { self.session.prepare_to_encode_frames() };
    }

    /// Encode a picture; the frame comes out later (see `drain`).
    pub fn encode(&mut self, picture: &CVPixelBuffer, pts: CMTime, duration: CMTime, properties: Option<&CFDictionary>) -> Result<()> {
        let mut info = VTEncodeInfoFlags(0);
        // SAFETY: a live session, picture and properties; no per-frame context; out-parameter.
        let st = unsafe { self.session.encode_frame(picture, pts, duration, properties, null_mut(), &mut info) };
        ensure!(st == 0, "HEVC encode: OSStatus {st}");
        Ok(())
    }

    /// Finish every frame given so far (they come out before this returns).
    pub fn complete(&mut self) -> Result<()> {
        // SAFETY: a live session; an invalid time means all frames.
        let st = unsafe { self.session.complete_frames(INVALID_TIME) };
        ensure!(st == 0, "HEVC encode: OSStatus {st}");
        Ok(())
    }

    /// The frames (or errors) that have come out since the last call.
    pub fn drain(&mut self) -> Vec<Frame> {
        self.outbox.drain()
    }
}

impl Drop for Compression {
    fn drop(&mut self) {
        // SAFETY: a live session; after this no callback runs, so the outbox may go.
        unsafe { self.session.invalidate() };
    }
}
