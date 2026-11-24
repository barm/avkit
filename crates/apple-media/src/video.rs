//! Full-frame video with VideoToolbox: decoding broadcast video into pixel buffers (MPEG-2 has no hardware decoder
//! on Apple Silicon, but VideoToolbox's software one works) and encoding them to HEVC in hardware.

use crate::sys::*;
use anyhow::{anyhow, ensure};
use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A callback sink's lock. VideoToolbox calls its output callbacks on its own threads (always for compression,
/// sometimes for decompression), so the sink is shared with the caller and must be locked by both; a poisoned lock
/// is still usable (nothing in a sink can be left half-updated).
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A retained CVPixelBuffer (a decoded picture), released on drop.
pub struct PixelBuffer(CVImageBufferRef);

unsafe impl Send for PixelBuffer {}

impl PixelBuffer {
    /// Take a +1 reference.
    /// SAFETY: `pb` must be a valid CVPixelBuffer.
    unsafe fn retain(pb: CVImageBufferRef) -> Self {
        unsafe { CFRetain(pb) };
        Self(pb)
    }
    /// Own a +1 reference created elsewhere (a pool's new buffer).
    pub fn adopt(pb: CVImageBufferRef) -> Self {
        Self(pb)
    }
    pub fn raw(&self) -> CVImageBufferRef {
        self.0
    }
    pub fn size(&self) -> (usize, usize) {
        // SAFETY: a live pixel buffer.
        unsafe { (CVPixelBufferGetWidth(self.0), CVPixelBufferGetHeight(self.0)) }
    }
}

impl Clone for PixelBuffer {
    fn clone(&self) -> Self {
        // SAFETY: a live pixel buffer; another reference to it.
        unsafe { Self::retain(self.0) }
    }
}

impl Drop for PixelBuffer {
    fn drop(&mut self) {
        // SAFETY: we hold one reference.
        unsafe { CFRelease(self.0) };
    }
}

/// A decoded picture and its presentation time (in the timescale it was given in).
pub struct Decoded {
    pub pts: i64,
    pub picture: PixelBuffer,
}

struct DecodeSink {
    out: Vec<Decoded>,
    error: Option<OSStatus>,
}

extern "C" fn on_decoded(refcon: *mut c_void, _f: *mut c_void, status: OSStatus, _flags: u32, image: CVImageBufferRef, pts: CMTime, _d: CMTime) {
    // SAFETY: refcon is the decoder's boxed, locked sink, alive for the session's life.
    let mut sink = lock(unsafe { &*(refcon as *const Mutex<DecodeSink>) });
    if status != 0 {
        sink.error = Some(status);
    } else if !image.is_null() {
        // SAFETY: a valid pixel buffer for this call; retained to outlive it.
        sink.out.push(Decoded { pts: pts.value, picture: unsafe { PixelBuffer::retain(image) } });
    }
}

/// Decodes broadcast MPEG-2 (or another codec VideoToolbox knows by its four-character code) into pixel buffers.
pub struct Decoder {
    session: VTDecompressionSessionRef,
    format: CMFormatDescriptionRef,
    sink: Box<Mutex<DecodeSink>>,
    timescale: u32,
}

unsafe impl Send for Decoder {}

impl Decoder {
    pub fn new(codec: u32, width: u32, height: u32, timescale: u32) -> anyhow::Result<Self> {
        let mut format: CMFormatDescriptionRef = null();
        // SAFETY: out-parameter.
        let st = unsafe { CMVideoFormatDescriptionCreate(null(), codec, width as i32, height as i32, null(), &raw mut format) };
        ensure!(st == 0, "video format description: OSStatus {st}");
        Self::with_format(format, timescale)
    }

    /// An H.264 decoder for these parameter sets (NAL units without start codes); samples are then given as
    /// length-prefixed NAL units (4-byte lengths).
    pub fn h264(sps: &[u8], pps: &[u8], timescale: u32) -> anyhow::Result<Self> {
        let ptrs = [sps.as_ptr(), pps.as_ptr()];
        let sizes = [sps.len(), pps.len()];
        let mut format: CMFormatDescriptionRef = null();
        // SAFETY: the pointer and size arrays describe live slices; out-parameter.
        let st = unsafe { CMVideoFormatDescriptionCreateFromH264ParameterSets(null(), 2, ptrs.as_ptr(), sizes.as_ptr(), 4, &raw mut format) };
        ensure!(st == 0, "H.264 format description: OSStatus {st}");
        Self::with_format(format, timescale)
    }

    /// A decoder for a format description (taken over: released with the decoder).
    fn with_format(format: CMFormatDescriptionRef, timescale: u32) -> anyhow::Result<Self> {
        let sink = Box::new(Mutex::new(DecodeSink { out: Vec::new(), error: None }));
        let record = VTDecompressionOutputCallbackRecord { callback: on_decoded, refcon: (&raw const *sink).cast_mut().cast() };
        let mut session: VTDecompressionSessionRef = null();
        // SAFETY: as in mac.rs: attributes created and released here, record copied, session out-parameter.
        let st = unsafe {
            let pixel = 0x3432_3076i32; // 420v, what the encoder takes
            let number = CFNumberCreate(null(), 3, (&raw const pixel).cast());
            let keys = [kCVPixelBufferPixelFormatTypeKey];
            let values = [number];
            let attrs = CFDictionaryCreate(
                null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            );
            let st = VTDecompressionSessionCreate(null(), format, null(), attrs, &raw const record, &raw mut session);
            CFRelease(attrs);
            CFRelease(number);
            st
        };
        if st != 0 {
            // SAFETY: created above.
            unsafe { CFRelease(format) };
            return Err(anyhow!("video decoder: OSStatus {st}"));
        }
        Ok(Self { session, format, sink, timescale })
    }

    /// Decode one coded picture; returns the pictures that came out (in decode order: sort by `pts`).
    pub fn decode(&mut self, data: &[u8], dts: i64, pts: i64) -> anyhow::Result<Vec<Decoded>> {
        // SAFETY: as HevcDecoder::decode in mac.rs.
        unsafe {
            let mut block: CMBlockBufferRef = null();
            let st = CMBlockBufferCreateWithMemoryBlock(null(), null_mut(), data.len(), null(), null(), 0, data.len(), 0, &raw mut block);
            ensure!(st == 0, "block buffer: OSStatus {st}");
            CMBlockBufferReplaceDataBytes(data.as_ptr().cast(), block, 0, data.len());
            let timing = CMSampleTimingInfo {
                duration: CMTime::INVALID,
                presentationTimeStamp: CMTime::new(pts, self.timescale),
                decodeTimeStamp: CMTime::new(dts, self.timescale),
            };
            let size = data.len();
            let mut sample: CMSampleBufferRef = null();
            let st = CMSampleBufferCreateReady(null(), block, self.format, 1, 1, &raw const timing, 1, &raw const size, &raw mut sample);
            CFRelease(block);
            ensure!(st == 0, "sample buffer: OSStatus {st}");
            let mut info = 0u32;
            let st = VTDecompressionSessionDecodeFrame(self.session, sample, 0, null_mut(), &raw mut info);
            CFRelease(sample);
            if st != 0 {
                return Err(anyhow!("video decode: OSStatus {st}"));
            }
        }
        let mut sink = lock(&self.sink);
        if let Some(e) = sink.error.take() {
            return Err(anyhow!("video decode: OSStatus {e}"));
        }
        Ok(std::mem::take(&mut sink.out))
    }

    pub fn flush(&mut self) -> Vec<Decoded> {
        // SAFETY: a live session.
        unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        std::mem::take(&mut lock(&self.sink).out)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: created in `new`, released once.
        unsafe {
            VTDecompressionSessionInvalidate(self.session);
            CFRelease(self.session);
            CFRelease(self.format);
        }
    }
}

/// An encoded picture: times in the encoder's timescale, data as length-prefixed NAL units.
pub struct Encoded {
    pub pts: i64,
    pub dts: i64,
    pub data: Vec<u8>,
    pub keyframe: bool,
}

struct EncodeSink {
    out: Vec<Encoded>,
    error: Option<OSStatus>,
    /// The `hvcC` box body, from the first output's format description.
    hvcc: Option<Vec<u8>>,
}

extern "C" fn on_encoded(refcon: *mut c_void, _f: *mut c_void, status: OSStatus, _flags: u32, sample: CMSampleBufferRef) {
    // SAFETY: refcon is the encoder's boxed, locked sink, alive for the session's life. VideoToolbox calls this on
    // its own thread, while the encoder's owner may be taking output: hence the lock.
    let mut sink = lock(unsafe { &*(refcon as *const Mutex<EncodeSink>) });
    if status != 0 {
        sink.error = Some(status);
        return;
    }
    if sample.is_null() {
        return; // dropped frame
    }
    // SAFETY: a valid sample buffer for this call; everything read from it is copied out.
    unsafe {
        let block = CMSampleBufferGetDataBuffer(sample);
        let len = CMBlockBufferGetDataLength(block);
        let mut data = vec![0u8; len];
        if CMBlockBufferCopyDataBytes(block, 0, len, data.as_mut_ptr().cast()) != 0 {
            sink.error = Some(-3);
            return;
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample);
        let dts = CMSampleBufferGetDecodeTimeStamp(sample);
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, 0);
        let not_sync = !attachments.is_null() && CFArrayGetCount(attachments) > 0 && {
            let dict = CFArrayGetValueAtIndex(attachments, 0);
            let v = CFDictionaryGetValue(dict, kCMSampleAttachmentKey_NotSync);
            !v.is_null() && v == kCFBooleanTrue
        };
        if sink.hvcc.is_none() {
            let format = CMSampleBufferGetFormatDescription(sample);
            let atoms = CMFormatDescriptionGetExtension(format, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms);
            if !atoms.is_null() {
                let key = CFStringCreateWithCString(null(), c"hvcC".as_ptr(), 0x0800_0100);
                let data = CFDictionaryGetValue(atoms, key);
                CFRelease(key);
                if !data.is_null() {
                    sink.hvcc = Some(std::slice::from_raw_parts(CFDataGetBytePtr(data), CFDataGetLength(data) as usize).to_vec());
                }
            }
        }
        let pts = pts.value;
        let dts = if dts.flags & 1 != 0 { dts.value } else { pts };
        sink.out.push(Encoded { pts, dts, data, keyframe: !not_sync });
    }
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
    session: VTCompressionSessionRef,
    sink: Box<Mutex<EncodeSink>>,
    timescale: u32,
    force_key: CFDictionaryRef,
}

unsafe impl Send for Encoder {}

impl Encoder {
    pub fn new(width: u32, height: u32, rate: Rate, fps: f64, max_key_secs: f64, realtime: bool, timescale: u32) -> anyhow::Result<Self> {
        let sink = Box::new(Mutex::new(EncodeSink { out: Vec::new(), error: None, hvcc: None }));
        let mut session: VTCompressionSessionRef = null();
        // SAFETY: the dictionaries and numbers are created, used and released here; session is an out-parameter
        // and the sink outlives it (it is dropped after the session is invalidated).
        unsafe {
            let spec_keys = [kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder];
            let spec_values = [kCFBooleanTrue];
            let spec = CFDictionaryCreate(
                null(),
                spec_keys.as_ptr(),
                spec_values.as_ptr(),
                1,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            );
            let st = VTCompressionSessionCreate(
                null(),
                width as i32,
                height as i32,
                CODEC_HEVC,
                spec,
                null(),
                null(),
                on_encoded,
                (&raw const *sink).cast_mut().cast(),
                &raw mut session,
            );
            CFRelease(spec);
            ensure!(st == 0, "HEVC encoder: OSStatus {st}");
            let number_i64 = |v: i64| CFNumberCreate(null(), CF_NUMBER_SINT64, (&raw const v).cast());
            let number_f64 = |v: f64| CFNumberCreate(null(), CF_NUMBER_FLOAT64, (&raw const v).cast());
            let (rate_key, rate_value) = match rate {
                Rate::Bitrate(bps) => (kVTCompressionPropertyKey_AverageBitRate, number_i64(bps)),
                Rate::Quality(q) => (kVTCompressionPropertyKey_Quality, number_f64(f64::from(q.clamp(0.0, 1.0)))),
            };
            let set = |k: CFStringRef, v: CFTypeRef, release: bool| {
                let st = VTSessionSetProperty(session, k, v);
                if release {
                    CFRelease(v);
                }
                st
            };
            let results = [
                set(kVTCompressionPropertyKey_RealTime, if realtime { kCFBooleanTrue } else { kCFBooleanFalse }, false),
                set(kVTCompressionPropertyKey_ProfileLevel, kVTProfileLevel_HEVC_Main_AutoLevel, false),
                set(kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse, false),
                set(rate_key, rate_value, true),
                set(kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, number_f64(max_key_secs), true),
                set(kVTCompressionPropertyKey_ExpectedFrameRate, number_f64(fps), true),
            ];
            if let Some(st) = results.iter().find(|&&st| st != 0) {
                VTCompressionSessionInvalidate(session);
                CFRelease(session);
                return Err(anyhow!("HEVC encoder settings: OSStatus {st}"));
            }
            VTCompressionSessionPrepareToEncodeFrames(session);
            let keys = [kVTEncodeFrameOptionKey_ForceKeyFrame];
            let values = [kCFBooleanTrue];
            let force_key = CFDictionaryCreate(
                null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            );
            Ok(Self { session, sink, timescale, force_key })
        }
    }

    /// Encode a picture shown at `pts` for `duration`; `key` forces a keyframe. Returns what came out.
    pub fn encode(&mut self, picture: &PixelBuffer, pts: i64, duration: i64, key: bool) -> anyhow::Result<Vec<Encoded>> {
        let mut info = 0u32;
        // SAFETY: a live session and pixel buffer; the frame-properties dictionary lives as long as the encoder.
        let st = unsafe {
            VTCompressionSessionEncodeFrame(
                self.session,
                picture.raw(),
                CMTime::new(pts, self.timescale),
                CMTime::new(duration, self.timescale),
                if key { self.force_key } else { null() },
                null_mut(),
                &raw mut info,
            )
        };
        ensure!(st == 0, "HEVC encode: OSStatus {st}");
        let mut sink = lock(&self.sink);
        if let Some(e) = sink.error.take() {
            return Err(anyhow!("HEVC encode: OSStatus {e}"));
        }
        Ok(std::mem::take(&mut sink.out))
    }

    /// Everything still in the encoder.
    pub fn finish(&mut self) -> anyhow::Result<Vec<Encoded>> {
        // SAFETY: a live session; an invalid time means "all frames".
        let st = unsafe { VTCompressionSessionCompleteFrames(self.session, CMTime::INVALID) };
        ensure!(st == 0, "HEVC encode: OSStatus {st}");
        Ok(std::mem::take(&mut lock(&self.sink).out))
    }

    /// The `hvcC` box body, once the first picture is out.
    pub fn hvcc(&self) -> Option<Vec<u8>> {
        lock(&self.sink).hvcc.clone()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: created in `new`, released once; invalidating first stops callbacks.
        unsafe {
            VTCompressionSessionInvalidate(self.session);
            CFRelease(self.session);
            CFRelease(self.force_key);
        }
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
                    dec = Some(Decoder::new(CODEC_MPEG2, w, h, 90_000).unwrap());
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
