#![cfg(target_os = "macos")]

//! Deinterlacing on the GPU (Metal): yadif, the edge-directed spatial/temporal deinterlacer ffmpeg also has,
//! over decoded 1080i (and 480i) pictures. One picture out per field (59.94 a second from 29.97i, like
//! `bwdif=mode=send_field`), or one per frame. Decoded pixel buffers are read as Metal textures in place
//! (CoreVideo's texture cache); output goes to buffers from a pool, ready for the encoder.

use anyhow::{Context, anyhow, ensure};
use apple_media::video::PixelBuffer;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer, CVPixelBufferPool, kCVPixelBufferHeightKey,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLPixelFormat, MTLSize,
};
use std::ptr::{NonNull, null_mut};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint keep;   // parity of the lines kept from `cur` (0: top field, 1: bottom)
    uint early;  // 1: the missing lines' nearest fields are in (prev, cur); 0: in (cur, next)
};

static inline float4 px(texture2d<float, access::read> t, int x, int y) {
    x = clamp(x, 0, int(t.get_width()) - 1);
    y = clamp(y, 0, int(t.get_height()) - 1);
    return t.read(uint2(x, y));
}

kernel void yadif(texture2d<float, access::read> prev [[texture(0)]],
                  texture2d<float, access::read> cur [[texture(1)]],
                  texture2d<float, access::read> next [[texture(2)]],
                  texture2d<float, access::write> out [[texture(3)]],
                  constant Params &p [[buffer(0)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    int x = int(gid.x), y = int(gid.y);
    if ((uint(y) & 1u) == p.keep) { out.write(cur.read(gid), gid); return; }

    float4 c = px(cur, x, y - 1), e = px(cur, x, y + 1);
    // The same line in the two nearest fields of the missing parity.
    float4 p2 = p.early ? px(prev, x, y) : px(cur, x, y);
    float4 n2 = p.early ? px(cur, x, y) : px(next, x, y);
    float4 d = (p2 + n2) * 0.5;
    float4 td0 = abs(p2 - n2);
    float4 td1 = (abs(px(prev, x, y - 1) - c) + abs(px(prev, x, y + 1) - e)) * 0.5;
    float4 td2 = (abs(px(next, x, y - 1) - c) + abs(px(next, x, y + 1) - e)) * 0.5;
    float4 diff = max(max(td0 * 0.5, td1), td2);

    // Spatial: the best of the vertical and four diagonal directions.
    float4 sp = (c + e) * 0.5;
    float4 score = abs(px(cur, x - 1, y - 1) - px(cur, x - 1, y + 1)) + abs(c - e) + abs(px(cur, x + 1, y - 1) - px(cur, x + 1, y + 1)) - 1.0 / 255.0;
    for (int j = -1; j >= -2; j--) {
        float4 s = abs(px(cur, x - 1 + j, y - 1) - px(cur, x - 1 - j, y + 1)) + abs(px(cur, x + j, y - 1) - px(cur, x - j, y + 1)) + abs(px(cur, x + 1 + j, y - 1) - px(cur, x + 1 - j, y + 1));
        bool4 better = s < score;
        score = select(score, s, better);
        sp = select(sp, (px(cur, x + j, y - 1) + px(cur, x - j, y + 1)) * 0.5, better);
    }
    for (int j = 1; j <= 2; j++) {
        float4 s = abs(px(cur, x - 1 + j, y - 1) - px(cur, x - 1 - j, y + 1)) + abs(px(cur, x + j, y - 1) - px(cur, x - j, y + 1)) + abs(px(cur, x + 1 + j, y - 1) - px(cur, x + 1 - j, y + 1));
        bool4 better = s < score;
        score = select(score, s, better);
        sp = select(sp, (px(cur, x + j, y - 1) + px(cur, x - j, y + 1)) * 0.5, better);
    }

    // Keep the spatial guess within what the neighbouring fields allow (yadif mode 0's extra check).
    float4 b = ((p.early ? px(prev, x, y - 2) : px(cur, x, y - 2)) + (p.early ? px(cur, x, y - 2) : px(next, x, y - 2))) * 0.5;
    float4 f = ((p.early ? px(prev, x, y + 2) : px(cur, x, y + 2)) + (p.early ? px(cur, x, y + 2) : px(next, x, y + 2))) * 0.5;
    float4 mx = max(max(d - e, d - c), min(b - c, f - e));
    float4 mn = min(min(d - e, d - c), max(b - c, f - e));
    diff = max(max(diff, mn), -mx);
    out.write(clamp(sp, d - diff, d + diff), gid);
}
"#;

#[repr(C)]
struct Params {
    keep: u32,
    early: u32,
}

/// A deinterlacer for pictures of one size.
pub struct Deinterlacer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cache: CFRetained<CVMetalTextureCache>,
    pool: CFRetained<CVPixelBufferPool>,
    width: usize,
    height: usize,
}

// SAFETY: Metal devices, queues and pipeline states, and CoreVideo caches and pools, may be used from any thread;
// a Deinterlacer is used by one thread at a time (`&mut self`).
unsafe impl Send for Deinterlacer {}

/// Take ownership of a +1 reference a successful create call returned through `out`.
///
/// # Safety
/// `out` must be what the call just returned (an object it created for us), and `st` its status.
unsafe fn adopt<T: objc2_core_foundation::Type>(st: i32, out: *mut T, what: &str) -> anyhow::Result<CFRetained<T>> {
    ensure!(st == 0, "{what}: CVReturn {st}");
    let out = NonNull::new(out).with_context(|| format!("{what}: nothing returned"))?;
    // SAFETY: by the contract above, a +1 reference we now own.
    Ok(unsafe { CFRetained::from_raw(out) })
}

impl Deinterlacer {
    pub fn new(width: usize, height: usize) -> anyhow::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("no Metal command queue")?;
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(SHADER), None).map_err(|e| anyhow!("deinterlacer shader: {e}"))?;
        let function = library.newFunctionWithName(&NSString::from_str("yadif")).context("no yadif kernel")?;
        let pipeline = device.newComputePipelineStateWithFunction_error(&function).map_err(|e| anyhow!("deinterlacer pipeline: {e}"))?;

        let mut cache: *mut CVMetalTextureCache = null_mut();
        // SAFETY: a live device; out-parameter, adopted right after.
        let cache = unsafe {
            let st = CVMetalTextureCache::create(None, None, &device, None, NonNull::from(&mut cache));
            adopt(st, cache, "Metal texture cache")?
        };
        // Output buffers: 420v at this size, backed by IOSurfaces Metal can write.
        // SAFETY: CoreVideo's exported constant strings, initialised before main and never changed.
        let (k_format, k_width, k_height, k_surface, k_metal) = unsafe {
            (
                kCVPixelBufferPixelFormatTypeKey,
                kCVPixelBufferWidthKey,
                kCVPixelBufferHeightKey,
                kCVPixelBufferIOSurfacePropertiesKey,
                kCVPixelBufferMetalCompatibilityKey,
            )
        };
        let (format, w, h) = (CFNumber::new_i32(0x3432_3076), CFNumber::new_i32(width as i32), CFNumber::new_i32(height as i32));
        let surface = CFDictionary::<CFString, CFType>::empty();
        let attributes = CFDictionary::<CFString, CFType>::from_slices(
            &[k_format, k_width, k_height, k_surface, k_metal],
            &[&format, &w, &h, &surface, CFBoolean::new(true)],
        );
        let mut pool: *mut CVPixelBufferPool = null_mut();
        // SAFETY: a live attributes dictionary; out-parameter, adopted right after.
        let pool = unsafe {
            let st = CVPixelBufferPool::create(None, None, Some(attributes.as_ref()), NonNull::from(&mut pool));
            adopt(st, pool, "pixel buffer pool")?
        };
        Ok(Self { device, queue, pipeline, cache, pool, width, height })
    }

    /// A plane of a pixel buffer as a Metal texture (keep it alive until the GPU is done with it).
    fn texture(&self, pb: &PixelBuffer, plane: usize) -> anyhow::Result<CFRetained<CVMetalTexture>> {
        let (format, w, h) =
            if plane == 0 { (MTLPixelFormat::R8Unorm, self.width, self.height) } else { (MTLPixelFormat::RG8Unorm, self.width / 2, self.height / 2) };
        let mut tex: *mut CVMetalTexture = null_mut();
        // SAFETY: a live cache and pixel buffer; out-parameter, adopted right after.
        unsafe {
            let st = CVMetalTextureCache::create_texture_from_image(None, &self.cache, pb.cv(), None, format, w, h, plane, NonNull::from(&mut tex));
            adopt(st, tex, &format!("Metal texture for plane {plane}"))
        }
    }

    /// The picture for one field of `cur`: the lines of parity `keep` (0 top, 1 bottom) from `cur`, the others
    /// made up. `early` says whether that field comes first in its frame (then the missing lines' nearest
    /// fields are in `prev` and `cur`; otherwise in `cur` and `next`).
    pub fn field(&mut self, prev: &PixelBuffer, cur: &PixelBuffer, next: &PixelBuffer, keep: u32, early: bool) -> anyhow::Result<PixelBuffer> {
        // Metal hands back autoreleased objects; a recording's thread has no pool of its own to drain them, so each
        // picture gets one (without it, memory grew ~2 KB a picture: ~400 MB over a three-hour game).
        objc2::rc::autoreleasepool(|_| self.field_in_pool(prev, cur, next, keep, early))
    }

    fn field_in_pool(&mut self, prev: &PixelBuffer, cur: &PixelBuffer, next: &PixelBuffer, keep: u32, early: bool) -> anyhow::Result<PixelBuffer> {
        let mut out: *mut CVPixelBuffer = null_mut();
        // SAFETY: a live pool; out-parameter, adopted right after.
        let out = PixelBuffer::new(unsafe {
            let st = CVPixelBufferPool::create_pixel_buffer(None, &self.pool, NonNull::from(&mut out));
            adopt(st, out, "pixel buffer from pool")?
        });
        let params = Params { keep, early: u32::from(early) };
        let cb = self.queue.commandBuffer().context("no command buffer")?;
        let enc = cb.computeCommandEncoder().context("no compute encoder")?;
        enc.setComputePipelineState(&self.pipeline);
        let mut keep_alive = Vec::new();
        for plane in 0..2 {
            let textures = [self.texture(prev, plane)?, self.texture(cur, plane)?, self.texture(next, plane)?, self.texture(&out, plane)?];
            for (i, t) in textures.iter().enumerate() {
                let texture = CVMetalTextureGetTexture(t).context("a texture without a Metal texture")?;
                // SAFETY: a live texture, bound for this dispatch only.
                unsafe { enc.setTexture_atIndex(Some(&texture), i) };
            }
            // SAFETY: `params` lives until the encoder has copied it (setBytes copies immediately).
            unsafe { enc.setBytes_length_atIndex(NonNull::from(&params).cast(), size_of::<Params>(), 0) };
            let (w, h) = if plane == 0 { (self.width, self.height) } else { (self.width / 2, self.height / 2) };
            enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: w, height: h, depth: 1 }, MTLSize { width: 16, height: 16, depth: 1 });
            keep_alive.extend(textures);
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        drop(keep_alive);
        self.cache.flush(0);
        let _ = &self.device;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apple_media::video::{Decoder, PixelBuffer};

    /// Current resident memory (bytes), from the kernel.
    fn resident() -> u64 {
        // SAFETY: task_info with the matching flavor and a correctly sized out-struct.
        unsafe {
            let mut info: libc::mach_task_basic_info = std::mem::zeroed();
            let mut count = (size_of::<libc::mach_task_basic_info>() / size_of::<libc::natural_t>()) as u32;
            #[allow(deprecated)]
            let task = libc::mach_task_self();
            libc::task_info(task, libc::MACH_TASK_BASIC_INFO, (&raw mut info).cast(), &raw mut count);
            info.resident_size
        }
    }

    /// Deinterlacing thousands of pictures must not grow memory (a long recording does this ~200,000 times an hour).
    /// Uses a real broadcast capture when `DVR_TS_SAMPLE` names one.
    #[test]
    fn memory_stays_flat_over_thousands_of_pictures() {
        let Ok(path) = std::env::var("DVR_TS_SAMPLE") else { return };
        let bytes = std::fs::read(path).unwrap();
        let mut d = mpegts::Demuxer::new();
        let mut units = d.push(&bytes);
        units.extend(d.flush());
        let mut dec: Option<Decoder> = None;
        let mut pics: Vec<PixelBuffer> = Vec::new();
        for u in &units {
            if let mpegts::Unit::Video { pts, dts, data, .. } = u {
                if dec.is_none() {
                    let Some((w, h)) = mpegts::mpeg2::size(data) else { continue };
                    dec = Some(Decoder::new(apple_media::video::CODEC_MPEG2, w, h, 90_000).unwrap());
                }
                pics.extend(dec.as_mut().unwrap().decode(data, *dts, *pts).unwrap().into_iter().map(|p| p.picture));
                if pics.len() >= 8 {
                    break;
                }
            }
        }
        let (w, h) = pics[0].size();
        let mut di = Deinterlacer::new(w, h).unwrap();
        let mut run = |n: usize| {
            for i in 0..n {
                let k = i % (pics.len() - 2);
                let out = di.field(&pics[k], &pics[k + 1], &pics[k + 2], (i % 2) as u32, i % 2 == 0).unwrap();
                drop(out);
            }
        };
        run(300);
        let before = resident();
        run(3000);
        let grown = resident().saturating_sub(before) as f64 / 1e6;
        eprintln!("resident memory grew {grown:.1} MB over 3000 pictures");
        assert!(grown < 1.0, "{grown:.1} MB: a leak of {:.0} bytes a picture", grown * 1e6 / 3000.0);
    }
}
