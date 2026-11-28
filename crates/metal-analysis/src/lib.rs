#![cfg(target_os = "macos")]

//! `apple_media::analysis`'s reductions on the GPU (Metal), for pictures that already live there: a transcoder's
//! decoded or deinterlaced frames, which are slow for the CPU to read. The small grey picture
//! ([`Reducer::luma`]) and the colour picture ([`Reducer::rgb`]) come out exactly as the CPU versions give them:
//! the GPU adds whole bytes over the same cells (`analysis::cells`), and for the grey picture takes the same
//! integer mean and stretch (`analysis::stretch`); for colour it hands the sums back and the CPU finishes as
//! `analysis::rgb` does (`analysis::rgb_from_means`). Pixel buffers are read as Metal textures in place
//! (CoreVideo's texture cache).

use anyhow::{Context, anyhow, ensure};
use apple_media::analysis;
use apple_media::video::PixelBuffer;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetWidthOfPlane};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLLibrary, MTLPixelFormat, MTLResourceOptions, MTLSize,
};
use std::ptr::{NonNull, null_mut};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// A grid of `dw`×`dh` cells over a `w`×`h` plane of `channels` bytes a pixel.
struct Grid { uint w; uint h; uint dw; uint dh; uint channels; uint video_range; };

// The source pixels cell `o` of `n` covers along a side of `len`: [a, b), at least one (analysis::cells).
static inline uint2 cell(uint o, uint len, uint n) {
    uint a = o * len / n;
    return uint2(a, max((o + 1) * len / n, a + 1));
}

// The grey picture: each cell's integer mean, video range stretched to full (analysis::stretch).
kernel void grey(texture2d<uint, access::read> plane [[texture(0)]],
                 device uchar *out [[buffer(0)]],
                 constant Grid &g [[buffer(1)]],
                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= g.dw || gid.y >= g.dh) return;
    uint2 xs = cell(gid.x, g.w, g.dw), ys = cell(gid.y, g.h, g.dh);
    uint sum = 0;
    for (uint y = ys.x; y < ys.y; y++)
        for (uint x = xs.x; x < xs.y; x++)
            sum += plane.read(uint2(x, y)).r;
    uint mean = sum / ((xs.y - xs.x) * (ys.y - ys.x));
    uint v = g.video_range ? min(((mean > 16 ? mean - 16 : 0) * 255 + 109) / 219, 255u) : mean;
    out[gid.y * g.dw + gid.x] = uchar(v);
}

// Each cell's sum, per channel (one or two), interleaved like the plane.
kernel void sums(texture2d<uint, access::read> plane [[texture(0)]],
                 device uint *out [[buffer(0)]],
                 constant Grid &g [[buffer(1)]],
                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= g.dw || gid.y >= g.dh) return;
    uint2 xs = cell(gid.x, g.w, g.dw), ys = cell(gid.y, g.h, g.dh);
    uint s0 = 0, s1 = 0;
    for (uint y = ys.x; y < ys.y; y++)
        for (uint x = xs.x; x < xs.y; x++) {
            uint4 p = plane.read(uint2(x, y));
            s0 += p.r;
            s1 += p.g;
        }
    uint i = (gid.y * g.dw + gid.x) * g.channels;
    out[i] = s0;
    if (g.channels == 2) out[i + 1] = s1;
}
"#;

#[repr(C)]
struct Grid {
    w: u32,
    h: u32,
    dw: u32,
    dh: u32,
    channels: u32,
    video_range: u32,
}

#[derive(Clone, Copy)]
enum Kernel {
    Grey { video_range: bool },
    Sums,
}

/// The reductions on the GPU. Make one and keep it: it compiles its shaders.
pub struct Reducer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    grey: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    sums: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cache: CFRetained<CVMetalTextureCache>,
}

// SAFETY: the Metal objects are used from one thread at a time (`&mut self`); Metal devices, queues and pipelines
// may be used from any thread.
unsafe impl Send for Reducer {}

impl Reducer {
    pub fn new() -> anyhow::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("no Metal command queue")?;
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(SHADER), None).map_err(|e| anyhow!("analysis shaders: {e}"))?;
        let pipeline = |name: &str| -> anyhow::Result<_> {
            let f = library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no {name} kernel"))?;
            device.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow!("{name} pipeline: {e}"))
        };
        let (grey, sums) = (pipeline("grey")?, pipeline("sums")?);
        let mut cache: *mut CVMetalTextureCache = null_mut();
        // SAFETY: `cache` is written by the call; on success it holds a +1 reference we take.
        let cache = unsafe {
            let st = CVMetalTextureCache::create(None, None, &device, None, NonNull::from(&mut cache));
            ensure!(st == 0, "Metal texture cache: CVReturn {st}");
            CFRetained::from_raw(NonNull::new(cache).context("no Metal texture cache")?)
        };
        Ok(Self { device, queue, grey, sums, cache })
    }

    /// A plane of `pb` as an integer texture (`R8Uint` for luma, `RG8Uint` for chroma) and its size.
    fn texture(&self, pb: &PixelBuffer, plane: usize) -> anyhow::Result<(CFRetained<CVMetalTexture>, usize, usize)> {
        let (w, h) = (CVPixelBufferGetWidthOfPlane(pb.cv(), plane), CVPixelBufferGetHeightOfPlane(pb.cv(), plane));
        let format = if plane == 0 { MTLPixelFormat::R8Uint } else { MTLPixelFormat::RG8Uint };
        let mut tex: *mut CVMetalTexture = null_mut();
        // SAFETY: `tex` is written by the call; on success it holds a +1 reference we take.
        let tex = unsafe {
            let st = CVMetalTextureCache::create_texture_from_image(None, &self.cache, pb.cv(), None, format, w, h, plane, NonNull::from(&mut tex));
            ensure!(st == 0, "Metal texture for plane {plane}: CVReturn {st}");
            CFRetained::from_raw(NonNull::new(tex).context("no Metal texture")?)
        };
        Ok((tex, w, h))
    }

    /// Runs a kernel over plane `plane` of `pb` on a `dw`×`dh` grid; returns its output buffer.
    fn run(&mut self, kernel: Kernel, pb: &PixelBuffer, plane: usize, dw: usize, dh: usize) -> anyhow::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let channels = if plane == 0 { 1 } else { 2 };
        let (pipeline, bytes, video_range) = match kernel {
            Kernel::Grey { video_range } => (self.grey.clone(), dw * dh, video_range),
            Kernel::Sums => (self.sums.clone(), dw * dh * channels * 4, false),
        };
        objc2::rc::autoreleasepool(|_| {
            let (tex, w, h) = self.texture(pb, plane)?;
            let grid =
                Grid { w: w as u32, h: h as u32, dw: dw as u32, dh: dh as u32, channels: if plane == 0 { 1 } else { 2 }, video_range: u32::from(video_range) };
            let out = self.device.newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared).context("no Metal buffer")?;
            let cb = self.queue.commandBuffer().context("no command buffer")?;
            let enc = cb.computeCommandEncoder().context("no compute encoder")?;
            enc.setComputePipelineState(&pipeline);
            let texture = CVMetalTextureGetTexture(&tex).context("a texture without a Metal texture")?;
            // SAFETY: the texture, buffer and parameters outlive the command buffer, which is waited on below.
            unsafe {
                enc.setTexture_atIndex(Some(&texture), 0);
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                enc.setBytes_length_atIndex(NonNull::from(&grid).cast(), size_of::<Grid>(), 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: dw, height: dh, depth: 1 }, MTLSize { width: 16, height: 16, depth: 1 });
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            drop(tex);
            self.cache.flush(0);
            Ok(out)
        })
    }

    /// `analysis::luma` on the GPU: the picture's luma shrunk to `dw`×`dh`, stretched to full range.
    pub fn luma(&mut self, pb: &PixelBuffer, dw: usize, dh: usize) -> anyhow::Result<Vec<u8>> {
        let video_range = analysis::video_range(pb)?;
        let out = self.run(Kernel::Grey { video_range }, pb, 0, dw, dh)?;
        // SAFETY: a shared buffer of at least `dw * dh` bytes, written by the finished command buffer.
        Ok(unsafe { std::slice::from_raw_parts(out.contents().as_ptr().cast::<u8>(), dw * dh) }.to_vec())
    }

    /// `analysis::rgb` on the GPU: the picture as planar RGB (0.0–1.0), `size`×`size`.
    pub fn rgb(&mut self, pb: &PixelBuffer, size: usize) -> anyhow::Result<Vec<f32>> {
        analysis::video_range(pb)?;
        let means = |r: &mut Self, plane: usize, channels: usize| -> anyhow::Result<Vec<f32>> {
            let (w, h) = (CVPixelBufferGetWidthOfPlane(pb.cv(), plane), CVPixelBufferGetHeightOfPlane(pb.cv(), plane));
            let n = size * size * channels;
            let out = r.run(Kernel::Sums, pb, plane, size, size)?;
            // SAFETY: a shared buffer of at least `n` u32s, written by the finished command buffer.
            let s = unsafe { std::slice::from_raw_parts(out.contents().as_ptr().cast::<u32>(), n) };
            let (xs, ys) = (analysis::cells(w, size), analysis::cells(h, size));
            let mut m = vec![0f32; n];
            for (oy, &(y0, y1)) in ys.iter().enumerate() {
                for (ox, &(x0, x1)) in xs.iter().enumerate() {
                    let count = ((x1 - x0) * (y1 - y0)) as u32;
                    for ch in 0..channels {
                        let i = (oy * size + ox) * channels + ch;
                        m[i] = s[i] as f32 / count as f32;
                    }
                }
            }
            Ok(m)
        };
        let y = means(self, 0, 1)?;
        let c = means(self, 1, 2)?;
        Ok(analysis::rgb_from_means(&y, &c, size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::{CFBoolean, CFDictionary, CFString, CFType};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    };

    /// A `w`×`h` 4:2:0 picture (`420v` or `420f`) of pseudo-random bytes, backed by an IOSurface as decoders' are.
    fn picture(w: usize, h: usize, video_range: bool, seed: u32) -> PixelBuffer {
        let format: u32 = u32::from_be_bytes(if video_range { *b"420v" } else { *b"420f" });
        // SAFETY: CoreVideo calls on a buffer we create, lock while writing inside each plane's bounds, then unlock.
        unsafe {
            let surface = CFDictionary::<CFString, CFType>::empty();
            let attrs = CFDictionary::<CFString, CFType>::from_slices(
                &[kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey],
                &[&surface, CFBoolean::new(true)],
            );
            let mut pb: *mut CVPixelBuffer = null_mut();
            assert_eq!(CVPixelBufferCreate(None, w, h, format, Some(attrs.as_ref()), NonNull::from(&mut pb)), 0);
            let pb = CFRetained::from_raw(NonNull::new(pb).unwrap());
            assert_eq!(CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags(0)), 0);
            let mut s = seed;
            for plane in 0..2 {
                let base = CVPixelBufferGetBaseAddressOfPlane(&pb, plane).cast::<u8>();
                let (pw, ph, stride) =
                    (CVPixelBufferGetWidthOfPlane(&pb, plane), CVPixelBufferGetHeightOfPlane(&pb, plane), CVPixelBufferGetBytesPerRowOfPlane(&pb, plane));
                let bytes = if plane == 0 { pw } else { pw * 2 };
                for y in 0..ph {
                    for x in 0..bytes {
                        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                        *base.add(y * stride + x) = (s >> 16) as u8;
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags(0));
            PixelBuffer::new(pb)
        }
    }

    /// Exactly the CPU's pictures, at every size the break scan meets.
    #[test]
    fn the_gpu_gives_the_cpus_pictures() {
        let mut r = Reducer::new().unwrap();
        for (i, &(w, h)) in [(1280, 720), (1920, 1080), (704, 480), (334, 212)].iter().enumerate() {
            for vr in [true, false] {
                let pb = picture(w, h, vr, i as u32 + 7);
                assert_eq!(r.luma(&pb, 480, 270).unwrap(), analysis::luma(&pb, 480, 270).unwrap(), "grey {w}x{h} video range {vr}");
                assert_eq!(r.luma(&pb, 48, 27).unwrap(), analysis::luma(&pb, 48, 27).unwrap(), "small grey {w}x{h}");
                assert_eq!(r.rgb(&pb, 256).unwrap(), analysis::rgb(&pb, 256).unwrap(), "colour {w}x{h} video range {vr}");
            }
        }
    }

    /// What it costs the CPU, against the CPU versions (`--nocapture` to see).
    #[test]
    fn cpu_time() {
        let mut r = Reducer::new().unwrap();
        let pb = picture(1920, 1080, true, 1);
        let time = |f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..50 {
                f();
            }
            t.elapsed().as_secs_f64() / 50.0 * 1000.0
        };
        let gpu = time(&mut || drop(r.luma(&pb, 480, 270).unwrap()));
        let cpu = time(&mut || drop(analysis::luma(&pb, 480, 270).unwrap()));
        eprintln!("grey 1080 to 480x270: GPU {gpu:.2} ms (wall, mostly waiting), CPU {cpu:.2} ms");
    }
}
