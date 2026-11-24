//! FFI declarations for the macOS media frameworks the native pipeline uses (CoreFoundation, CoreVideo,
//! CoreMedia, VideoToolbox, AudioToolbox): only what is called, with the C names. The calls themselves, and
//! their `unsafe`, are in the modules that use them.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

use std::ffi::c_void;

pub type OSStatus = i32;
pub type CFTypeRef = *const c_void;
pub type CFAllocatorRef = *const c_void;
pub type CFDictionaryRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CMFormatDescriptionRef = *const c_void;
pub type CMBlockBufferRef = *const c_void;
pub type CMSampleBufferRef = *const c_void;
pub type CVImageBufferRef = *const c_void;
pub type VTDecompressionSessionRef = *const c_void;
pub type AudioConverterRef = *const c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const INVALID: CMTime = CMTime { value: 0, timescale: 0, flags: 0, epoch: 0 };
    pub fn new(value: i64, timescale: u32) -> Self {
        CMTime { value, timescale: timescale as i32, flags: 1, epoch: 0 } // kCMTimeFlags_Valid
    }
}

#[repr(C)]
pub struct CMSampleTimingInfo {
    pub duration: CMTime,
    pub presentationTimeStamp: CMTime,
    pub decodeTimeStamp: CMTime,
}

pub type VTDecompressionOutputCallback = extern "C" fn(*mut c_void, *mut c_void, OSStatus, u32, CVImageBufferRef, CMTime, CMTime);

#[repr(C)]
pub struct VTDecompressionOutputCallbackRecord {
    pub callback: VTDecompressionOutputCallback,
    pub refcon: *mut c_void,
}

#[repr(C)]
pub struct CFDictionaryCallBacks {
    pub _private: [u8; 0],
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFRelease(cf: CFTypeRef);
    pub fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        count: isize,
        key_callbacks: *const CFDictionaryCallBacks,
        value_callbacks: *const CFDictionaryCallBacks,
    ) -> CFDictionaryRef;
    pub fn CFNumberCreate(allocator: CFAllocatorRef, the_type: isize, value: *const c_void) -> CFTypeRef;
    pub static kCFTypeDictionaryKeyCallBacks: CFDictionaryCallBacks;
    pub static kCFTypeDictionaryValueCallBacks: CFDictionaryCallBacks;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    pub static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    pub fn CVPixelBufferLockBaseAddress(pb: CVImageBufferRef, flags: u64) -> i32;
    pub fn CVPixelBufferUnlockBaseAddress(pb: CVImageBufferRef, flags: u64) -> i32;
    pub fn CVPixelBufferGetBaseAddressOfPlane(pb: CVImageBufferRef, plane: usize) -> *const u8;
    pub fn CVPixelBufferGetBytesPerRowOfPlane(pb: CVImageBufferRef, plane: usize) -> usize;
    pub fn CVPixelBufferGetWidthOfPlane(pb: CVImageBufferRef, plane: usize) -> usize;
    pub fn CVPixelBufferGetHeightOfPlane(pb: CVImageBufferRef, plane: usize) -> usize;
    pub fn CVPixelBufferGetPixelFormatType(pb: CVImageBufferRef) -> u32;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    pub fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        allocator: CFAllocatorRef,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_unit_header_length: i32,
        extensions: CFDictionaryRef,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    pub fn CMBlockBufferCreateWithMemoryBlock(
        allocator: CFAllocatorRef,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: CFAllocatorRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        out: *mut CMBlockBufferRef,
    ) -> OSStatus;
    pub fn CMBlockBufferReplaceDataBytes(source: *const c_void, dest: CMBlockBufferRef, offset: usize, length: usize) -> OSStatus;
    pub fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        data_buffer: CMBlockBufferRef,
        format: CMFormatDescriptionRef,
        num_samples: isize,
        num_timing: isize,
        timing: *const CMSampleTimingInfo,
        num_sizes: isize,
        sizes: *const usize,
        out: *mut CMSampleBufferRef,
    ) -> OSStatus;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    pub fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        format: CMFormatDescriptionRef,
        decoder_specification: CFDictionaryRef,
        image_buffer_attributes: CFDictionaryRef,
        callback: *const VTDecompressionOutputCallbackRecord,
        out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    pub fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample: CMSampleBufferRef,
        flags: u32,
        frame_refcon: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    pub fn VTDecompressionSessionWaitForAsynchronousFrames(session: VTDecompressionSessionRef) -> OSStatus;
    pub fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AudioStreamBasicDescription {
    pub mSampleRate: f64,
    pub mFormatID: u32,
    pub mFormatFlags: u32,
    pub mBytesPerPacket: u32,
    pub mFramesPerPacket: u32,
    pub mBytesPerFrame: u32,
    pub mChannelsPerFrame: u32,
    pub mBitsPerChannel: u32,
    pub mReserved: u32,
}

#[repr(C)]
pub struct AudioBuffer {
    pub mNumberChannels: u32,
    pub mDataByteSize: u32,
    pub mData: *mut c_void,
}

#[repr(C)]
pub struct AudioBufferList {
    pub mNumberBuffers: u32,
    pub mBuffers: [AudioBuffer; 1],
}

#[repr(C)]
#[derive(Default)]
pub struct AudioStreamPacketDescription {
    pub mStartOffset: i64,
    pub mVariableFramesInPacket: u32,
    pub mDataByteSize: u32,
}

pub type InputProc = extern "C" fn(AudioConverterRef, *mut u32, *mut AudioBufferList, *mut *mut AudioStreamPacketDescription, *mut c_void) -> OSStatus;

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    pub fn AudioConverterNew(input: *const AudioStreamBasicDescription, output: *const AudioStreamBasicDescription, out: *mut AudioConverterRef) -> OSStatus;
    pub fn AudioConverterSetProperty(converter: AudioConverterRef, property: u32, size: u32, data: *const c_void) -> OSStatus;
    pub fn AudioConverterFillComplexBuffer(
        converter: AudioConverterRef,
        input: InputProc,
        user: *mut c_void,
        packets: *mut u32,
        output: *mut AudioBufferList,
        descriptions: *mut AudioStreamPacketDescription,
    ) -> OSStatus;
    pub fn AudioConverterDispose(converter: AudioConverterRef) -> OSStatus;
}

// ---- more CoreFoundation / CoreMedia / VideoToolbox for full-frame decoding and encoding ----

pub type CFDataRef = *const c_void;
pub type CFArrayRef = *const c_void;
pub type CFBooleanRef = *const c_void;
pub type VTCompressionSessionRef = *const c_void;

pub type VTCompressionOutputCallback = extern "C" fn(*mut c_void, *mut c_void, OSStatus, u32, CMSampleBufferRef);

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    pub fn CFDictionaryGetValue(dict: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
    pub fn CFDataGetBytePtr(data: CFDataRef) -> *const u8;
    pub fn CFDataGetLength(data: CFDataRef) -> isize;
    pub fn CFArrayGetCount(array: CFArrayRef) -> isize;
    pub fn CFArrayGetValueAtIndex(array: CFArrayRef, index: isize) -> CFTypeRef;
    pub fn CFStringCreateWithCString(alloc: CFAllocatorRef, s: *const std::ffi::c_char, encoding: u32) -> CFStringRef;
    pub static kCFBooleanTrue: CFBooleanRef;
    pub static kCFBooleanFalse: CFBooleanRef;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    pub fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        alloc: CFAllocatorRef,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_unit_header_length: i32,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    pub fn CMVideoFormatDescriptionCreate(
        alloc: CFAllocatorRef,
        codec: u32,
        width: i32,
        height: i32,
        extensions: CFDictionaryRef,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    pub fn CMSampleBufferGetDataBuffer(sample: CMSampleBufferRef) -> CMBlockBufferRef;
    pub fn CMSampleBufferGetFormatDescription(sample: CMSampleBufferRef) -> CMFormatDescriptionRef;
    pub fn CMSampleBufferGetPresentationTimeStamp(sample: CMSampleBufferRef) -> CMTime;
    pub fn CMSampleBufferGetDecodeTimeStamp(sample: CMSampleBufferRef) -> CMTime;
    pub fn CMSampleBufferGetSampleAttachmentsArray(sample: CMSampleBufferRef, create: u8) -> CFArrayRef;
    pub fn CMBlockBufferGetDataLength(block: CMBlockBufferRef) -> usize;
    pub fn CMBlockBufferCopyDataBytes(block: CMBlockBufferRef, offset: usize, length: usize, dest: *mut c_void) -> OSStatus;
    pub fn CMFormatDescriptionGetExtension(desc: CMFormatDescriptionRef, key: CFStringRef) -> CFTypeRef;
    pub static kCMSampleAttachmentKey_NotSync: CFStringRef;
    pub static kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms: CFStringRef;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    pub fn CVPixelBufferGetWidth(pb: CVImageBufferRef) -> usize;
    pub fn CVPixelBufferGetHeight(pb: CVImageBufferRef) -> usize;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    pub fn VTCompressionSessionCreate(
        alloc: CFAllocatorRef,
        width: i32,
        height: i32,
        codec: u32,
        encoder_specification: CFDictionaryRef,
        source_image_buffer_attributes: CFDictionaryRef,
        compressed_data_allocator: CFAllocatorRef,
        callback: VTCompressionOutputCallback,
        refcon: *mut c_void,
        out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    pub fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image: CVImageBufferRef,
        pts: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        frame_refcon: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    pub fn VTCompressionSessionCompleteFrames(session: VTCompressionSessionRef, until: CMTime) -> OSStatus;
    pub fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;
    pub fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);
    pub fn VTSessionSetProperty(session: CFTypeRef, key: CFStringRef, value: CFTypeRef) -> OSStatus;
    pub static kVTCompressionPropertyKey_RealTime: CFStringRef;
    pub static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
    pub static kVTCompressionPropertyKey_Quality: CFStringRef;
    pub static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
    pub static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    pub static kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration: CFStringRef;
    pub static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
    pub static kVTProfileLevel_HEVC_Main_AutoLevel: CFStringRef;
    pub static kVTEncodeFrameOptionKey_ForceKeyFrame: CFStringRef;
    pub static kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: CFStringRef;
}

pub const CODEC_MPEG2: u32 = u32::from_be_bytes(*b"mp2v");
pub const CODEC_HEVC: u32 = u32::from_be_bytes(*b"hvc1");
pub const CF_NUMBER_SINT64: isize = 4;
pub const CF_NUMBER_FLOAT64: isize = 6;

// ---- AudioToolbox properties for decoding AC-3 and encoding AAC ----

#[repr(C)]
pub struct AudioChannelLayout {
    pub mChannelLayoutTag: u32,
    pub mChannelBitmap: u32,
    pub mNumberChannelDescriptions: u32,
    pub mChannelDescriptions: [u8; 20],
}

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    pub fn AudioConverterGetProperty(converter: AudioConverterRef, property: u32, size: *mut u32, data: *mut c_void) -> OSStatus;
    pub fn AudioConverterGetPropertyInfo(converter: AudioConverterRef, property: u32, size: *mut u32, writable: *mut u8) -> OSStatus;
}

pub const FORMAT_AC3: u32 = u32::from_be_bytes(*b"ac-3");
pub const FORMAT_EAC3: u32 = u32::from_be_bytes(*b"ec-3");
pub const PROP_INPUT_LAYOUT: u32 = u32::from_be_bytes(*b"icl ");
pub const PROP_OUTPUT_LAYOUT: u32 = u32::from_be_bytes(*b"ocl ");
pub const PROP_ENCODE_BITRATE: u32 = u32::from_be_bytes(*b"brat");
pub const PROP_COMPRESSION_COOKIE: u32 = u32::from_be_bytes(*b"cmgc");
pub const PROP_MAX_OUTPUT_PACKET: u32 = u32::from_be_bytes(*b"xops");
/// L R C LFE Ls Rs.
pub const LAYOUT_MPEG_5_1_A: u32 = (121 << 16) | 6;

// ---- CoreVideo: Metal textures over pixel buffers, and pixel buffer pools ----

pub type CVMetalTextureCacheRef = *const c_void;
pub type CVMetalTextureRef = *const c_void;
pub type CVPixelBufferPoolRef = *const c_void;

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    pub fn CVMetalTextureCacheCreate(
        alloc: CFAllocatorRef,
        cache_attrs: CFDictionaryRef,
        device: *mut c_void,
        texture_attrs: CFDictionaryRef,
        out: *mut CVMetalTextureCacheRef,
    ) -> i32;
    pub fn CVMetalTextureCacheCreateTextureFromImage(
        alloc: CFAllocatorRef,
        cache: CVMetalTextureCacheRef,
        image: CVImageBufferRef,
        texture_attrs: CFDictionaryRef,
        pixel_format: usize,
        width: usize,
        height: usize,
        plane: usize,
        out: *mut CVMetalTextureRef,
    ) -> i32;
    pub fn CVMetalTextureGetTexture(texture: CVMetalTextureRef) -> *mut c_void;
    pub fn CVMetalTextureCacheFlush(cache: CVMetalTextureCacheRef, options: u64);
    pub fn CVPixelBufferPoolCreate(alloc: CFAllocatorRef, pool_attrs: CFDictionaryRef, buffer_attrs: CFDictionaryRef, out: *mut CVPixelBufferPoolRef) -> i32;
    pub fn CVPixelBufferPoolCreatePixelBuffer(alloc: CFAllocatorRef, pool: CVPixelBufferPoolRef, out: *mut CVImageBufferRef) -> i32;
    pub static kCVPixelBufferWidthKey: CFStringRef;
    pub static kCVPixelBufferHeightKey: CFStringRef;
    pub static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
    pub static kCVPixelBufferMetalCompatibilityKey: CFStringRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFDictionaryCreateMutable(
        alloc: CFAllocatorRef,
        capacity: isize,
        key_cb: *const CFDictionaryCallBacks,
        value_cb: *const CFDictionaryCallBacks,
    ) -> CFDictionaryRef;
    pub fn CFDictionarySetValue(dict: CFDictionaryRef, key: CFTypeRef, value: CFTypeRef);
}

/// MTLPixelFormat values for the planes of 8-bit 4:2:0 biplanar buffers.
pub const MTL_R8_UNORM: usize = 10;
pub const MTL_RG8_UNORM: usize = 30;
