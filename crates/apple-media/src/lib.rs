//! Broadcast media with Apple's frameworks, no ffmpeg: VideoToolbox decoding (MPEG-2, in software; HEVC) and HEVC
//! encoding (in hardware), AudioToolbox AC-3 decoding and 5.1 AAC encoding, and small decoders for analysis.
//! Through the `objc2-*` bindings: every call into the frameworks that Rust cannot check is in `vt` (VideoToolbox,
//! CoreMedia, CoreVideo) or `at` (AudioToolbox), behind safe functions; the codecs on top are safe code, and the
//! callbacks only send through an `outbox`. macOS only.

#![cfg(target_os = "macos")]

pub mod analysis;
mod at;
pub mod audio;
mod outbox;
pub mod thread;
pub mod video;
mod vt;
