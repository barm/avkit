//! Broadcast media with Apple's frameworks, no ffmpeg: VideoToolbox decoding (MPEG-2, in software; HEVC) and HEVC
//! encoding (in hardware), AudioToolbox AC-3 decoding and 5.1 AAC encoding, and small decoders for analysis.
//! Plain FFI to the C APIs (`sys`); every `unsafe` call is in this crate. macOS only.

#![cfg(target_os = "macos")]

pub mod analysis;
pub mod audio;
mod outbox;
pub mod sys;
pub mod video;
