//! Fragmented MP4 for HLS recordings: reading (`read`) and writing (`write`) the single-file form (an init segment,
//! then a `moof` + `mdat` per segment) with one HEVC and one AAC track, and the byte-range playlist over it
//! (`hls`); and cutting segments out without re-encoding (`edit`).

#![forbid(unsafe_code)]

pub mod edit;
pub mod hls;
pub mod read;
pub mod stream;
pub mod write;
