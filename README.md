# avkit

Broadcast TV media in Rust, on Apple's hardware codecs. No ffmpeg.

Used by [barm/adbreak](https://github.com/barm/adbreak) (commercial-break detection) among others.

| Crate | What it does | Platform |
|---|---|---|
| `mpegts` | MPEG transport stream demux for ATSC broadcasts: video, main audio and A/53 captions, with timestamps | any |
| `captions` | CEA-608 captions (CC1) to WebVTT: pop-on, roll-up and paint-on | any |
| `fmp4` | Fragmented MP4 (HEVC + AAC) and byte-range HLS playlists: read and write the single-file form | any |
| `apple-media` | VideoToolbox MPEG-2/H.264/HEVC decode and HEVC encode, AudioToolbox AC-3/E-AC-3/AAC | macOS |
| `metal-analysis` | apple-media's analysis reductions (grey picture, colour means) on the GPU, the same values | macOS |
| `metal-deinterlace` | yadif deinterlacing on the GPU over CoreVideo pixel buffers | macOS |
| `fmp4-scan` | a fragmented MP4 recording to small grey pictures and mono sound for analysis, and optionally one colour picture a second | macOS |
| `apple-coreml` | run Core ML models on float32 tensors (load an .mlpackage or .mlmodelc, predict) | macOS |
| `fmp4-shrink` | a small copy of a recording (360p HEVC, mono AAC) with a byte-range playlist, in hardware | macOS |

The pure crates, and those built only on the others (`fmp4-scan`, `fmp4-shrink`), `#![forbid(unsafe_code)]`. In `apple-media`, every framework call Rust can't check is in `vt.rs`
or `at.rs` behind a safe API; callbacks never share mutable state with VideoToolbox's threads (`outbox.rs`).

Use one as a git dependency pinned by rev:

```toml
fmp4 = { git = "https://github.com/barm/avkit", rev = "<commit>" }
```

## License

This project is dual-licensed. You may use it under either of these licenses, whichever you prefer:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

Both grants apply to everyone; the choice between them is yours as the user, and there is nothing to ask for.

Unless you say otherwise, any contribution you intentionally submit for inclusion in this project, as defined in
the Apache-2.0 license, is dual-licensed as above, with no additional terms or conditions.
