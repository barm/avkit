# avkit agent guide

Media crates shared by barm/dvr and barm/adbreak (git dependencies pinned by rev: a change reaches them only when
their pin moves). avkit depends on neither.

- One job per crate; new self-contained functionality gets its own crate.
- `unsafe`: the pure crates and those built on the others (`mpegts`, `cea608`, `fmp4`, `fmp4-scan`, `fmp4-shrink`) `#![forbid(unsafe_code)]`. FFI to Apple's
  frameworks lives only in `apple-media` (`vt.rs`, `at.rs`), `metal-deinterlace` and `apple-coreml`.
- Test fixtures live in each crate's `tests/fixtures`.
- Before committing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Land changes through squash-merged PRs. CI runs on GitHub's macOS runners (public repo).
