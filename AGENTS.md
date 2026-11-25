# avkit agent guide

Media crates used by other projects, barm/adbreak among them, as git dependencies pinned by rev (a change reaches
them only when their pin moves). avkit depends on none of them. This repository is public: don't mention private
projects, hosts or paths in code, docs or commit messages.

- One job per crate; new self-contained functionality gets its own crate.
- `unsafe`: the pure crates and those built on the others (`mpegts`, `captions`, `fmp4`, `fmp4-scan`, `fmp4-shrink`) `#![forbid(unsafe_code)]`. FFI to Apple's
  frameworks lives only in `apple-media` (`vt.rs`, `at.rs`), `metal-deinterlace` and `apple-coreml`.
- Test fixtures live in each crate's `tests/fixtures`.
- Before committing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Land changes through squash-merged PRs. CI runs on GitHub's macOS runners (public repo).
