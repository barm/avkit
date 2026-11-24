//! Editing a single-file recording without re-encoding: a copy that leaves out whole segments (each starts with a
//! keyframe, so any of them can go), with every track's decode times moved back by what was left out before it, so
//! the timeline has no holes.

use crate::hls::{ByteRanges, HlsWriter, Layout};
use crate::read::{self, boxes};
use anyhow::{Context, ensure};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// What a cut did: where in the original the left-out stretches were, and how long the result is (seconds).
#[derive(Debug, PartialEq)]
pub struct Cut {
    /// (start, length) of each stretch left out, on the original timeline, merged where they touch.
    pub removed: Vec<(f64, f64)>,
    pub duration: f64,
}

/// Copy the recording `src` (described by `pl`) into `dst_dir` (`media.mp4` and a closed `index.m3u8`), keeping
/// the segments for which `keep(i)` is true.
pub fn drop_segments(src: &Path, pl: &ByteRanges, keep: impl Fn(usize) -> bool, dst_dir: &Path) -> anyhow::Result<Cut> {
    let mut f = std::fs::File::open(src)?;
    let read_range = |f: &mut std::fs::File, off: u64, len: u64| -> anyhow::Result<Vec<u8>> {
        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf).context("the media file is shorter than its playlist")?;
        Ok(buf)
    };
    let init_bytes = read_range(&mut f, pl.init.0, pl.init.1)?;
    let init = read::parse_init(&init_bytes)?;
    let target = pl.segs.windows(2).map(|w| (w[1].start - w[0].start).ceil() as u32).max().unwrap_or(2).max(1);
    let mut out = HlsWriter::create(dst_dir, &init_bytes, target, Layout::SingleFile)?;
    // Per track: decode time to take off (its timescale) because of segments left out before.
    let mut shift: HashMap<u32, u64> = HashMap::new();
    let mut removed: Vec<(f64, f64)> = Vec::new();
    for (i, seg) in pl.segs.iter().enumerate() {
        let secs = pl.segs.get(i + 1).map_or_else(|| seg_secs(&init, &[]), |n| n.start - seg.start);
        let mut frag = read_range(&mut f, seg.off, seg.len)?;
        if keep(i) {
            retime(&mut frag, &shift)?;
            let secs = if i + 1 == pl.segs.len() { seg_secs(&init, &read::parse_fragments(&init, &frag)?) } else { secs };
            out.append_fragment(secs, &frag)?;
        } else {
            let samples = read::parse_fragments(&init, &frag)?;
            for s in &samples {
                *shift.entry(s.track).or_default() += u64::from(s.duration);
            }
            let secs = if i + 1 == pl.segs.len() { seg_secs(&init, &samples) } else { secs };
            match removed.last_mut() {
                Some(last) if (last.0 + last.1 - seg.start).abs() < 0.01 => last.1 += secs,
                _ => removed.push((seg.start, secs)),
            }
        }
    }
    let duration = out.finish()?;
    Ok(Cut { removed, duration })
}

/// A fragment's length from its video samples (seconds), for the last segment (no next one to measure to).
fn seg_secs(init: &read::Init, samples: &[read::Sample]) -> f64 {
    let Some(v) = init.track(|k| matches!(k, read::TrackKind::Hevc { .. })) else { return 0.0 };
    let ticks: u64 = samples.iter().filter(|s| s.track == v.id).map(|s| u64::from(s.duration)).sum();
    ticks as f64 / f64::from(v.timescale)
}

/// Move each track's `tfdt` back by its shift, in place.
fn retime(frag: &mut [u8], shift: &HashMap<u32, u64>) -> anyhow::Result<()> {
    for (kind, s, e) in boxes(frag, 0, frag.len())? {
        if &kind != b"moof" {
            continue;
        }
        for (k, ts, te) in boxes(frag, s, e)? {
            if &k != b"traf" {
                continue;
            }
            let inner = boxes(frag, ts, te)?;
            let track = inner.iter().find(|b| &b.0 == b"tfhd").map(|&(_, hs, _)| u32::from_be_bytes(frag[hs + 4..hs + 8].try_into().expect("4 bytes")));
            let by = track.and_then(|t| shift.get(&t)).copied().unwrap_or(0);
            if let Some(&(_, ds, _)) = inner.iter().find(|b| &b.0 == b"tfdt")
                && by > 0
            {
                if frag[ds] == 1 {
                    let v = u64::from_be_bytes(frag[ds + 4..ds + 12].try_into().expect("8 bytes"));
                    ensure!(v >= by, "a decode time before the start");
                    frag[ds + 4..ds + 12].copy_from_slice(&(v - by).to_be_bytes());
                } else {
                    let v = u64::from(u32::from_be_bytes(frag[ds + 4..ds + 8].try_into().expect("4 bytes")));
                    ensure!(v >= by, "a decode time before the start");
                    frag[ds + 4..ds + 8].copy_from_slice(&((v - by) as u32).to_be_bytes());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hls::parse;

    #[test]
    fn leaving_out_a_segment_closes_up_the_timeline() {
        // The fixture: three fragments of 1.001, 1.001 and 0.5005 s. Leave out the middle one.
        let dir = std::env::temp_dir().join(format!("fmp4-cut-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media.mp4");
        let pl = parse(include_str!("../tests/fixtures/index.m3u8")).unwrap();
        let cut = drop_segments(&src, &pl, |i| i != 1, &dir).unwrap();
        assert_eq!(cut.removed.len(), 1);
        assert!((cut.removed[0].0 - 1.001).abs() < 1e-6 && (cut.removed[0].1 - 1.001).abs() < 1e-6, "{:?}", cut.removed);
        assert!((cut.duration - 1.5015).abs() < 0.01, "{}", cut.duration);
        // Read back: two fragments, and the second starts where the first ends, on every track.
        let media = std::fs::read(dir.join("media.mp4")).unwrap();
        let out = parse(&std::fs::read_to_string(dir.join("index.m3u8")).unwrap()).unwrap();
        assert_eq!(out.segs.len(), 2);
        let init = read::parse_init(&media[..out.init.1 as usize]).unwrap();
        let frag = |k: usize| read::parse_fragments(&init, &media[out.segs[k].off as usize..(out.segs[k].off + out.segs[k].len) as usize]).unwrap();
        let (a, b) = (frag(0), frag(1));
        for t in &init.tracks {
            let end_a = a.iter().filter(|s| s.track == t.id).map(|s| s.dts + i64::from(s.duration)).max().unwrap();
            let start_b = b.iter().filter(|s| s.track == t.id).map(|s| s.dts).min().unwrap();
            assert!((end_a - start_b).abs() <= 1024, "track {}: {end_a} then {start_b}", t.id);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
