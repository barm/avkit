//! HLS playlists over one fragmented MP4 by byte range: writing them as segments are appended (a recording's
//! EVENT playlist, or a sliding live window), and reading them back.

use crate::write::{Run, fragment};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Where a segment's data is: a byte range of the single media file, or a file of its own (`seg%05d.m4s`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Loc {
    Range { off: u64, len: u64 },
    File(u32),
}

/// How the output is laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    /// One growing `media.mp4` addressed by byte range, every segment listed (a recording).
    SingleFile,
    /// `init.mp4` and a file per segment (`seg00000.m4s`, ...); the playlist lists the last `segments`, and files
    /// are deleted once `keep` more have passed out of it (a live window, like ffmpeg's `hls_list_size` and
    /// `hls_delete_threshold`), so a player still fetching one never finds it gone.
    Window { segments: usize, keep: usize },
}

/// The playlist over a writer's output.
pub struct Playlist {
    /// The init segment's length (at offset 0 of the single file).
    init_len: u64,
    /// (duration in seconds, where) per segment, from the first written.
    segments: Vec<(f64, Loc)>,
    target: u32,
    closed: bool,
    /// List only the last this many (the media sequence counts the ones dropped from the list).
    window: Option<usize>,
}

impl Playlist {
    fn new(init_len: u64, target_secs: u32, window: Option<usize>) -> Self {
        Self { init_len, segments: Vec::new(), target: target_secs, closed: false, window }
    }

    fn push(&mut self, secs: f64, loc: Loc) {
        self.target = self.target.max(secs.ceil() as u32);
        self.segments.push((secs, loc));
    }

    pub fn duration(&self) -> f64 {
        self.segments.iter().map(|s| s.0).sum()
    }

    /// The playlist text, in the form ffmpeg wrote: a recording's EVENT playlist (closed when done), or a live
    /// window over segment files.
    fn render(&self) -> String {
        let skip = self.window.map_or(0, |w| self.segments.len().saturating_sub(w));
        let kind = if self.window.is_some() { "" } else { "#EXT-X-PLAYLIST-TYPE:EVENT\n" };
        let map = if self.window.is_some() { "URI=\"init.mp4\"".to_string() } else { format!("URI=\"media.mp4\",BYTERANGE=\"{}@0\"", self.init_len) };
        let mut s = format!("#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:{}\n#EXT-X-MEDIA-SEQUENCE:{skip}\n{kind}#EXT-X-INDEPENDENT-SEGMENTS\n#EXT-X-MAP:{map}\n", self.target);
        for (secs, loc) in &self.segments[skip..] {
            match loc {
                Loc::Range { off, len } => s.push_str(&format!("#EXTINF:{secs:.6},\n#EXT-X-BYTERANGE:{len}@{off}\nmedia.mp4\n")),
                Loc::File(n) => s.push_str(&format!("#EXTINF:{secs:.6},\n{}\n", segment_name(*n))),
            }
        }
        if self.closed {
            s.push_str("#EXT-X-ENDLIST\n");
        }
        s
    }
}

fn segment_name(n: u32) -> String {
    format!("seg{n:05}.m4s")
}

/// HLS output in `dir`, laid out as `layout` says; `index.m3u8` is replaced whole on every segment, so a reader never
/// sees half of it.
pub struct HlsWriter {
    dir: PathBuf,
    layout: Layout,
    /// The single file (`Layout::SingleFile`).
    media: Option<std::fs::File>,
    written: u64,
    playlist: Playlist,
    sequence: u32,
}

impl HlsWriter {
    pub fn create(dir: &Path, init: &[u8], target_secs: u32, layout: Layout) -> std::io::Result<Self> {
        let (media, window) = match layout {
            Layout::SingleFile => {
                let mut f = std::fs::File::create(dir.join("media.mp4"))?;
                f.write_all(init)?;
                (Some(f), None)
            }
            Layout::Window { segments, .. } => {
                write_whole(&dir.join("init.mp4"), init)?;
                (None, Some(segments))
            }
        };
        let w = Self { dir: dir.to_path_buf(), layout, media, written: init.len() as u64, playlist: Playlist::new(init.len() as u64, target_secs, window), sequence: 0 };
        w.write_playlist()?;
        Ok(w)
    }

    /// Add a segment lasting `secs`.
    pub fn segment(&mut self, secs: f64, video: &Run, audio: Option<&Run>) -> std::io::Result<()> {
        let index = self.sequence;
        self.sequence += 1;
        let frag = fragment(self.sequence, video, audio);
        match (&mut self.media, self.layout) {
            (Some(media), _) => {
                media.write_all(&frag)?;
                media.flush()?;
                self.playlist.push(secs, Loc::Range { off: self.written, len: frag.len() as u64 });
                self.written += frag.len() as u64;
            }
            (None, Layout::Window { segments, keep }) => {
                write_whole(&self.dir.join(segment_name(index)), &frag)?;
                self.playlist.push(secs, Loc::File(index));
                if let Some(old) = index.checked_sub((segments + keep) as u32) {
                    let _ = std::fs::remove_file(self.dir.join(segment_name(old)));
                }
            }
            (None, Layout::SingleFile) => unreachable!("a single file is always open"),
        }
        self.write_playlist()
    }

    /// Append a fragment already in its final form (`moof` + `mdat`, as `write::fragment` makes) to the single file.
    pub fn append_fragment(&mut self, secs: f64, frag: &[u8]) -> std::io::Result<()> {
        let media = self.media.as_mut().ok_or_else(|| std::io::Error::other("only a single-file layout takes raw fragments"))?;
        media.write_all(frag)?;
        self.playlist.push(secs, Loc::Range { off: self.written, len: frag.len() as u64 });
        self.written += frag.len() as u64;
        self.write_playlist()
    }

    /// Seconds written so far.
    pub fn duration(&self) -> f64 {
        self.playlist.duration()
    }

    /// Close the playlist; returns the duration.
    pub fn finish(mut self) -> std::io::Result<f64> {
        if let Some(m) = &self.media {
            m.sync_all()?;
        }
        self.playlist.closed = true;
        self.write_playlist()?;
        Ok(self.playlist.duration())
    }

    fn write_playlist(&self) -> std::io::Result<()> {
        write_whole(&self.dir.join("index.m3u8"), self.playlist.render().as_bytes())
    }
}

/// Write a file under a temporary name and rename it into place: readers see all of it or none.
fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_file_name(format!(".{}.tmp", path.file_name().and_then(|n| n.to_str()).unwrap_or("out")));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::write::Sample;

    #[test]
    fn a_live_window_lists_the_last_segments_and_deletes_old_files() {
        let dir = std::env::temp_dir().join(format!("fmp4-window-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = HlsWriter::create(&dir, b"init", 2, Layout::Window { segments: 3, keep: 2 }).unwrap();
        let samples = [Sample { data: vec![0; 10], duration: 3000, cts_offset: 0, keyframe: true }];
        for i in 0..15u64 {
            w.segment(2.0, &Run { dts: i * 180_000, samples: &samples }, None).unwrap();
        }
        let pl = std::fs::read_to_string(dir.join("index.m3u8")).unwrap();
        assert!(pl.contains("#EXT-X-MEDIA-SEQUENCE:12\n") && pl.contains("#EXT-X-MAP:URI=\"init.mp4\"\n"), "{pl}");
        assert_eq!(pl.matches("#EXTINF").count(), 3);
        assert!(pl.contains("seg00012.m4s") && pl.contains("seg00014.m4s") && !pl.contains("PLAYLIST-TYPE"));
        // Listed (12–14) plus the grace of two (10, 11); older ones gone.
        let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".m4s")).collect();
        files.sort();
        assert_eq!(files, ["seg00010.m4s", "seg00011.m4s", "seg00012.m4s", "seg00013.m4s", "seg00014.m4s"]);
        assert!(dir.join("init.mp4").exists() && !dir.join("media.mp4").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// One fragment of the media file, from the playlist.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seg {
    /// Seconds from the start of the recording.
    pub start: f64,
    pub off: u64,
    pub len: u64,
}

/// A single-file fMP4 playlist: the init segment's byte range and the fragments.
#[derive(Debug, PartialEq)]
pub struct ByteRanges {
    pub init: (u64, u64),
    pub segs: Vec<Seg>,
}

/// Parse a playlist that addresses one file by byte range (as the recorder writes them). `None` for anything else.
pub fn parse(text: &str) -> Option<ByteRanges> {
    let mut init = None;
    let mut segs = Vec::new();
    let (mut t, mut dur) = (0.0, None);
    for line in text.lines() {
        if let Some(m) = line.strip_prefix("#EXT-X-MAP:") {
            let r = m.split("BYTERANGE=\"").nth(1)?.split('"').next()?;
            init = Some(range(r)?);
        } else if let Some(d) = line.strip_prefix("#EXTINF:") {
            dur = d.split(',').next()?.trim().parse::<f64>().ok();
        } else if let Some(r) = line.strip_prefix("#EXT-X-BYTERANGE:") {
            let (off, len) = range(r)?;
            segs.push(Seg { start: t, off, len });
            t += dur.take()?;
        }
    }
    Some(ByteRanges { init: init?, segs })
}

/// `len@off` as (offset, length).
fn range(r: &str) -> Option<(u64, u64)> {
    let (len, off) = r.split_once('@')?;
    Some((off.trim().parse().ok()?, len.trim().parse().ok()?))
}

