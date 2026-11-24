//! Writing what `read` reads: a fragmented MP4 in one file (an init segment, then a `moof` + `mdat` per segment)
//! and the HLS playlist that addresses it by byte range, the form every recording takes. Pure Rust; the native
//! pipeline (#119) hands it encoded samples.

/// The video track: HEVC, its `hvcC` box body as the encoder describes it.
#[derive(Clone, Debug)]
pub struct VideoTrack {
    pub width: u16,
    pub height: u16,
    pub timescale: u32,
    pub hvcc: Vec<u8>,
}

/// The audio track: AAC, its ES descriptor (the `esds` body after version and flags).
#[derive(Clone, Debug)]
pub struct AudioTrack {
    pub channels: u16,
    pub sample_rate: u32,
    pub esds: Vec<u8>,
}

/// One coded sample for a fragment. Times are in the track's timescale.
#[derive(Clone, Debug)]
pub struct Sample {
    pub data: Vec<u8>,
    pub duration: u32,
    /// Presentation minus decode time.
    pub cts_offset: i32,
    pub keyframe: bool,
}

const VIDEO_ID: u32 = 1;
const AUDIO_ID: u32 = 2;

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + body.len());
    b.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
    b.extend_from_slice(kind);
    b.extend_from_slice(body);
    b
}

fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version];
    b.extend_from_slice(&flags.to_be_bytes()[1..]);
    b.extend_from_slice(body);
    bx(kind, &b)
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn matrix() -> Vec<u8> {
    MATRIX.iter().flat_map(|m| m.to_be_bytes()).collect()
}

/// The init segment (`ftyp` + `moov`) for these tracks.
pub fn init_segment(video: &VideoTrack, audio: Option<&AudioTrack>) -> Vec<u8> {
    let ftyp = bx(b"ftyp", &cat(&[b"iso5", &512u32.to_be_bytes(), b"iso5", b"iso6", b"mp41", b"hvc1"]));
    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&[0; 8]); // creation, modification
    mvhd.extend_from_slice(&1000u32.to_be_bytes());
    mvhd.extend_from_slice(&0u32.to_be_bytes()); // duration: fragmented
    mvhd.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
    mvhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
    mvhd.extend_from_slice(&[0; 10]);
    mvhd.extend_from_slice(&matrix());
    mvhd.extend_from_slice(&[0; 24]);
    mvhd.extend_from_slice(&(if audio.is_some() { 3u32 } else { 2 }).to_be_bytes()); // next track id
    let mut moov = full(b"mvhd", 0, 0, &mvhd);
    moov.extend(video_trak(video));
    if let Some(a) = audio {
        moov.extend(audio_trak(a));
    }
    let trex = |id: u32| full(b"trex", 0, 0, &cat(&[&id.to_be_bytes(), &1u32.to_be_bytes(), &[0; 12]]));
    let mut mvex = trex(VIDEO_ID);
    if audio.is_some() {
        mvex.extend(trex(AUDIO_ID));
    }
    moov.extend(bx(b"mvex", &mvex));
    cat(&[&ftyp, &bx(b"moov", &moov)])
}

fn tkhd(id: u32, audio: bool, width: u16, height: u16) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&id.to_be_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&0u32.to_be_bytes()); // duration
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&0u16.to_be_bytes()); // layer
    b.extend_from_slice(&(u16::from(audio)).to_be_bytes()); // alternate group
    b.extend_from_slice(&(if audio { 0x0100u16 } else { 0 }).to_be_bytes()); // volume
    b.extend_from_slice(&[0; 2]);
    b.extend_from_slice(&matrix());
    b.extend_from_slice(&(u32::from(width) << 16).to_be_bytes());
    b.extend_from_slice(&(u32::from(height) << 16).to_be_bytes());
    full(b"tkhd", 0, 3, &b) // enabled, in movie
}

fn mdhd(timescale: u32) -> Vec<u8> {
    full(b"mdhd", 0, 0, &cat(&[&[0; 8], &timescale.to_be_bytes(), &0u32.to_be_bytes(), &0x55c4u16.to_be_bytes(), &[0; 2]]))
}

fn hdlr(kind: &[u8; 4], name: &str) -> Vec<u8> {
    full(b"hdlr", 0, 0, &cat(&[&[0; 4], kind, &[0; 12], name.as_bytes(), &[0]]))
}

fn dinf() -> Vec<u8> {
    bx(b"dinf", &full(b"dref", 0, 0, &cat(&[&1u32.to_be_bytes(), &full(b"url ", 0, 1, &[])])))
}

/// An empty sample table around one sample entry (the samples are all in fragments).
fn stbl(entry: Vec<u8>) -> Vec<u8> {
    let stsd = full(b"stsd", 0, 0, &cat(&[&1u32.to_be_bytes(), &entry]));
    let empty = |k: &[u8; 4]| full(k, 0, 0, &0u32.to_be_bytes());
    let stsz = full(b"stsz", 0, 0, &[0; 8]);
    bx(b"stbl", &cat(&[&stsd, &empty(b"stts"), &empty(b"stsc"), &stsz, &empty(b"stco")]))
}

fn video_trak(v: &VideoTrack) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    entry.extend_from_slice(&[0; 16]);
    entry.extend_from_slice(&v.width.to_be_bytes());
    entry.extend_from_slice(&v.height.to_be_bytes());
    entry.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // 72 dpi
    entry.extend_from_slice(&0x0048_0000u32.to_be_bytes());
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(&1u16.to_be_bytes()); // frames per sample
    entry.extend_from_slice(&[0; 32]); // compressor name
    entry.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    entry.extend_from_slice(&0xffffu16.to_be_bytes());
    entry.extend(bx(b"hvcC", &v.hvcc));
    let minf = bx(b"minf", &cat(&[&full(b"vmhd", 0, 1, &[0; 8]), &dinf(), &stbl(bx(b"hvc1", &entry))]));
    let mdia = bx(b"mdia", &cat(&[&mdhd(v.timescale), &hdlr(b"vide", "VideoHandler"), &minf]));
    bx(b"trak", &cat(&[&tkhd(VIDEO_ID, false, v.width, v.height), &mdia]))
}

fn audio_trak(a: &AudioTrack) -> Vec<u8> {
    let mut entry = Vec::new();
    entry.extend_from_slice(&[0; 6]);
    entry.extend_from_slice(&1u16.to_be_bytes());
    entry.extend_from_slice(&[0; 8]);
    entry.extend_from_slice(&a.channels.to_be_bytes());
    entry.extend_from_slice(&16u16.to_be_bytes()); // sample size
    entry.extend_from_slice(&[0; 4]);
    entry.extend_from_slice(&(a.sample_rate << 16).to_be_bytes());
    entry.extend(full(b"esds", 0, 0, &a.esds));
    let minf = bx(b"minf", &cat(&[&full(b"smhd", 0, 0, &[0; 4]), &dinf(), &stbl(bx(b"mp4a", &entry))]));
    let mdia = bx(b"mdia", &cat(&[&mdhd(a.sample_rate), &hdlr(b"soun", "SoundHandler"), &minf]));
    bx(b"trak", &cat(&[&tkhd(AUDIO_ID, true, 0, 0), &mdia]))
}

/// One track's samples in a fragment, starting at `dts` (its timescale).
pub struct Run<'a> {
    pub dts: u64,
    pub samples: &'a [Sample],
}

/// A fragment: `moof` + `mdat` with the video run and (if any) the audio run.
pub fn fragment(sequence: u32, video: &Run, audio: Option<&Run>) -> Vec<u8> {
    let runs: Vec<(u32, &Run, bool)> = std::iter::once((VIDEO_ID, video, true)).chain(audio.map(|a| (AUDIO_ID, a, false))).collect();
    // The moof's size doesn't depend on the data offsets' values, so build it once with zeros to measure it.
    let build = |offsets: &[u32]| -> Vec<u8> {
        let mut moof = full(b"mfhd", 0, 0, &sequence.to_be_bytes());
        for (i, (id, run, is_video)) in runs.iter().enumerate() {
            let tfhd = full(b"tfhd", 0, 0x02_0000, &id.to_be_bytes()); // default-base-is-moof
            let tfdt = full(b"tfdt", 1, 0, &run.dts.to_be_bytes());
            // data offset, duration, size, and for video flags and composition offsets
            let flags = if *is_video { 0x1 | 0x100 | 0x200 | 0x400 | 0x800 } else { 0x1 | 0x100 | 0x200 };
            let mut t = Vec::new();
            t.extend_from_slice(&(run.samples.len() as u32).to_be_bytes());
            t.extend_from_slice(&offsets[i].to_be_bytes());
            for s in run.samples {
                t.extend_from_slice(&s.duration.to_be_bytes());
                t.extend_from_slice(&(s.data.len() as u32).to_be_bytes());
                if *is_video {
                    let f: u32 = if s.keyframe { 0x0200_0000 } else { 0x0101_0000 };
                    t.extend_from_slice(&f.to_be_bytes());
                    t.extend_from_slice(&s.cts_offset.to_be_bytes());
                }
            }
            moof.extend(bx(b"traf", &cat(&[&tfhd, &tfdt, &full(b"trun", 1, flags, &t)])));
        }
        bx(b"moof", &moof)
    };
    let moof_len = build(&vec![0; runs.len()]).len() as u32;
    let mut offsets = Vec::new();
    let mut at = moof_len + 8; // past the mdat header
    for (_, run, _) in &runs {
        offsets.push(at);
        at += run.samples.iter().map(|s| s.data.len() as u32).sum::<u32>();
    }
    let mut out = build(&offsets);
    let data: Vec<u8> = runs.iter().flat_map(|(_, run, _)| run.samples.iter().flat_map(|s| s.data.iter().copied())).collect();
    out.extend(bx(b"mdat", &data));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hls::{HlsWriter, parse};
    use crate::read::{self as mp4, TrackKind};
    use std::path::Path;

    const MEDIA: &[u8] = include_bytes!("../tests/fixtures/media.mp4");
    const FRAGS: [(usize, usize, f64); 3] = [(1433, 57712, 1.001), (59145, 31272, 1.001), (90417, 1488, 0.5005)];

    /// Read the fixture's samples back out of ffmpeg's file and write them again with this writer.
    fn remux(dir: &Path) -> (mp4::Init, Vec<Vec<mp4::Sample>>) {
        let init = mp4::parse_init(&MEDIA[..1433]).unwrap();
        let v = init.track(|k| matches!(k, TrackKind::Hevc { .. })).unwrap().clone();
        let a = init.track(|k| matches!(k, TrackKind::Aac { .. })).unwrap().clone();
        let TrackKind::Hevc { width, height, hvcc, .. } = v.kind.clone() else { unreachable!() };
        let TrackKind::Aac { channels, sample_rate, esds } = a.kind.clone() else { unreachable!() };
        let mut w = HlsWriter::create(
            dir,
            &init_segment(&VideoTrack { width, height, timescale: v.timescale, hvcc }, Some(&AudioTrack { channels, sample_rate, esds })),
            1,
            crate::hls::Layout::SingleFile,
        )
        .unwrap();
        let mut per_frag = Vec::new();
        for (off, len, secs) in FRAGS {
            let bytes = &MEDIA[off..off + len];
            let samples = mp4::parse_fragments(&init, bytes).unwrap();
            let pick = |id: u32| -> Vec<Sample> {
                let mine: Vec<&mp4::Sample> = samples.iter().filter(|s| s.track == id).collect();
                mine.iter()
                    .enumerate()
                    .map(|(i, s)| Sample {
                        data: bytes[s.offset..s.offset + s.size].to_vec(),
                        duration: mine.get(i + 1).map_or(if id == v.id { 1001 } else { 1024 }, |n| (n.dts - s.dts) as u32),
                        cts_offset: (s.pts - s.dts) as i32,
                        keyframe: i == 0,
                    })
                    .collect()
            };
            let (vs, as_) = (pick(v.id), pick(a.id));
            let vdts = samples.iter().find(|s| s.track == v.id).unwrap().dts as u64;
            let adts = samples.iter().find(|s| s.track == a.id).map_or(0, |s| s.dts as u64);
            w.segment(secs, &Run { dts: vdts, samples: &vs }, Some(&Run { dts: adts, samples: &as_ })).unwrap();
            per_frag.push(samples);
        }
        w.finish().unwrap();
        (init, per_frag)
    }

    #[test]
    fn a_remuxed_recording_reads_back_the_same() {
        let dir = std::env::temp_dir().join(format!("hdhr-dvr-fmp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (_, original) = remux(&dir);
        let out = std::fs::read(dir.join("media.mp4")).unwrap();
        let pl = std::fs::read_to_string(dir.join("index.m3u8")).unwrap();
        let parsed = parse(&pl).expect("a byte-range playlist the scanner reads");
        assert_eq!(parsed.segs.len(), 3);
        assert!(pl.ends_with("#EXT-X-ENDLIST\n"));
        let init = mp4::parse_init(&out[..parsed.init.1 as usize]).unwrap();
        assert_eq!(init.tracks.len(), 2);
        for (seg, orig) in parsed.segs.iter().zip(&original) {
            let bytes = &out[seg.off as usize..(seg.off + seg.len) as usize];
            let again = mp4::parse_fragments(&init, bytes).unwrap();
            assert_eq!(again.len(), orig.len());
            for (x, y) in again.iter().zip(orig) {
                assert_eq!((x.dts, x.pts, x.size), (y.dts, y.pts, y.size));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
