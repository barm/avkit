//! Reading a fragmented MP4 file as a stream, never all in memory: the init segment, then the fragments in
//! batches of whole `moof` + `mdat` pairs.

use anyhow::{Context, bail};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// The file's fragments, about `batch_bytes` at a time.
pub struct Fragments<R> {
    r: R,
    batch_bytes: usize,
}

/// Opens a fragmented MP4: its init segment (everything up to and including `moov`) and its fragments.
pub fn open(path: &Path, batch_bytes: usize) -> anyhow::Result<(Vec<u8>, Fragments<BufReader<File>>)> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut init = Vec::new();
    loop {
        let Some((kind, bytes)) = next_box(&mut r)? else { bail!("{}: no moov box", path.display()) };
        init.extend_from_slice(&bytes);
        if &kind == b"moov" {
            break;
        }
    }
    Ok((init, Fragments { r, batch_bytes }))
}

impl<R: Read> Iterator for Fragments<R> {
    type Item = anyhow::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::new();
        loop {
            match next_box(&mut self.r) {
                Ok(Some((kind, bytes))) => {
                    batch.extend_from_slice(&bytes);
                    // A batch ends after an `mdat`, so every `moof` is with its data.
                    if &kind == b"mdat" && batch.len() >= self.batch_bytes {
                        return Some(Ok(batch));
                    }
                }
                Ok(None) => return (!batch.is_empty()).then_some(Ok(batch)),
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// The next top-level box, header included, or `None` at the end of the file (or at a box cut off by it, as in a
/// recording still being written).
pub fn next_box(r: &mut impl Read) -> anyhow::Result<Option<([u8; 4], Vec<u8>)>> {
    let mut head = [0u8; 8];
    match r.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let kind: [u8; 4] = head[4..8].try_into().unwrap();
    let mut buf = head.to_vec();
    let size = match u32::from_be_bytes(head[..4].try_into().unwrap()) {
        1 => {
            let mut large = [0u8; 8];
            r.read_exact(&mut large)?;
            buf.extend_from_slice(&large);
            u64::from_be_bytes(large)
        }
        0 => bail!("a box running to the end of the file isn't supported"),
        n => u64::from(n),
    };
    let rest = size.checked_sub(buf.len() as u64).context("box shorter than its header")?;
    let start = buf.len();
    buf.resize(start + usize::try_from(rest)?, 0);
    match r.read_exact(&mut buf[start..]) {
        Ok(()) => Ok(Some((kind, buf))),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_then_fragments_in_batches() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media.mp4");
        let (init, frags) = open(&path, 1).unwrap();
        assert_eq!(init.len(), 1433);
        let batches: Vec<Vec<u8>> = frags.map(Result::unwrap).collect();
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![57712, 31272, 1488]);
        let (_, frags) = open(&path, 1 << 20).unwrap();
        assert_eq!(frags.map(|b| b.unwrap().len()).sum::<usize>(), 57712 + 31272 + 1488);
    }
}
