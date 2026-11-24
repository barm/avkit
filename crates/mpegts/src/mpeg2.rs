//! MPEG-2 video elementary stream headers: picture size, interlacing, and the A/53 captions a picture carries.

/// From an MPEG-2 picture's headers: whether the stream is interlaced (the sequence extension's
/// `progressive_sequence` is 0) and whether its top field comes first (the picture coding extension). `None`
/// where the headers aren't there.
pub fn scan(es: &[u8]) -> (Option<bool>, Option<bool>) {
    let (mut interlaced, mut tff) = (None, None);
    let mut i = 0;
    while i + 9 <= es.len() {
        if es[i..i + 4] == [0, 0, 1, 0xb5] {
            match es[i + 4] >> 4 {
                1 => interlaced = Some((es[i + 5] >> 3) & 1 == 0),
                8 => tff = Some(es[i + 7] >> 7 == 1),
                _ => {}
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    (interlaced, tff)
}

/// The picture size in an MPEG-2 sequence header (`00 00 01 B3`), if `es` has one.
pub fn size(es: &[u8]) -> Option<(u32, u32)> {
    let at = es.windows(4).position(|w| w == [0, 0, 1, 0xb3])?;
    let h = es.get(at + 4..at + 7)?;
    Some(((u32::from(h[0]) << 4) | (u32::from(h[1]) >> 4), (u32::from(h[1] & 0x0f) << 8) | u32::from(h[2])))
}

/// The A/53 caption triplets in an MPEG-2 picture's user data (`00 00 01 B2` "GA94" 0x03 cc_data).
pub fn captions(es: &[u8]) -> Vec<(u8, u8, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 < es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 && es[i + 3] == 0xb2 {
            let ud = &es[i + 4..];
            if ud.len() > 7 && &ud[..4] == b"GA94" && ud[4] == 0x03 && ud[5] & 0x40 != 0 {
                let count = usize::from(ud[5] & 0x1f);
                for t in ud[7..].as_chunks::<3>().0.iter().take(count) {
                    if t[0] & 0x04 != 0 {
                        out.push((t[0] & 0x03, t[1], t[2])); // valid
                    }
                }
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    out
}
