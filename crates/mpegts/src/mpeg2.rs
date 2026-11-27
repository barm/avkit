//! MPEG-2 video elementary stream headers: picture size, shape and frame rate, interlacing, and the A/53 captions a
//! picture carries.

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

/// What an MPEG-2 sequence header says about the pictures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sequence {
    pub width: u32,
    pub height: u32,
    /// The picture's shape on screen (display aspect ratio, reduced: `(16, 9)`), if `aspect_ratio_information` is
    /// one of the defined codes.
    pub display_aspect: Option<(u32, u32)>,
    /// Frames a second as a fraction (`30000/1001`), if `frame_rate_code` is one of the defined codes.
    pub frame_rate: Option<(u32, u32)>,
}

/// The sequence header (`00 00 01 B3`) in `es`, if it has one.
pub fn sequence(es: &[u8]) -> Option<Sequence> {
    let at = es.windows(4).position(|w| w == [0, 0, 1, 0xb3])?;
    let h = es.get(at + 4..at + 8)?;
    let width = (u32::from(h[0]) << 4) | (u32::from(h[1]) >> 4);
    let height = (u32::from(h[1] & 0x0f) << 8) | u32::from(h[2]);
    // ISO/IEC 13818-2 Table 6-3: 1 is square pixels; the others give the display shape itself.
    let display_aspect = match h[3] >> 4 {
        1 if width > 0 && height > 0 => Some(crate::reduce(u64::from(width), u64::from(height))),
        2 => Some((4, 3)),
        3 => Some((16, 9)),
        4 => Some((221, 100)),
        _ => None,
    };
    // Table 6-4.
    let frame_rate = match h[3] & 0x0f {
        1 => Some((24000, 1001)),
        2 => Some((24, 1)),
        3 => Some((25, 1)),
        4 => Some((30000, 1001)),
        5 => Some((30, 1)),
        6 => Some((50, 1)),
        7 => Some((60000, 1001)),
        8 => Some((60, 1)),
        _ => None,
    };
    Some(Sequence { width, height, display_aspect, frame_rate })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_sequence_header() {
        // 1920x1080, 16:9, 29.97 (after a picture's tail, as a PES would carry it).
        let hd = [0xff, 0, 0, 1, 0xb3, 0x78, 0x04, 0x38, 0x34, 0xff];
        assert_eq!(sequence(&hd), Some(Sequence { width: 1920, height: 1080, display_aspect: Some((16, 9)), frame_rate: Some((30000, 1001)) }));
        assert_eq!(size(&hd), Some((1920, 1080)));
        // 704x480 shown 4:3.
        let sd = [0, 0, 1, 0xb3, 0x2c, 0x01, 0xe0, 0x24];
        assert_eq!(sequence(&sd).map(|s| (s.width, s.height, s.display_aspect)), Some((704, 480, Some((4, 3)))));
        // Square pixels: the shape is the size's; an unknown code says nothing.
        let square = [0, 0, 1, 0xb3, 0x28, 0x01, 0xe0, 0x1f];
        assert_eq!(sequence(&square), Some(Sequence { width: 640, height: 480, display_aspect: Some((4, 3)), frame_rate: None }));
        assert_eq!(sequence(&[0, 0, 1, 0xb3, 0x78, 0x04]), None, "cut short");
        assert_eq!(sequence(&[0, 0, 1, 0x00, 1, 2, 3, 4]), None, "no sequence header");
    }
}
