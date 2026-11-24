//! H.264 video elementary stream (Annex B, as broadcast): splitting NAL units, the sequence parameter set's picture
//! size and interlacing, the A/53 captions carried in SEI messages, and conversion to the length-prefixed form
//! decoders that take MP4-style samples want.

/// NAL unit types used here.
pub const SPS: u8 = 7;
pub const PPS: u8 = 8;
const SEI: u8 = 6;
const AUD: u8 = 9;
const FILLER: u8 = 12;

/// The NAL units of an Annex B stream (start codes `00 00 01` / `00 00 00 01` removed).
pub fn nals(es: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let mut end = starts.get(k + 1).map_or(es.len(), |&n| n - 3);
        while end > s && es[end - 1] == 0 {
            end -= 1; // trailing zero of a 4-byte start code, or padding
        }
        if end > s {
            out.push(&es[s..end]);
        }
    }
    out
}

pub fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |b| b & 0x1f)
}

/// The NAL payload with emulation-prevention bytes (`00 00 03`) removed.
fn rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Bits of an RBSP, with Exp-Golomb codes.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.pos / 8)?;
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(u32::from(b))
    }
    fn bits(&mut self, n: u32) -> Option<u32> {
        (0..n).try_fold(0, |acc, _| Some((acc << 1) | self.bit()?))
    }
    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1 << zeros) - 1 + self.bits(zeros)?)
    }
    fn se(&mut self) -> Option<i32> {
        let v = self.ue()?;
        Some(if v % 2 == 1 { v.div_ceil(2) as i32 } else { -((v / 2) as i32) })
    }
}

/// What a sequence parameter set says about the pictures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpsInfo {
    pub width: u32,
    pub height: u32,
    /// Field-coded (not `frame_mbs_only`): an interlaced broadcast.
    pub interlaced: bool,
    /// Bits of `frame_num` in slice headers (needed to read them).
    pub log2_max_frame_num: u32,
}

/// Read a sequence parameter set NAL unit.
pub fn sps_info(nal: &[u8]) -> Option<SpsInfo> {
    let data = rbsp(nal.get(1..)?);
    let mut b = Bits { data: &data, pos: 0 };
    let profile = b.bits(8)?;
    b.bits(16)?; // constraint flags, level
    b.ue()?; // seq_parameter_set_id
    let mut chroma = 1;
    if [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].contains(&profile) {
        chroma = b.ue()?;
        if chroma == 3 {
            b.bit()?;
        }
        b.ue()?; // bit depth luma
        b.ue()?; // bit depth chroma
        b.bit()?; // qpprime_y_zero_transform_bypass
        if b.bit()? == 1 {
            // scaling matrices
            for i in 0..if chroma == 3 { 12 } else { 8 } {
                if b.bit()? == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    let (mut last, mut next) = (8i32, 8i32);
                    for _ in 0..size {
                        if next != 0 {
                            next = (last + b.se()? + 256) % 256;
                        }
                        last = if next == 0 { last } else { next };
                    }
                }
            }
        }
    }
    let log2_max_frame_num = b.ue()? + 4;
    match b.ue()? {
        0 => {
            b.ue()?;
        }
        1 => {
            b.bit()?;
            b.se()?;
            b.se()?;
            for _ in 0..b.ue()? {
                b.se()?;
            }
        }
        _ => {}
    }
    b.ue()?; // max_num_ref_frames
    b.bit()?; // gaps_in_frame_num_allowed
    let width_mbs = b.ue()? + 1;
    let height_units = b.ue()? + 1;
    let frame_mbs_only = b.bit()? == 1;
    if !frame_mbs_only {
        b.bit()?; // mb_adaptive_frame_field
    }
    b.bit()?; // direct_8x8_inference
    let (mut crop_x, mut crop_y) = (0, 0);
    if b.bit()? == 1 {
        let (l, r, t, bo) = (b.ue()?, b.ue()?, b.ue()?, b.ue()?);
        let (unit_x, unit_y) = match chroma {
            0 | 3 => (1, 2 - u32::from(frame_mbs_only)),
            2 => (2, 2 - u32::from(frame_mbs_only)),
            _ => (2, 2 * (2 - u32::from(frame_mbs_only))),
        };
        crop_x = (l + r) * unit_x;
        crop_y = (t + bo) * unit_y;
    }
    let height = (2 - u32::from(frame_mbs_only)) * height_units * 16;
    Some(SpsInfo { width: width_mbs * 16 - crop_x, height: height - crop_y, interlaced: !frame_mbs_only, log2_max_frame_num })
}

/// What an access unit's first slice says about how it is coded: its `frame_num`, and for a field picture which
/// field (`Some(true)` bottom, `Some(false)` top; `None` a whole frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    pub frame_num: u32,
    pub field: Option<bool>,
}

/// Read the first slice header of an access unit.
pub fn first_slice(es: &[u8], sps: &SpsInfo) -> Option<Slice> {
    let nal = nals(es).into_iter().find(|n| matches!(nal_type(n), 1 | 5))?;
    let data = rbsp(nal.get(1..)?);
    let mut b = Bits { data: &data, pos: 0 };
    b.ue()?; // first_mb_in_slice
    b.ue()?; // slice_type
    b.ue()?; // pic_parameter_set_id
    let frame_num = b.bits(sps.log2_max_frame_num)?;
    let field = if sps.interlaced && b.bit()? == 1 { Some(b.bit()? == 1) } else { None };
    Some(Slice { frame_num, field })
}

/// The access unit as length-prefixed NAL units (4-byte big-endian lengths), leaving out the parameter sets, access
/// unit delimiters and filler (the decoder has the parameter sets from its format description).
pub fn to_length_prefixed(es: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(es.len() + 16);
    for nal in nals(es) {
        if matches!(nal_type(nal), SPS | PPS | AUD | FILLER) {
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

/// The first SPS and PPS in an access unit, if it has them (broadcasts repeat them before each keyframe).
pub fn parameter_sets(es: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let all = nals(es);
    let sps = all.iter().find(|n| nal_type(n) == SPS)?;
    let pps = all.iter().find(|n| nal_type(n) == PPS)?;
    Some((sps.to_vec(), pps.to_vec()))
}

/// The A/53 caption triplets in an access unit's SEI messages (registered ITU-T T.35 user data: USA, provider
/// 0x0031, "GA94", type 3).
pub fn captions(es: &[u8]) -> Vec<(u8, u8, u8)> {
    let mut out = Vec::new();
    for nal in nals(es) {
        if nal_type(nal) != SEI {
            continue;
        }
        let p = rbsp(&nal[1..]);
        let mut i = 0;
        while i < p.len() && p[i] != 0x80 {
            let (mut kind, mut size) = (0usize, 0usize);
            while i < p.len() && p[i] == 0xff {
                kind += 255;
                i += 1;
            }
            kind += usize::from(*p.get(i).unwrap_or(&0));
            i += 1;
            while i < p.len() && p[i] == 0xff {
                size += 255;
                i += 1;
            }
            size += usize::from(*p.get(i).unwrap_or(&0));
            i += 1;
            let Some(body) = p.get(i..i + size) else { break };
            if kind == 4 && body.len() > 10 && body[0] == 0xb5 && body[1..3] == [0x00, 0x31] && &body[3..7] == b"GA94" && body[7] == 3 && body[8] & 0x40 != 0 {
                let count = usize::from(body[8] & 0x1f);
                for t in body[10..].as_chunks::<3>().0.iter().take(count) {
                    if t[0] & 0x04 != 0 {
                        out.push((t[0] & 0x03, t[1], t[2]));
                    }
                }
            }
            i += size;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_annex_b_and_converts_to_length_prefixed() {
        let es = [0, 0, 0, 1, 0x09, 0xf0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 9, 9, 9];
        let n = nals(&es);
        assert_eq!(n.iter().map(|x| nal_type(x)).collect::<Vec<_>>(), [AUD, SPS, PPS, 5]);
        assert_eq!(to_length_prefixed(&es), [0, 0, 0, 4, 0x65, 9, 9, 9]);
    }

    /// The video access units of a fixture stream.
    fn video(ts: &[u8]) -> Vec<Vec<u8>> {
        let mut d = crate::Demuxer::new();
        let mut units = d.push(ts);
        units.extend(d.flush());
        units.into_iter().filter_map(|u| if let crate::Unit::Video { data, .. } = u { Some(data) } else { None }).collect()
    }

    #[test]
    fn reads_the_size_and_scan_of_broadcast_parameter_sets() {
        // 720p progressive and 480i (field-coded, top field first), made by x264 from real captures.
        let progressive = video(include_bytes!("../tests/fixtures/h264-720p.ts"));
        let (sps, _) = progressive.iter().find_map(|au| parameter_sets(au)).expect("parameter sets before a keyframe");
        assert_eq!(sps_info(&sps), Some(SpsInfo { width: 1280, height: 720, interlaced: false, log2_max_frame_num: 4 }));
        let interlaced = video(include_bytes!("../tests/fixtures/h264-480i-eac3.ts"));
        let (sps, _) = interlaced.iter().find_map(|au| parameter_sets(au)).unwrap();
        assert_eq!(sps_info(&sps), Some(SpsInfo { width: 704, height: 480, interlaced: true, log2_max_frame_num: 4 }));
    }

    #[test]
    fn finds_a53_captions_in_sei() {
        let aus = video(include_bytes!("../tests/fixtures/h264-480i-eac3.ts"));
        let triplets: usize = aus.iter().map(|au| captions(au).len()).sum();
        assert!(triplets > 50, "{triplets}");
    }
}
