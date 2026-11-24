//! Broadcast captions (CEA-608, channel CC1, as carried in ATSC A/53 `cc_data`) to WebVTT cues. Pop-on captions
//! become a cue when they are shown; roll-up and paint-on text becomes a cue as each line completes.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::io::Write;

/// A caption shown from `start` to `end` (seconds on the recording's timeline).
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    PopOn,
    RollUp(usize),
    PaintOn,
}

/// The 608 decoder for CC1.
pub struct Decoder {
    mode: Mode,
    /// The pop-on buffer being loaded, as rows of text.
    loading: Vec<String>,
    /// What is on screen.
    shown: Vec<String>,
    shown_since: Option<f64>,
    /// The last control pair, to skip its transmitted repeat.
    last_control: Option<(u8, u8)>,
    /// Whether the data seen is for channel 1 (control codes switch channels).
    cc1: bool,
    cues: Vec<Cue>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self { mode: Mode::PopOn, loading: vec![String::new()], shown: Vec::new(), shown_since: None, last_control: None, cc1: true, cues: Vec::new() }
    }
}

/// The 608 basic character set where it differs from ASCII.
fn basic(c: u8) -> char {
    match c {
        0x2a => 'á',
        0x5c => 'é',
        0x5e => 'í',
        0x5f => 'ó',
        0x60 => 'ú',
        0x7b => 'ç',
        0x7c => '÷',
        0x7d => 'Ñ',
        0x7e => 'ñ',
        0x7f => '█',
        c => char::from(c),
    }
}

/// Special characters (`0x11 0x30..=0x3f`).
const SPECIAL: [char; 16] = ['®', '°', '½', '¿', '™', '¢', '£', '♪', 'à', ' ', 'è', 'â', 'ê', 'î', 'ô', 'û'];
/// Extended characters (`0x12`/`0x13 0x20..=0x3f`); each replaces the character before it.
const EXTENDED_12: [char; 32] = [
    'Á', 'É', 'Ó', 'Ú', 'Ü', 'ü', '‘', '¡', '*', '\'', '—', '©', '℠', '•', '“', '”', 'À', 'Â', 'Ç', 'È', 'Ê', 'Ë', 'ë', 'Î', 'Ï', 'ï', 'Ô', 'Ù', 'ù', 'Û', '«',
    '»',
];
const EXTENDED_13: [char; 32] = [
    'Ã', 'ã', 'Í', 'Ì', 'ì', 'Ò', 'ò', 'Õ', 'õ', '{', '}', '\\', '^', '_', '|', '~', 'Ä', 'ä', 'Ö', 'ö', 'ß', '¥', '¤', '¦', 'Å', 'å', 'Ø', 'ø', '┌', '┐', '└',
    '┘',
];

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The rows text is written to now.
    fn target(&mut self) -> &mut Vec<String> {
        if self.mode == Mode::PopOn { &mut self.loading } else { &mut self.shown }
    }

    fn put(&mut self, c: char) {
        let rows = self.target();
        if rows.is_empty() {
            rows.push(String::new());
        }
        rows.last_mut().expect("non-empty").push(c);
    }

    /// End what is on screen at `t` (a cue if it had text).
    fn end_shown(&mut self, t: f64) {
        let text = self.shown.iter().map(|r| r.trim()).filter(|r| !r.is_empty()).collect::<Vec<_>>().join("\n");
        if let Some(start) = self.shown_since.take()
            && !text.is_empty()
            && t > start
        {
            self.cues.push(Cue { start, end: t, text });
        }
    }

    /// A new line for roll-up and paint-on: the finished text on screen becomes a cue, and the rows scroll.
    fn carriage_return(&mut self, t: f64) {
        match self.mode {
            Mode::RollUp(n) => {
                self.end_shown(t);
                self.shown.push(String::new());
                let keep = self.shown.len().saturating_sub(n);
                self.shown.drain(..keep);
                self.shown_since = Some(t);
            }
            Mode::PaintOn => {
                self.end_shown(t);
                self.shown.push(String::new());
                self.shown_since = Some(t);
            }
            Mode::PopOn => self.loading.push(String::new()),
        }
    }

    /// Feed one caption byte pair (field 1) at time `t`.
    pub fn push(&mut self, b1: u8, b2: u8, t: f64) {
        let (b1, b2) = (b1 & 0x7f, b2 & 0x7f);
        if b1 == 0 && b2 == 0 {
            return; // padding
        }
        if (0x10..=0x1f).contains(&b1) {
            if self.last_control == Some((b1, b2)) {
                self.last_control = None; // the repeat every control code is sent with
                return;
            }
            self.last_control = Some((b1, b2));
            self.cc1 = b1 < 0x18;
            if !self.cc1 {
                return;
            }
            self.control(b1, b2, t);
            return;
        }
        self.last_control = None;
        if !self.cc1 || b1 < 0x20 {
            return;
        }
        if self.shown_since.is_none() && self.mode != Mode::PopOn {
            self.shown_since = Some(t);
        }
        self.put(basic(b1));
        if b2 >= 0x20 {
            self.put(basic(b2));
        }
    }

    fn control(&mut self, b1: u8, b2: u8, t: f64) {
        let c1 = b1 & !0x08; // channel 2 codes are channel 1's plus 8
        match (c1, b2) {
            (0x14 | 0x15, 0x20) => self.mode = Mode::PopOn, // resume caption loading
            (0x14 | 0x15, 0x25..=0x27) => {
                let rows = usize::from(b2 - 0x23);
                if self.mode != Mode::RollUp(rows) {
                    self.end_shown(t);
                    if !matches!(self.mode, Mode::RollUp(_)) {
                        self.shown.clear();
                    }
                }
                self.mode = Mode::RollUp(rows);
            }
            (0x14 | 0x15, 0x29) => self.mode = Mode::PaintOn,
            (0x14 | 0x15, 0x2c) => {
                // erase displayed memory
                self.end_shown(t);
                self.shown.clear();
            }
            (0x14 | 0x15, 0x2d) => self.carriage_return(t),
            (0x14 | 0x15, 0x2e) => self.loading = vec![String::new()], // erase non-displayed memory
            (0x14 | 0x15, 0x2f) => {
                // end of caption: show what was loaded
                self.end_shown(t);
                self.shown = std::mem::replace(&mut self.loading, vec![String::new()]);
                self.shown_since = Some(t);
            }
            (0x14 | 0x15, 0x21) => {
                if let Some(r) = self.target().last_mut() {
                    r.pop();
                }
            }
            (0x11, 0x30..=0x3f) => self.put(SPECIAL[usize::from(b2 - 0x30)]),
            (0x12, 0x20..=0x3f) | (0x13, 0x20..=0x3f) => {
                if let Some(r) = self.target().last_mut() {
                    r.pop();
                }
                self.put(if c1 == 0x12 { EXTENDED_12 } else { EXTENDED_13 }[usize::from(b2 - 0x20)]);
            }
            (0x11, 0x20..=0x2f) => self.put(' '), // mid-row style change: shows as a space
            (0x10..=0x17, 0x40..=0x7f) => {
                // preamble address: a new row (pop-on) or the row to type on
                let rows = self.target();
                if rows.last().is_some_and(|r| !r.trim().is_empty()) {
                    rows.push(String::new());
                }
            }
            _ => {}
        }
    }

    /// Cues finished so far (taken).
    pub fn take(&mut self) -> Vec<Cue> {
        std::mem::take(&mut self.cues)
    }

    /// End whatever is on screen at `t` and return the last cues.
    pub fn finish(&mut self, t: f64) -> Vec<Cue> {
        self.end_shown(t);
        self.take()
    }
}

fn stamp(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000);
    if h > 0 { format!("{h:02}:{m:02}:{s:02}.{ms:03}") } else { format!("{m:02}:{s:02}.{ms:03}") }
}

/// A WebVTT file written cue by cue (flushed, so a recording in progress can show what it has).
pub struct VttWriter {
    file: std::fs::File,
}

impl VttWriter {
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(b"WEBVTT\n")?;
        Ok(Self { file })
    }

    pub fn write(&mut self, cues: &[Cue]) -> std::io::Result<()> {
        if cues.is_empty() {
            return Ok(());
        }
        let mut s = String::new();
        for c in cues {
            let _ = write!(s, "\n{} --> {}\n{}\n", stamp(c.start), stamp(c.end), c.text);
        }
        self.file.write_all(s.as_bytes())?;
        self.file.flush()
    }
}

/// `MM:SS.mmm` or `HH:MM:SS.mmm` as seconds.
fn seconds(t: &str) -> Option<f64> {
    t.trim().split(':').try_fold(0.0, |acc, part| Some(acc * 60.0 + part.parse::<f64>().ok()?))
}

/// The cues of a WebVTT file (as `VttWriter` writes them; cue settings are ignored).
pub fn parse_vtt(text: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    for block in text.replace('\r', "").split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        let Some(at) = lines.iter().position(|l| l.contains("-->")) else { continue };
        let mut times = lines[at].split("-->");
        let (Some(a), Some(b)) = (times.next().and_then(seconds), times.next().and_then(|b| seconds(b.split_whitespace().next().unwrap_or("")))) else {
            continue;
        };
        let text = lines[at + 1..].join("\n");
        if !text.trim().is_empty() {
            cues.push(Cue { start: a, end: b, text });
        }
    }
    cues
}

/// The cues as they fall on a timeline from which `removed` stretches ((start, length), in order) were cut out:
/// moved back by what was removed before them, and dropped if they began inside a removed stretch.
pub fn cut_cues(cues: &[Cue], removed: &[(f64, f64)]) -> Vec<Cue> {
    let before = |t: f64| removed.iter().filter(|r| r.0 + r.1 <= t).map(|r| r.1).sum::<f64>();
    cues.iter()
        .filter(|c| !removed.iter().any(|r| c.start >= r.0 && c.start < r.0 + r.1))
        .map(|c| {
            // A cue running into a removed stretch ends where it starts.
            let end = removed.iter().find(|r| c.start < r.0 && c.end > r.0).map_or(c.end, |r| r.0);
            Cue { start: c.start - before(c.start), end: end - before(end.min(c.end)), text: c.text.clone() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(d: &mut Decoder, t: f64, s: &str) {
        let b = s.as_bytes();
        for pair in b.chunks(2) {
            d.push(pair[0], *pair.get(1).unwrap_or(&0), t);
        }
    }

    #[test]
    fn pop_on_captions_show_at_end_of_caption() {
        let mut d = Decoder::new();
        d.push(0x14, 0x20, 1.0); // RCL
        d.push(0x14, 0x20, 1.0); // its repeat
        d.push(0x14, 0x70, 1.0); // PAC
        feed(&mut d, 1.0, "HELLO THERE");
        d.push(0x14, 0x2f, 2.0); // EOC: shown
        d.push(0x14, 0x2c, 4.0); // EDM: gone
        assert_eq!(d.take(), vec![Cue { start: 2.0, end: 4.0, text: "HELLO THERE".into() }]);
    }

    #[test]
    fn roll_up_lines_become_cues_as_they_complete() {
        let mut d = Decoder::new();
        d.push(0x14, 0x25, 0.0); // RU2
        feed(&mut d, 0.5, "FIRST LINE");
        d.push(0x14, 0x2d, 1.0); // CR
        feed(&mut d, 1.5, "SECOND");
        d.push(0x14, 0x2d, 2.0);
        let cues = d.take();
        assert_eq!(cues[0], Cue { start: 0.5, end: 1.0, text: "FIRST LINE".into() });
        assert_eq!(cues[1], Cue { start: 1.0, end: 2.0, text: "FIRST LINE\nSECOND".into() });
    }

    #[test]
    fn special_and_extended_characters() {
        let mut d = Decoder::new();
        d.push(0x14, 0x29, 0.0); // paint-on
        feed(&mut d, 0.0, "CAF");
        d.push(0x12, 0x21, 0.0); // É, replacing the character before it (the F sent as its fallback)
        d.push(0x11, 0x37, 0.0); // ♪
        d.push(0x14, 0x2c, 1.0);
        assert_eq!(d.take()[0].text, "CAÉ♪");
    }

    #[test]
    fn reads_back_what_it_writes_and_cuts_out_stretches() {
        let dir = std::env::temp_dir().join(format!("cea608-vtt-{}.vtt", std::process::id()));
        let cues = vec![
            Cue { start: 1.0, end: 2.0, text: "BEFORE".into() },
            Cue { start: 10.5, end: 11.0, text: "IN THE BREAK".into() },
            Cue { start: 25.0, end: 27.5, text: "AFTER\nTWO LINES".into() },
        ];
        let mut w = VttWriter::create(&dir).unwrap();
        w.write(&cues).unwrap();
        let back = parse_vtt(&std::fs::read_to_string(&dir).unwrap());
        assert_eq!(back, cues);
        let cut = cut_cues(&back, &[(10.0, 12.0)]);
        assert_eq!(cut, vec![Cue { start: 1.0, end: 2.0, text: "BEFORE".into() }, Cue { start: 13.0, end: 15.5, text: "AFTER\nTWO LINES".into() }]);
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn writes_webvtt() {
        assert_eq!(stamp(61.5), "01:01.500");
        assert_eq!(stamp(3723.25), "01:02:03.250");
    }

    /// The captions of a whole capture (`DVR_TS_SAMPLE`), next to what ffmpeg reads from it.
    #[test]
    fn reads_broadcast_captions() {
        let Ok(path) = std::env::var("DVR_TS_SAMPLE") else { return };
        let bytes = std::fs::read(&path).unwrap();
        let mut demux = mpegts::Demuxer::new();
        let mut d = Decoder::new();
        let mut first = None;
        let mut last = 0.0;
        // Caption data rides with each picture in transmission order; it is read in display order.
        let mut by_time: Vec<(i64, mpegts::Triplets)> = demux
            .push(&bytes)
            .into_iter()
            .filter_map(|u| if let mpegts::Unit::Captions { pts, triplets } = u { Some((pts, triplets)) } else { None })
            .collect();
        by_time.sort_by_key(|(pts, _)| *pts);
        for (pts, triplets) in by_time {
            {
                let t = (pts - *first.get_or_insert(pts)) as f64 / 90_000.0;
                last = t;
                for (ty, a, b) in triplets {
                    if ty == 0 {
                        d.push(a, b, t);
                    }
                }
            }
        }
        let cues = d.finish(last);
        let text: Vec<String> = cues.iter().map(|c| c.text.replace('\n', " / ")).collect();
        eprintln!("{} cues: {:?}", cues.len(), &text[..text.len().min(6)]);
        assert!(!cues.is_empty());
    }
}
