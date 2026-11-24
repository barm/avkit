//! `cargo run --release --example shrink -- <input.mp4> <out dir> [height] [quality]`

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(input), Some(out)) = (args.first(), args.get(1)) else { anyhow::bail!("usage: shrink <input.mp4> <out dir> [height] [quality]") };
    let mut s = fmp4_shrink::Settings::default();
    if let Some(h) = args.get(2) {
        s.height = h.parse()?;
    }
    if let Some(q) = args.get(3) {
        s.quality = q.parse()?;
    }
    let started = std::time::Instant::now();
    let mut last = 0.0;
    let summary = fmp4_shrink::shrink(input.as_ref(), out.as_ref(), &s, &mut |secs| {
        if secs - last >= 600.0 {
            last = secs;
            eprintln!("{:.0} min done, {:.0}x real time", secs / 60.0, secs / started.elapsed().as_secs_f64());
        }
    })?;
    let t = started.elapsed().as_secs_f64();
    eprintln!("{summary:?}: {:.0} s, {:.0}x real time, {:.0} kbit/s", t, summary.duration / t, summary.bytes as f64 * 8.0 / summary.duration / 1000.0);
    Ok(())
}
