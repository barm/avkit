//! `cargo run --release --example predict -- <model> <input name> <shape,comma,separated> <input.f32> <output name> [ref.f32]`
//! Runs a model on a raw little-endian float32 file; with a reference output, prints the cosine similarity.
use std::time::Instant;

fn read(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let shape: Vec<usize> = a[2].split(',').map(|s| s.parse().unwrap()).collect();
    let t = Instant::now();
    let m = apple_coreml::Model::load(a[0].as_ref(), apple_coreml::Compute::All)?;
    println!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    let x = read(&a[3]);
    let y = m.predict(&a[1], &shape, &x, &a[4])?;
    let t = Instant::now();
    for _ in 0..50 {
        m.predict(&a[1], &shape, &x, &a[4])?;
    }
    println!("{} outputs, {:.2} ms per prediction", y.len(), t.elapsed().as_secs_f64() * 1000.0 / 50.0);
    if let Some(r) = a.get(5) {
        let r = read(r);
        let dot: f32 = y.iter().zip(&r).map(|(a, b)| a * b).sum();
        let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("cosine to reference {:.6}", dot / (n(&y) * n(&r)));
    }
    Ok(())
}
