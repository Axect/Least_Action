use peroxide::fuga::*;

#[allow(non_snake_case)]
fn main() {
    let T = 1f64;
    let omega = linspace(1e-3, std::f64::consts::PI / 2f64, 1000);

    let x = omega.mul_s(T);

    let y_harmonic = x.fmap(harmonic_half);
    let y_pade = x.fmap(pade);
    let y_taylor = x.fmap(taylor); 
    let y_one = x.fmap(one_node);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("harmonic", Series::new(y_harmonic));
    df.push("pade", Series::new(y_pade));
    df.push("taylor", Series::new(y_taylor));
    df.push("one", Series::new(y_one));

    df.print();
    df.write_parquet("data.parquet", CompressionOptions::Uncompressed).unwrap();
}

fn harmonic_half(x: f64) -> f64 {
    1f64 / (x / 2f64).cos()
}

fn pade(x: f64) -> f64 {
    (1f64 + 1f64 / 48f64 * x.powi(2)) / (1f64 - 5f64 / 48f64 * x.powi(2))
}

fn taylor(x: f64) -> f64 {
    1f64 + x.powi(2) / 8f64
}

fn one_node(x: f64) -> f64 {
    (1f64 + 1f64 / 16f64 * x.powi(2)) / (1f64 - 1f64 / 16f64 * x.powi(2))
}
