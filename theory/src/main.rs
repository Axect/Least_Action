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
    let y_pade2 = x.fmap(pade2);
    let y_three = x.fmap(three_node);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("harmonic", Series::new(y_harmonic));
    df.push("pade", Series::new(y_pade));
    df.push("taylor", Series::new(y_taylor));
    df.push("one", Series::new(y_one));
    df.push("pade2", Series::new(y_pade2));
    df.push("three", Series::new(y_three));

    df.print();
    df.write_parquet("data.parquet", CompressionOptions::Uncompressed)
        .unwrap();
}

fn harmonic_half(x: f64) -> f64 {
    1f64 / (x / 2f64).cos()
}

fn pade(x: f64) -> f64 {
    (1f64 + 1f64 / 48f64 * x.powi(2)) / (1f64 - 5f64 / 48f64 * x.powi(2))
}

fn pade2(x: f64) -> f64 {
    (1f64 + 11f64 / 1008f64 * x.powi(2) + 13f64 * x.powi(4) / 241920f64)
        / (1f64 + 115f64 * x.powi(2) / 1008f64 + 313f64 * x.powi(4) / 241920f64)
}

fn taylor(x: f64) -> f64 {
    1f64 + x.powi(2) / 8f64
}

fn one_node(x: f64) -> f64 {
    (1f64 + 1f64 / 16f64 * x.powi(2)) / (1f64 - 1f64 / 16f64 * x.powi(2))
}

fn three_node(x: f64) -> f64 {
    (1f64 + x.powi(2) / 16f64 + x.powi(4) / 1024f64)
        / (1f64 - 3f64 * x.powi(2) / 16f64 + x.powi(4) / 1024f64)
}
