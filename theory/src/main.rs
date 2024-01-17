use std::f64::consts::PI;

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
    let y_three = x.fmap(three_node);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("harmonic", Series::new(y_harmonic));
    df.push("pade", Series::new(y_pade));
    df.push("taylor", Series::new(y_taylor));
    df.push("one", Series::new(y_one));
    
    df.push("three", Series::new(y_three));

    df.print();
    df.write_parquet("data.parquet", CompressionOptions::Uncompressed)
        .unwrap();

    let omega_t = PI / 2f64;

    let k = seq(1, 10, 1);
    let y_hat = k.fmap(|x| one_least_harmonic(x, omega_t));
    let y_true = k.fmap(|_| 1f64 / (omega_t / 2f64).cos()); 
    let mut dg = DataFrame::new(vec![]);
    dg.push("k", Series::new(k));
    dg.push("y_hat", Series::new(y_hat));
    dg.push("y_true", Series::new(y_true));
    dg.print();
    dg.write_parquet("data2.parquet", CompressionOptions::Uncompressed)
        .unwrap();

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

fn three_node(x: f64) -> f64 {
    (1f64 + x.powi(2) / 32f64 + x.powi(4) / 4096f64)
        / (1f64 - 3f64 * x.powi(2) / 32f64 + x.powi(4) / 4096f64)
}

#[allow(non_snake_case)]
fn least_harmonic(N: usize, omega: f64, T: f64, A: f64, B: f64) -> Vec<f64> {
    let f = omega.powi(2) * T.powi(2) / (4f64 * (N+1).pow(2) as f64);
    let g = (1f64 - f) / (1f64 + f);
    let theta = g.acos();
    let a = (0 ..=N).map(|n| ((n + 1) as f64 * theta).sin()).collect::<Vec<_>>();
    let B_N = B / a[N];

    let mut S = 0f64;
    let mut q = vec![0f64; N];

    for n in (0 .. N).rev() {
        S += 1f64 / (a[n] * a[n-1]);
        q[n] = a[n-1] * (B_N + A * S);
    }

    q
}

#[allow(non_snake_case)]
fn one_least_harmonic(k: f64, x: f64) -> f64 {
    let a = x / 4f64;
    let f = a.powi(2) / k.powi(2);
    let g = (1f64 - f) / (1f64 + f);
    let theta = g.acos();
    1f64 / (k * theta).cos()
}
