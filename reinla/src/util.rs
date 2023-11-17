use std::f64::consts::PI;

pub fn elu(x: f64) -> f64 {
    if x > 0f64 {
        x
    } else {
        x.exp() - 1f64
    }
}

pub fn gelu(x: f64) -> f64 {
    0.5f64 * x * (1f64 + ((2f64 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

pub fn huber(x: f64, delta: f64) -> f64 {
    if x.abs() < delta {
        0.5 * x.powi(2)
    } else {
        delta * (x.abs() - 0.5 * delta)
    }
}
