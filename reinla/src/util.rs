pub fn elu(x: f64) -> f64 {
    if x > 0f64 {
        x
    } else {
        x.exp() - 1f64
    }
}
