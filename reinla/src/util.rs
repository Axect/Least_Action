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

pub fn comb(n: i64, k: usize) -> Vec<Vec<i64>> {
    let mut p = (1i64 .. (k+1) as i64).collect::<Vec<i64>>();
    let mut result: Vec<Vec<i64>> = Vec::new();

    loop {
        result.push(p.clone());
        let mut i = k;

        while i > 0 && p[i-1] == (n as usize+i-k) as i64 {
            i -= 1;
        }
        if i > 0 {
            p[i-1] += 1;
            for j in i .. k {
                p[j] = p[j-1] + 1;
            }
        } else {
            break;
        }
    }
    result
}
