pub trait Lagrangian {
    type Q;

    fn calc(&self, q: &Self::Q, dq: &Self::Q) -> f64;
}

pub mod one_dim;
