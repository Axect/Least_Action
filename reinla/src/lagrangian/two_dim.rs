use crate::lagrangian::Lagrangian;

// ┌──────────────────────────────────────────────────────────┐
//  Free Body
// └──────────────────────────────────────────────────────────┘
pub struct FreeBody {
    mass: f64,
}

impl FreeBody {
    pub fn new(mass: f64) -> Self {
        Self { mass }
    }
}

impl Lagrangian for FreeBody {
    type Q = (f64, f64);

    fn calc(&self, _q: &Self::Q, dq: &Self::Q) -> f64 {
        0.5 * self.mass * (dq.0.powi(2) + dq.1.powi(2))
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Uniform Gravity
// └──────────────────────────────────────────────────────────┘
pub struct UniformGravity {
    mass: f64,
    g: f64,
}

impl UniformGravity {
    pub fn new(mass: f64, g: f64) -> Self {
        Self { mass, g }
    }
}

impl Lagrangian for UniformGravity {
    type Q = (f64, f64);

    fn calc(&self, q: &Self::Q, dq: &Self::Q) -> f64 {
        0.5 * self.mass * (dq.0.powi(2) + dq.1.powi(2)) - self.mass * self.g * q.1
    }
}
