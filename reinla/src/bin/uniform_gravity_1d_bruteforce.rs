use forger::prelude::*;
use reinla::lagrangian::one_dim::UniformGravity;
use reinla::lattice::one_dim::Lattice1D;

type S = (usize, i64);
type A = i64;
type P = EGreedyPolicy<A>;
type L = UniformGravity;
type E = Lattice1D<L>;

const M: f64 = 1.0;
const G: f64 = 2.0;

pub fn main() {
    let env = E::new(21, 0, 20, 4, L::new(M, G));
    let result = env.brute_force();
    println!("{:?}", result);
}
