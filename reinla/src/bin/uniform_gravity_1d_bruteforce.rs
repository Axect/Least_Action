use reinla::lagrangian::one_dim::UniformGravity;
use reinla::lattice::one_dim::Lattice1D;
use std::env::args;

type E = Lattice1D<L>;
type L = UniformGravity;

const M: f64 = 1.0;
const G: f64 = 10.0;

#[allow(non_snake_case)]
pub fn main() {
    let args = args().collect::<Vec<String>>();
    let N = args[1].parse::<usize>().unwrap();
    let m = args[2].parse::<usize>().unwrap();
    let env = E::new(N+1, 0, N as i64, m, L::new(M, G));
    let result = env.brute_force();
    println!("{:?}", result);
}
