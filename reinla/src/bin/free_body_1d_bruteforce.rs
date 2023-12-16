use reinla::lagrangian::one_dim::FreeBody;
use reinla::lattice::one_dim::Lattice1D;
use std::env::args;

const M: f64 = 1.0;

#[allow(non_snake_case)]
pub fn main() {
    let args = args().collect::<Vec<String>>();
    let N = args[1].parse::<usize>().unwrap();
    let m = args[2].parse::<usize>().unwrap();
    let env = Lattice1D::new(N+1, 0, N as i64, m, FreeBody::new(M));
    let result = env.brute_force();
    println!("{:?}", result);
}
