use reinla::lagrangian::one_dim::UniformGravity;
use reinla::lattice::one_dim::Lattice1D;

type E = Lattice1D<L>;
type L = UniformGravity;

const M: f64 = 1.0;
const G: f64 = 2.0;

pub fn main() {
    let env = E::new(21, 0, 20, 4, L::new(M, G));
    let result = env.brute_force();
    println!("{:?}", result);
}
