use reinla::util::comb;

pub fn main() {
    for q_vec in comb(5, 3) {
        println!("{:?}", q_vec);
    }
}
