use std::f64::consts::PI;
use itertools::{Itertools, repeat_n};
use peroxide::fuga::*;
use rayon::prelude::*;

#[allow(non_snake_case)]
fn main() {
    let A = 0.0;
    let B = 20.0;
    let dq = 0.5;
    let node_pool = seq(A + dq, B - dq, dq);

    let omega = 1f64;
    let T = PI / 2f64;
    let potential = HarmonicOscillator1D::new(omega);

    let t_true = linspace(0, T, 1000);
    let q_true = t_true.fmap(|t| A * (omega * t).cos() + (B - A * (omega * T).cos()) / (omega *
        T).sin() * (omega * t).sin());

    let mut df = DataFrame::new(vec![]);
    df.push("t", Series::new(t_true));
    df.push("q", Series::new(q_true));
    df.print();

    df.write_parquet("true.parquet", CompressionOptions::Uncompressed).unwrap();

    let mut df = DataFrame::new(vec![]);

    let N_pool = 1 .. 4;
    for N in N_pool {
        let bf_nodes = bruteforce_1d_parallel(&node_pool, 2usize.pow(N as u32) - 1, T, A, B, potential);
        bf_nodes.print();
        let dc_nodes = divide_and_conquer_1d(&node_pool, N-1, T, A, B, potential);
        dc_nodes.print();
        let dcc_nodes = divide_and_conquer_and_correct_1d(&node_pool, N-1, T, A, B, potential);
        dcc_nodes.print();

        df.push(&format!("bf_{N}"), Series::new(bf_nodes));
        df.push(&format!("dc_{N}"), Series::new(dc_nodes));
        df.push(&format!("dcc_{N}"), Series::new(dcc_nodes));
    }

    df.print();

    df.write_parquet("data.parquet", CompressionOptions::Uncompressed).unwrap();
}

pub trait Potential1D: Clone {
    fn eval(&self, q: f64) -> f64;
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct HarmonicOscillator1D {
    omega: f64,
    omega2: f64,
}

impl HarmonicOscillator1D {
    pub fn new(omega: f64) -> HarmonicOscillator1D {
        HarmonicOscillator1D { omega, omega2: omega.powi(2) }
    }
}

impl Potential1D for HarmonicOscillator1D {
    fn eval(&self, q: f64) -> f64 {
        0.5 * self.omega2 * q.powi(2)
    }
}

#[allow(non_snake_case)]
pub fn action_1d<F: Potential1D>(nodes: &[f64], T: f64, start: f64, end: f64, potential: F) -> f64 {
    let N = nodes.len();
    let dt = T / ((N+1) as f64);

    let mut nodes = nodes.to_vec();
    nodes.insert(0, start);
    nodes.push(end);

    let nodes_head = &nodes[0..N+1];
    let nodes_tail = &nodes[1..];

    let mut action = 0.0;
    for (&q_i, &q_ip1) in nodes_head.iter().zip(nodes_tail) {
        let q = (q_i + q_ip1) / 2.0;
        let dq = (q_ip1 - q_i) / dt;

        action += 0.5 * dq.powi(2) - potential.eval(q);
    }
    action *= dt;
    action
}

#[allow(non_snake_case)]
pub fn bruteforce_1d<F: Potential1D>(node_pool: &[f64], N: usize, T: f64, start: f64, end: f64, potential: F) -> Vec<f64> {
    let mut best_action = std::f64::MAX;
    let mut best_nodes = vec![0f64; N];
    for nodes in node_pool.iter().cloned().combinations(N) {
        let action = action_1d(&nodes, T, start, end, potential.clone());
        if action < best_action {
            best_action = action;
            best_nodes[..].copy_from_slice(&nodes);
        }
    }
    best_nodes
}

#[allow(non_snake_case)]
pub fn bruteforce_1d_parallel<F: Potential1D + Sync>(node_pool: &[f64], N: usize, T: f64, start: f64, end: f64, potential: F) -> Vec<f64> {
    let node_combs = node_pool.iter().cloned().combinations(N).collect_vec();

    node_combs.into_par_iter().map(|nodes| {
        let action = action_1d(&nodes, T, start, end, potential.clone());
        (action, nodes.clone())
    }).max_by(|a, b| b.0.partial_cmp(&a.0).unwrap()).unwrap().1
}

#[allow(non_snake_case)]
pub fn divide_and_conquer_1d<F: Potential1D>(node_pool: &[f64], D: usize, T: f64, start: f64, end: f64, potential: F) -> Vec<f64> {
    if D == 0 {
        bruteforce_1d(node_pool, 1, T, start, end, potential.clone())
    } else {
        let q = bruteforce_1d(node_pool, 1, T, start, end, potential.clone());
        let index_q = node_pool.iter().position(|&x| x == q[0]).unwrap();
        let node_1 = &node_pool[0..index_q];
        let node_2 = &node_pool[index_q+1..];

        let best_node_1 = divide_and_conquer_1d(node_1, D - 1, T / 2f64, start, q[0], potential.clone());
        let best_node_2 = divide_and_conquer_1d(node_2, D - 1, T / 2f64, q[0], end, potential.clone());
        best_node_1.into_iter().chain(q).chain(best_node_2).collect()
    }
}

#[allow(non_snake_case)]
pub fn divide_and_conquer_and_correct_1d<F: Potential1D>(node_pool: &[f64], D: usize, T: f64, start: f64, end: f64, potential: F) -> Vec<f64> {
    if D == 0 {
        bruteforce_1d(node_pool, 1, T, start, end, potential.clone())
    } else {
        let q = bruteforce_1d(node_pool, 1, T, start, end, potential.clone());
        let index_q = node_pool.iter().position(|&x| x == q[0]).unwrap();
        let node_1 = &node_pool[0..index_q];
        let node_2 = &node_pool[index_q+1..];

        let best_node_1 = divide_and_conquer_and_correct_1d(node_1, D - 1, T / 2f64, start, q[0], potential.clone());
        let best_node_2 = divide_and_conquer_and_correct_1d(node_2, D - 1, T / 2f64, q[0], end, potential.clone());
        let mut best_node = best_node_1.clone().into_iter().chain(q).chain(best_node_2.clone()).collect_vec();
        let mut action = action_1d(&best_node, T, start, end, potential.clone());

        if index_q > 0 && index_q < node_pool.len() - 1 {
            let q_in1 = node_pool[index_q - 1];
            let q_ip1 = node_pool[index_q + 1];

        }
        todo!()

        //let indices_1 = best_node_1.iter().map(|x| node_pool.iter().position(|&y| y == *x).unwrap());
        //let indices_2 = best_node_2.iter().map(|x| node_pool.iter().position(|&y| y == *x).unwrap());
        //let indices = indices_1.chain(vec![index_q]).chain(indices_2).map(|i| i as i32).collect_vec();

        //let best_nodes: Vec<f64> = best_node_1.into_iter().chain(q).chain(best_node_2).collect();

        //// Correct
        //let correct_pool = repeat_n(vec![-1, 0, 1], best_nodes.len()).multi_cartesian_product();
        //correct_pool.map(|cs| {
        //    let nodes = indices.iter().zip(cs.iter()).map(|(i, c)| *i + c).filter(|&i| i >= 0 && i < best_nodes.len() as i32).map(|i| best_nodes[i as usize]).collect_vec();
        //    let action = action_1d(&nodes, T, start, end, potential.clone());
        //    (action, nodes)
        //}).max_by(|a, b| b.0.partial_cmp(&a.0).unwrap()).unwrap().1
    }
}
