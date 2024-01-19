use itertools::{repeat_n, Itertools};
use peroxide::{fuga::*, traits::float::FloatWithPrecision};
use rayon::prelude::*;
use std::f64::consts::PI;

#[allow(non_snake_case)]
fn main() {
    let A = 0.0;
    let B = 20.0;
    let dq = 0.1;
    let node_pool = seq(A + dq, B, dq);
    let node_pool = node_pool.fmap(|x| x.round_with_precision(1));

    let omega = 1f64;
    let T = PI / 2f64;
    let potential = HarmonicOscillator1D::new(omega);

    let t_true = linspace(0, T, 1000);
    let q_true = t_true.fmap(|t| {
        A * (omega * t).cos() + (B - A * (omega * T).cos()) / (omega * T).sin() * (omega * t).sin()
    });

    let mut df = DataFrame::new(vec![]);
    df.push("t", Series::new(t_true));
    df.push("q", Series::new(q_true));
    df.print();

    df.write_parquet("true.parquet", CompressionOptions::Uncompressed)
        .unwrap();

    let mut df = DataFrame::new(vec![]);

    let N_pool = 1..6;
    for N in N_pool {
        //let bf_nodes = bruteforce_1d_parallel(&node_pool, 2usize.pow(N as u32) - 1, T, A, B, potential);
        //bf_nodes.print();
        let dc_nodes = divide_and_conquer_1d(&node_pool, N - 1, T, A, B, potential);
        dc_nodes.print();
        let dcc_nodes = divide_and_conquer_and_correct_1d(&node_pool, N - 1, T, A, dq, B, potential);
        dcc_nodes.print();

        //df.push(&format!("bf_{N}"), Series::new(bf_nodes));
        df.push(&format!("dc_{N}"), Series::new(dc_nodes));
        df.push(&format!("dcc_{N}"), Series::new(dcc_nodes));
    }

    df.print();

    df.write_parquet("data.parquet", CompressionOptions::Uncompressed)
        .unwrap();
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
        HarmonicOscillator1D {
            omega,
            omega2: omega.powi(2),
        }
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
    let dt = T / ((N + 1) as f64);

    let mut nodes = nodes.to_vec();
    nodes.insert(0, start);
    nodes.push(end);

    let nodes_head = &nodes[0..N + 1];
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
pub fn bruteforce_1d<F: Potential1D>(
    node_pool: &[f64],
    N: usize,
    T: f64,
    start: f64,
    end: f64,
    potential: F,
) -> Vec<f64> {
    if node_pool.len() == 1 {
        return vec![node_pool[0]];
    }

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
pub fn bruteforce_1d_parallel<F: Potential1D + Sync>(
    node_pool: &[f64],
    N: usize,
    T: f64,
    start: f64,
    end: f64,
    potential: F,
) -> Vec<f64> {
    let node_combs = node_pool.iter().cloned().combinations(N).collect_vec();

    node_combs
        .into_par_iter()
        .map(|nodes| {
            let action = action_1d(&nodes, T, start, end, potential.clone());
            (action, nodes.clone())
        })
        .max_by(|a, b| b.0.partial_cmp(&a.0).unwrap())
        .unwrap()
        .1
}

#[allow(non_snake_case)]
pub fn divide_and_conquer_1d<F: Potential1D>(
    node_pool: &[f64],
    D: usize,
    T: f64,
    start: f64,
    end: f64,
    potential: F,
) -> Vec<f64> {
    if D == 0 {
        bruteforce_1d(node_pool, 1, T, start, end, potential.clone())
    } else {
        let q = bruteforce_1d(node_pool, 1, T, start, end, potential.clone());
        let index_q = node_pool.iter().position(|&x| x == q[0]).unwrap();
        let node_1 = &node_pool[0..index_q];
        let node_2 = &node_pool[index_q + 1..];

        let best_node_1 =
            divide_and_conquer_1d(node_1, D - 1, T / 2f64, start, q[0], potential.clone());
        let best_node_2 =
            divide_and_conquer_1d(node_2, D - 1, T / 2f64, q[0], end, potential.clone());
        best_node_1
            .into_iter()
            .chain(q)
            .chain(best_node_2)
            .collect()
    }
}

#[allow(non_snake_case)]
pub fn divide_and_conquer_and_correct_1d<F: Potential1D>(
    node_pool: &[f64],
    D: usize,
    T: f64,
    start: f64,
    dq: f64,
    end: f64,
    potential: F,
) -> Vec<f64> {
    if (end - start).abs().round_with_precision(1) <= dq {
        return vec![(start + end) / 2f64];
    }

    if D == 0 {
        bruteforce_1d(node_pool, 1, T, start, end, potential.clone())
    } else {
        let q = bruteforce_1d(node_pool, 1, T, start, end, potential.clone());
        let q_i = q[0].round_with_precision(1);
        let index_q = node_pool.iter().position(|&x| x == q_i).unwrap();

        let node_1 = &node_pool[0..index_q];
        let node_2 = &node_pool[index_q + 1..];
        let best_node_1 = divide_and_conquer_and_correct_1d(
            node_1,
            D - 1,
            T / 2f64,
            start,
            dq,
            q_i,
            potential.clone(),
        );
        let best_node_2 =
            divide_and_conquer_and_correct_1d(node_2, D - 1, T / 2f64, q_i, dq, end, potential.clone());
        let mut best_node = best_node_1
            .clone()
            .into_iter()
            .chain(vec![q_i])
            .chain(best_node_2.clone())
            .collect_vec();
        let mut best_action = action_1d(&best_node, T, start, end, potential.clone());

        let D_isize = D as isize;
        let d_vec = (-D_isize..0).chain(1..D_isize + 1);

        for d in d_vec {
            let index_d = index_q as isize + d;
            if index_d < 0 || index_d >= node_pool.len() as isize {
                continue;
            } else {
                let index_d = index_d as usize;
                let q_id = node_pool[index_d];
                let node_d_1 = &node_pool[0..index_d];
                let node_d_2 = &node_pool[index_d + 1..];

                let best_node_d_1 = divide_and_conquer_and_correct_1d(
                    node_d_1,
                    D - 1,
                    T / 2f64,
                    start,
                    dq,
                    q_id,
                    potential.clone(),
                );
                let best_node_d_2 = divide_and_conquer_and_correct_1d(
                    node_d_2,
                    D - 1,
                    T / 2f64,
                    q_id,
                    dq,
                    end,
                    potential.clone(),
                );
                let best_node_d = best_node_d_1
                    .clone()
                    .into_iter()
                    .chain(vec![q_id])
                    .chain(best_node_d_2.clone())
                    .collect_vec();
                let action_d = action_1d(&best_node_d, T, start, end, potential.clone());

                if action_d < best_action {
                    best_action = action_d;
                    best_node = best_node_d;
                }
            }
        }

        best_node
    }
}
