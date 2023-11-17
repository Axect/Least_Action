use std::collections::HashSet;

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

fn main() {
    let mut env = E::new(21, 0, 20, 3, L::new(M, G));
    let mut agent = QEveryVisitMC::<S, A, P, E>::new(1.0);
    let mut policy = P::new(1.0, 0.99);

    // Annealing Procedure to find median of lagrangian
    let mut lagrangians = HashSet::new();
    for _ in 0 .. 100 {
        let mut state = (0, env.get_init_node());
        let mut episode = vec![];
        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            lagrangians.insert((-reward).to_bits());
            episode.push((state, action.unwrap(), reward));
            match next_state {
                Some(next_state) => {
                    state = next_state;
                }
                None => {
                    break;
                }
            }
        }

        agent.update(&episode);
        policy.decay_epsilon();
    }

    let lagrangians = lagrangians.into_iter().map(f64::from_bits).collect::<Vec<_>>();
    let min = lagrangians.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = lagrangians.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("min: {}, max: {}", min, max);
    env.set_l_min_max(*min, *max);

    // Main training
    let mut agent = QEveryVisitMC::<S, A, P, E>::new(1.0);
    let mut policy = P::new(1.0, 0.9);

    let mut history = Vec::new();
    for _ in 0..100 {
        let mut episode = vec![];
        let mut state = (0, env.get_init_node());

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            episode.push((state, action.unwrap(), reward));
            match next_state {
                Some(next_state) => {
                    state = next_state;
                }
                None => {
                    break;
                }
            }
        }

        agent.update(&episode);
        history.push(episode);
        policy.decay_epsilon();
    }

    // Test
    policy.eval();
    let mut episode = vec![];
    let mut phys_action = 0f64;
    let mut state = (0, env.get_init_node());
    env.reset_l_min_max();

    loop {
        let action = agent.select_action(&state, &mut policy, &env);
        let (next_state, r) = env.transition(&state, &action);
        episode.push((state, action.unwrap(), -r));
        phys_action -= r;
        match next_state {
            Some(next_state) => {
                state = next_state;
            }
            None => {
                break;
            }
        }
    }

    let q_min = agent.q_table.iter().map(|x| x.1).min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let q_max = agent.q_table.iter().map(|x| x.1).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();

    println!("{:?}", q_min);
    println!("{:?}", q_max);

    println!("{:?}", episode);
    println!("{:?}", phys_action);
}
