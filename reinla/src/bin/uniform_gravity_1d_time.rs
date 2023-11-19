use forger::prelude::*;
use peroxide::fuga::*;
use reinla::lagrangian::one_dim::UniformGravity;
use reinla::time_lattice::one_dim::{Move1D, State1D, TimeLattice1D};
use std::collections::{HashMap, HashSet};

const N: usize = 10;
const T: usize = 4;
type S = State1D<T>;
type A = Move1D;
type P = EGreedyPolicy<A>;
type L = UniformGravity;
type E = TimeLattice1D<UniformGravity, T>;

const M: f64 = 1.0;
const G: f64 = 1.0;

fn main() {
    let n_min = (N.pow(T as u32) * 10).max(1000);
    let p_min = 1f64 - (10f64 / n_min as f64);

    println!("n_min: {}, p_min: {}", n_min, p_min);

    let mut env = E::new(N, L::new(M, G));

    // Annealing
    let mut agent = QTD0::<S, A, P, E>::new(0.9, 1f64, 1f64);
    let mut policy = P::new(1.0, p_min);
    let mut actions = HashSet::new();
    for _ in 0..n_min {
        agent.reset_count();
        let u = Uniform(0, N as u32);
        let random_state = u
            .sample(T)
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<usize>>();
        let mut state = [0; T];
        state.copy_from_slice(&random_state);
        let mut state = State1D::new(state);

        for _ in 0..1000 {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            let next_state = next_state.unwrap();
            let action = action.unwrap();
            actions.insert((-reward).to_bits());
            let step = (
                state,
                action,
                reward,
                Some(next_state),
                env.available_actions(&next_state),
            );

            agent.update(&step);
            state = next_state;

            if action == Move1D::Hold {
                break;
            }
        }
    }

    let actions = actions
        .into_iter()
        .map(f64::from_bits)
        .collect::<Vec<f64>>();
    let s_min = actions
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let s_max = actions
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    println!("s_min: {}, s_max: {}", s_min, s_max);
    env.set_s_min_max(*s_min, *s_max);

    // Main training
    let mut agent = QTD0::<S, A, P, E>::new(0.9, 1f64, 1f64);
    let mut policy = P::new(1.0, p_min);

    let mut history = Vec::new();
    for _ in 0..n_min {
        agent.reset_count();
        let mut episode = vec![];

        // Random initial state
        let u = Uniform(0, N as u32);
        let random_state = u
            .sample(T)
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<usize>>();
        let mut state = [0; T];
        state.copy_from_slice(&random_state);
        let mut state = State1D::new(state);

        for _ in 0..1000 {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            let next_state = next_state.unwrap();
            let action = action.unwrap();
            let step = (
                state,
                action,
                reward,
                Some(next_state),
                env.available_actions(&next_state),
            );
            agent.update(&step);
            episode.push(step);
            state = next_state;

            if action == Move1D::Hold {
                break;
            }
        }

        history.push(episode);
        policy.decay_epsilon();
    }

    // Test
    let mut result = HashMap::new();
    policy.eval();
    agent.reset_count();
    let mut episode = vec![];

    for _ in 0..n_min / 10 {
        let u = Uniform(0, N as u32);
        let random_state = u
            .sample(T)
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<usize>>();
        let mut state = [0; T];
        state.copy_from_slice(&random_state);
        let mut state = State1D::new(state);
        for _ in 0..100 {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, _) = env.transition(&state, &action);
            episode.push((state, action.unwrap()));
            state = next_state.unwrap();

            if action.unwrap() == Move1D::Hold {
                // Increment count
                let count = result.entry(state.state).or_insert(0);
                *count += 1;
                break;
            }
        }
    }

    let q_min = agent
        .q_table
        .iter()
        .map(|x| x.1)
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let q_max = agent
        .q_table
        .iter()
        .map(|x| x.1)
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    println!("Q_min: {:.4}\tQ_max: {:.4}", q_min, q_max);

    // Sort result via values
    let mut result_vec = result.iter().collect::<Vec<(&[usize; T], &usize)>>();
    result_vec.sort_by(|x, y| y.1.partial_cmp(x.1).unwrap());
    if result_vec.len() > 5 {
        for (x, y) in result_vec[0..5].iter() {
            println!("Path: {:?}\tCount: {}", x, y);
        }
    } else {
        for (x, y) in result_vec.iter() {
            println!("Path: {:?}\tCount: {}", x, y);
        }
    }
}
