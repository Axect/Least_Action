use std::collections::HashSet;

use forger::prelude::*;
use reinla::lagrangian::one_dim::FreeBody;
use reinla::time_lattice::one_dim::{TimeLattice1D, Move1D, State1D};
use peroxide::fuga::*;

const N: usize = 4;
const T: usize = 3;
type S = State1D<T>;
type A = Move1D;
type P = EGreedyPolicy<A>;
type L = FreeBody;
type E = TimeLattice1D<FreeBody, T>;

const M: f64 = 1.0;

fn main() {
    let mut env = E::new(N, L::new(M));

    // Annealing
    let mut agent = QTD0::<S, A, P, E>::new(0.9, 1f64, 1f64);
    let mut policy = P::new(1.0, 0.9);

    let mut actions = HashSet::new();
    for _ in 0 .. 100 {
        agent.reset_count();
        let u = Uniform(0, N as u32);
        let random_state = u.sample(T).into_iter().map(|x| x as usize).collect::<Vec<usize>>();
        let mut state = [0; T];
        state.copy_from_slice(&random_state);
        let mut state = State1D::new(state);

        for _ in 0 .. 1000 {
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

    let actions = actions.into_iter().map(f64::from_bits).collect::<Vec<f64>>();
    let s_min = actions.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let s_max = actions.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    println!("s_min: {}, s_max: {}", s_min, s_max);
    env.set_s_min_max(*s_min, *s_max);

    // Main training
    let mut agent = QTD0::<S, A, P, E>::new(0.9, 1f64, 1f64);
    let mut policy = P::new(1.0, 0.9);

    let mut history = Vec::new();
    for _ in 0..100 {
        agent.reset_count();
        let mut episode = vec![];

        // Random initial state
        //let u = Uniform(0, N as u32);
        //let random_state = u.sample(T).into_iter().map(|x| x as usize).collect::<Vec<usize>>();
        let random_state = [0; T];
        let mut state = [0; T];
        state.copy_from_slice(&random_state);
        let mut state = State1D::new(state);

        for _ in 0 .. 1000 {
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

    //println!("{:?}", history.iter().map(|x| x.len()).collect::<Vec<usize>>());

    // Test
    policy.eval();
    agent.reset_count();
    let mut episode = vec![];
    let mut state = State1D::new([0; T]);

    for _ in 0 .. 100 {
        let action = agent.select_action(&state, &mut policy, &env);
        let (next_state, _) = env.transition(&state, &action);
        episode.push((state, action.unwrap()));
        state = next_state.unwrap();

        if action.unwrap() == Move1D::Hold {
            break;
        }
    }

    let q_min = agent.q_table.iter().map(|x| x.1).min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let q_max = agent.q_table.iter().map(|x| x.1).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();

    println!("{:?}", q_min);
    println!("{:?}", q_max);

    println!("{:?}", episode);
}
