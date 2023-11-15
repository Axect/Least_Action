use forger::prelude::*;
use reinla::lagrangian::one_dim::UniformGravity;
use reinla::lattice::one_dim::Lattice1D;

type S = (i64, usize);
type A = i64;
type P = EGreedyPolicy<A>;
type L = UniformGravity;
type E = Lattice1D<L>;

const M: f64 = 1.0;
const G: f64 = 2.0;

fn main() {
    let env = E::new(0, 24, 4, L::new(M, G));
    let mut agent = QTD0::<S, A, P, E>::new(1.0, 0.1f64, 1f64);
    let mut policy = P::new(1.0, 0.9);

    let mut history = Vec::new();
    for _ in 0..100 {
        agent.reset_count();
        let mut episode = vec![];
        let mut state = (0, 0);

        loop {
            let action = agent.select_action(&state, &mut policy, &env);
            let (next_state, reward) = env.transition(&state, &action);
            match next_state {
                Some(next_state) => {
                    let step = (
                        state,
                        action.unwrap(),
                        reward,
                        Some(next_state),
                        env.available_actions(&next_state),
                    );
                    agent.update(&step);
                    episode.push(step);
                    state = next_state;
                }
                None => {
                    let step = (state, action.unwrap(), reward, None, Vec::new());
                    agent.update(&step);
                    episode.push(step);
                    break;
                }
            }
        }

        history.push(episode);
        policy.decay_epsilon();
    }

    // Test
    policy.eval();
    agent.reset_count();
    let mut episode = vec![];
    let mut state = (0, 0);

    loop {
        let action = agent.select_action(&state, &mut policy, &env);
        let (next_state, _) = env.transition(&state, &action);
        episode.push((state, action.unwrap()));
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
    println!("{:?}", env.l_min());
    println!("{:?}", env.l_max());

    println!("{:?}", episode);
}
