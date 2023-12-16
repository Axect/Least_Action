use crate::lagrangian::Lagrangian;
//use crate::util::elu;
use forger::env::Env;
use std::hash::{Hash, Hasher};

pub type S<const T: usize> = State1D<T>;
pub type A = Move1D;

#[derive(Debug, Clone, Copy, Eq)]
pub struct State1D<const T: usize> {
    pub state: [usize; T],
}

impl<const T: usize> State1D<T> {
    pub fn new(state: [usize; T]) -> Self {
        Self { state }
    }
}

impl<const T: usize> PartialEq for State1D<T> {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}

impl<const T: usize> Hash for State1D<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum Move1D {
    Up(usize),
    Down(usize),
    Hold,
}

pub struct TimeLattice1D<L: Lagrangian, const T: usize> {
    num_nodes: usize,
    t_vec: [usize; T],
    lagrangian: L,
    s_min_max: Option<(f64, f64)>,
}

impl<L: Lagrangian<Q = f64>, const T: usize> TimeLattice1D<L, T> {
    pub fn new(num_nodes: usize, lagrangian: L) -> Self {
        Self {
            num_nodes,
            t_vec: [0; T],
            lagrangian,
            s_min_max: None,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn t(&self) -> usize {
        self.t_vec.len()
    }

    #[allow(non_snake_case)]
    pub fn L(&self, q: f64, dq: f64) -> f64 {
        self.lagrangian.calc(&q, &dq)
    }

    pub fn set_s_min_max(&mut self, s_min: f64, s_max: f64) {
        self.s_min_max = Some((s_min, s_max));
    }
}

impl<L: Lagrangian<Q = f64>, const T: usize> Env<S<T>, A> for TimeLattice1D<L, T> {
    fn is_terminal(&self, _state: &S<T>) -> bool {
        unimplemented!()
    }

    fn is_goal(&self, _state: &S<T>) -> bool {
        unimplemented!()
    }

    fn transition(&self, state: &S<T>, action: &Option<A>) -> (Option<S<T>>, f64) {
        let action = action.unwrap();

        // Obtain next action via sum of lagrangians
        let mut state_vec = state.state.to_vec();
        state_vec.insert(0, 0);
        state_vec.insert(T + 1, self.num_nodes());
        match action {
            Move1D::Up(s) => {
                state_vec[s + 1] += 1;
            }
            Move1D::Down(s) => {
                state_vec[s + 1] -= 1;
            }
            Move1D::Hold => {}
        }
        let s = state_vec[0..T + 1]
            .iter()
            .zip(state_vec[1..T + 2].iter())
            .fold(0f64, |acc, (q_c, q_n)| {
                let q = (*q_c as f64 + *q_n as f64) / 2.0;
                let dq = *q_n as f64 - *q_c as f64;
                acc + self.L(q, dq)
            });

        let mut state_new = [0usize; T];
        state_new.copy_from_slice(&state_vec[1..T + 1]);

        let reward = match self.s_min_max {
            Some((s_min, s_max)) => {
                let s = (s - s_min) / (s_max - s_min);
                //-s.exp()
                //-(s - s_min).powi(2)
                -s.powi((T as i32 + 2).max(5))
            }
            None => -s,
        };
        (Some(State1D::new(state_new)), reward)
    }

    fn available_actions(&self, state: &S<T>) -> Vec<A> {
        let mut actions = vec![Move1D::Hold];
        for (i, s) in state.state.iter().enumerate() {
            if *s < self.num_nodes() {
                actions.push(Move1D::Up(i));
            } else if *s > 0 {
                actions.push(Move1D::Down(i));
            }
        }
        actions
    }
}
