use crate::lagrangian::Lagrangian;
use forger::env::Env;

type S = (i64, usize);

#[derive(Debug)]
pub struct Lattice1D<L: Lagrangian> {
    init_node: i64,
    end_node: i64,
    t: usize,
    lagrangian: L,
}

impl<L: Lagrangian<Q = f64>> Lattice1D<L> {
    pub fn new(init_node: i64, end_node: i64, t: usize, lagrangian: L) -> Lattice1D<L> {
        Lattice1D {
            init_node,
            end_node,
            t,
            lagrangian,
        }
    }

    pub fn lagrangian(&self) -> &L {
        &self.lagrangian
    }

    pub fn v_mean(&self) -> f64 {
        (self.end_node - self.init_node) as f64 / self.t as f64
    }
}

impl<L: Lagrangian<Q = f64>> Env<S, i64> for Lattice1D<L> {
    fn is_terminal(&self, state: &S) -> bool {
        state.1 >= self.t
    }

    fn is_goal(&self, state: &S) -> bool {
        state.0 == self.end_node
    }

    fn transition(&self, state: &S, action: &Option<i64>) -> (Option<S>, f64) {
        if self.is_terminal(state) {
            return (None, 2.0);
        } else if self.is_goal(state) {
            return (None, 1.0);
        }
        let action = action.as_ref().unwrap();
        let r = if *action == 0 { -1.0 } else { 0.0 };
        let q_curr = state.0;
        let q_next = state.0 + action;

        let q = (q_curr + q_next) as f64 / 2f64;
        let dq = (q_next - q_curr) as f64;

        let l = self.lagrangian().calc(&q, &dq);

        (
            Some((q_next, state.1 + 1)),
            -(l - 0.5f64 * self.v_mean().powi(2)).tanh() + r,
        )
    }

    fn available_actions(&self, state: &S) -> Vec<i64> {
        (0..self.end_node - state.0 + 1).collect()
    }
}
