use crate::lagrangian::Lagrangian;
use crate::util::elu;
use forger::env::Env;

type S = (usize, i64);

#[derive(Debug)]
pub struct Lattice1D<L: Lagrangian> {
    num_nodes: usize,
    init_node: i64,
    end_node: i64,
    t: usize,
    lagrangian: L,
    _l_min_max: Option<(f64, f64)>,
}

impl<L: Lagrangian<Q = f64>> Lattice1D<L> {
    pub fn new(num_nodes: usize, init_node: i64, end_node: i64, t: usize, lagrangian: L) -> Lattice1D<L> {
        Lattice1D {
            num_nodes,
            init_node,
            end_node,
            t,
            lagrangian,
            _l_min_max: None
        }
    }

    pub fn lagrangian(&self) -> &L {
        &self.lagrangian
    }

    pub fn get_init_node(&self) -> i64 {
        self.init_node
    }

    pub fn get_end_node(&self) -> i64 {
        self.end_node
    }

    pub fn set_l_min_max(&mut self, l_min: f64, l_max: f64) {
        self._l_min_max = Some((l_min, l_max));
    }

    pub fn reset_l_min_max(&mut self) {
        self._l_min_max = None;
    }

    pub fn reward(&self, q: f64, dq: f64) -> f64 {
        let l = self.lagrangian.calc(&q, &dq);
        let c = self._l_min_max;
        match c {
            Some((l_min, l_max)) => {
                //let c_half = (2f64 * l_min + l_max) / 3f64;
                //let l_minmax = 6f64 * (l - c_half) / (l_max - l_min);
                //-(elu(l_minmax) + 1f64).powi(4) + 1f64
                let l_minmax = (l - l_min) / (l_max - l_min);
                -l_minmax.powi(2)
            }
            None => {
                -l
            }
        }
    }
}

impl<L: Lagrangian<Q = f64>> Env<S, i64> for Lattice1D<L> {
    fn is_terminal(&self, state: &S) -> bool {
        state.0 >= self.t
    }

    fn is_goal(&self, state: &S) -> bool {
        state.1 == self.end_node
    }

    fn transition(&self, state: &S, action: &Option<i64>) -> (Option<S>, f64) {
        //if self.is_terminal(state) {
        //    if self.is_goal(state) {
        //        return (None, 0.0);
        //    } else {
        //        //let delta = (state.1 - self.end_node).abs() as f64 / (self.num_nodes as f64).sqrt();
        //        //return (None, -(0.1f64 * delta).powi(3));
        //        return (None, 0.0);
        //    }
        //}
        
        if self.is_terminal(state) {
            let delta = (state.1 - self.end_node).abs() as f64 / (self.num_nodes as f64).sqrt();
            return (None, -(delta).powi(2));
        } else if self.is_goal(state) {
            let delta_t = (state.0 - self.t) as f64 / (self.t as f64).sqrt();
            return (None, -(delta_t).powi(2));
        }

        let action = action.as_ref().unwrap();
        let q_curr = state.1;
        let q_next = state.1 + action;

        let q = (q_curr + q_next) as f64 / 2f64;
        let dq = (q_next - q_curr) as f64;

        let reward = self.reward(q, dq);

        (
            Some((state.0 + 1, q_next)),
            reward,
        )
    }

    fn available_actions(&self, state: &S) -> Vec<i64> {
        let part_1 = (0 - state.1 .. 0).collect::<Vec<i64>>();
        let part_2 = (1 .. self.num_nodes as i64 - state.1).collect::<Vec<i64>>();
        part_1.into_iter().chain(part_2).collect()
    }
}
