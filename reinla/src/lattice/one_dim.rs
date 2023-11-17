use crate::lagrangian::Lagrangian;
use crate::util::comb;
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

    #[allow(non_snake_case)]
    pub fn L(&self, q: f64, dq: f64) -> f64 {
        self.lagrangian.calc(&q, &dq)
    }

    pub fn get_init_node(&self) -> i64 {
        self.init_node
    }

    pub fn get_end_node(&self) -> i64 {
        self.end_node
    }

    pub fn get_t(&self) -> usize {
        self.t
    }

    pub fn get_num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn set_l_min_max(&mut self, l_min: f64, l_max: f64) {
        self._l_min_max = Some((l_min, l_max));
    }

    pub fn reset_l_min_max(&mut self) {
        self._l_min_max = None;
    }

    pub fn reward(&self, q: f64, dq: f64) -> f64 {
        let l = self.L(q, dq);
        let c = self._l_min_max;
        match c {
            Some((l_min, l_max)) => {
                let l_minmax = (l - l_min) / (l_max - l_min);
                l_minmax.powi(5)
            }
            None => {
                l
            }
        }
    }

    pub fn brute_force(&self) -> Vec<i64> {
        let mut min_val = std::f64::MAX;
        let mut min_path = vec![0i64; self.t+1];

        min_path[0] = self.init_node;
        min_path[self.t] = self.end_node;

        for q_vec in comb(self.num_nodes as i64 - 2, self.t - 1).into_iter() {
            let mut action = 0f64;

            let v0 = (q_vec[0] - self.init_node) as f64;
            let x0 = (q_vec[0] + self.init_node) as f64 / 2f64;
            action += self.L(x0, v0);

            for i in (1 .. self.t - 1).rev() {
                let vi = (q_vec[i] - q_vec[i-1]) as f64;
                let xi = (q_vec[i] + q_vec[i-1]) as f64 / 2f64;
                action += self.L(xi, vi);
            }
            let qm = q_vec[self.t-2];
            let vm = (self.end_node - qm) as f64;
            let xm = (self.end_node + qm) as f64 / 2f64;
            action += self.L(xm, vm);

            if action < min_val {
                for j in 1 .. self.t {
                    min_path[j] = q_vec[j-1];
                }
                min_val = action;
            }
        }

        min_path
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
        if self.is_terminal(state) {
            let q = (state.1 + self.end_node) as f64 / 2f64;
            let dq = (self.end_node - state.1) as f64;
            
            let reward = self.reward(q, dq);

            return (None, reward);
        }

        let action = action.as_ref().unwrap();
        let q_curr = state.1;
        let q_next = *action;

        let q = (q_curr + q_next) as f64 / 2f64;
        let dq = (q_next - q_curr) as f64;

        let reward = self.reward(q, dq);

        (
            Some((state.0 + 1, q_next)),
            reward,
        )
    }

    fn available_actions(&self, state: &S) -> Vec<i64> {
        if self.is_terminal(state) {
            return vec![self.end_node];
        }
        // Monotonic increasing
        (state.1 .. self.end_node).collect()
    }
}
