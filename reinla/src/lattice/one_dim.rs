use crate::lagrangian::Lagrangian;
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

    //pub fn x_min(&self) -> f64 {
    //    self.init_node as f64
    //}

    //pub fn x_max(&self) -> f64 {
    //    self.end_node as f64
    //}

    //pub fn v_min(&self) -> f64 {
    //    1f64
    //}

    //pub fn v_max(&self) -> f64 {
    //    (self.end_node - self.init_node) as f64
    //}

    //pub fn l_min(&self) -> f64 {
    //    self.lagrangian.calc(&self.x_max(), &self.v_min())
    //}

    //pub fn l_max(&self) -> f64 {
    //    self.lagrangian.calc(&self.x_min(), &self.v_max())
    //}

    //pub fn l_amplify(&self, q: f64, dq: f64) -> f64 {
    //    let l = self.lagrangian.calc(&q, &dq);
    //    (l - self.l_min()).powi(2)
    //}

    pub fn set_l_min_max(&mut self, c_min: f64, c_max: f64) {
        self._l_min_max = Some((c_min, c_max));
    }

    pub fn reward(&self, q: f64, dq: f64) -> f64 {
        let l = self.lagrangian.calc(&q, &dq);
        let c = self._l_min_max;
        match c {
            Some((c_min, c_max)) => {
                //let c_half = (2f64 * c_min + c_max) / 3f64;
                //let l_minmax = 6f64 * (l - c_half) / (c_max - c_min);
                //-(elu(l_minmax) + 1f64).powi(4) + 1f64
                let l_minmax = (l - c_min) / (c_max - c_min);
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
        if self.is_terminal(state) {
            if self.is_goal(state) {
                return (None, 10.0);
            } else {
                return (None, 0.0);
            }
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
