use crate::lagrangian::Lagrangian;
//use forger::env::Env;

//type S = (usize, (i64, i64));
type Q = (f64, f64);

#[allow(dead_code)]
pub struct Lattice2D<L: Lagrangian> {
    init_node: (i64, i64),
    end_node: (i64, i64),
    t: usize,
    lagrangian: L,
    ground_state: Q,
}

impl<L: Lagrangian<Q = Q>> Lattice2D<L> {
    pub fn new(
        init_node: (i64, i64),
        end_node: (i64, i64),
        t: usize,
        lagrangian: L,
        ground_state: (f64, f64),
    ) -> Self {
        Self {
            init_node,
            end_node,
            t,
            lagrangian,
            ground_state,
        }
    }

    pub fn lagrangian(&self) -> &L {
        &self.lagrangian
    }

    pub fn l_min(&self) -> f64 {
        self.lagrangian.calc(&self.ground_state, &(1f64, 0f64))
    }

    pub fn l_amplify(&self, q: &Q, dq: &Q) -> f64 {
        let l = self.lagrangian.calc(q, dq);
        (l - self.l_min()).powi(2)
    }
}
