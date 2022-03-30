use std::time::{Duration, Instant};

use crate::{kernels::*, mesh::Mesh};

pub struct RunInfo {
    pub t_total: Duration,
    pub t_ddot: Duration,
    pub t_waxpby: Duration,
    pub t_sparsemv: Duration,
    pub iters: u32,
    pub normr: f32,
}

pub const MAX_ITER: u32 = 150;
pub const TOLERANCE: f32 = 0.0;

macro_rules! time_it {
    ($fn: expr) => {{
        let t0 = Instant::now();
        $fn;
        Instant::now() - t0
    }};
}

pub fn run(a: &Mesh, x: &mut [f32], b: &[f32]) -> RunInfo {
    let t_begin = Instant::now();
    let mut t_ddot = Duration::ZERO;
    let mut t_waxpby = Duration::ZERO;
    let mut t_sparsemv = Duration::ZERO;

    let nrow = a.nrow;

    let mut r = vec![0.0; nrow];
    let mut p = vec![0.0; nrow];
    let mut ap = vec![0.0; nrow];

    let mut normr: f32;
    let mut rtrans: f32;
    let mut oldrtrans: f32;

    t_waxpby += time_it!(p.copy_from_slice(x));

    t_sparsemv += time_it!(sparsemv(a, &p, &mut ap));

    t_waxpby += time_it!(wxmy(b, &ap, &mut r));

    t_ddot += time_it!(rtrans = ddot_same(&r));

    #[cfg(feature = "verbose")]
    {
        normr = f32::sqrt(rtrans);
        println!("Initial residual = {:e}", normr);
    }

    let mut k = 1;

    //if k ==1
    t_waxpby += time_it!(p.copy_from_slice(&r));

    //after the if/else

    normr = f32::sqrt(rtrans);

    #[cfg(feature = "verbose")]
    println!("Iteration = {} Residual = {:e}", k, normr);

    t_sparsemv += time_it!(sparsemv(a, &p, &mut ap));

    let mut alpha;
    t_ddot += time_it!(alpha = ddot(&p, &ap));

    alpha = rtrans / alpha;

    t_waxpby += time_it!(xxpby(x, alpha, &p));
    t_waxpby += time_it!(xxpby(&mut r, -alpha, &ap));

    while k < MAX_ITER && normr > TOLERANCE {
        oldrtrans = rtrans;

        t_ddot += time_it!(rtrans = ddot_same(&r));

        let beta = rtrans / oldrtrans;

        t_waxpby += time_it!(yxpby(&r, beta, &mut p));

        normr = f32::sqrt(rtrans);

        #[cfg(feature = "verbose")]
        println!("Iteration = {} Residual = {}", k, normr);

        t_sparsemv += time_it!(sparsemv(a, &p, &mut ap));

        let mut alpha;
        t_ddot += time_it!(alpha = ddot(&p, &ap));

        alpha = rtrans / alpha;

        t_waxpby += time_it!(xxpby(x, alpha, &p));
        t_waxpby += time_it!(xxpby(&mut r, -alpha, &ap));

        k += 1;
    }

    RunInfo {
        t_total: Instant::now() - t_begin,
        t_ddot,
        t_waxpby,
        t_sparsemv,
        iters: k,
        normr,
    }
}
