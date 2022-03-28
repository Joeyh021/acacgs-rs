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

    let t_0 = Instant::now();
    p.copy_from_slice(x);
    t_waxpby += Instant::now() - t_0;

    let t_0 = Instant::now();
    sparsemv(a, &p, &mut ap);
    t_sparsemv += Instant::now() - t_0;

    let t_0 = Instant::now();
    wxmy(b, &ap, &mut r);
    t_waxpby += Instant::now() - t_0;

    let t_0 = Instant::now();
    rtrans = ddot_same(&r);
    t_ddot += Instant::now() - t_0;

    #[cfg(feature = "verbose")]
    {
        normr = f32::sqrt(rtrans);
        println!("Initial residual = {}", normr);
    }

    let mut k = 1;

    //if k ==1
    let t_0 = Instant::now();
    p.copy_from_slice(&r);
    t_waxpby += Instant::now() - t_0;

    //after the if/else

    normr = f32::sqrt(rtrans);

    #[cfg(feature = "verbose")]
    println!("Iteration = {} Residual = {}", k, normr);

    let t_0 = Instant::now();
    sparsemv(a, &p, &mut ap);
    t_sparsemv += Instant::now() - t_0;

    let t_0 = Instant::now();
    let mut alpha = ddot(&p, &ap);
    t_ddot += Instant::now() - t_0;

    alpha = rtrans / alpha;

    let t_0 = Instant::now();
    xxpby(x, alpha, &p);
    xxpby(&mut r, -alpha, &ap);
    t_waxpby += Instant::now() - t_0;

    while k < MAX_ITER && normr > TOLERANCE {
        oldrtrans = rtrans;

        let t_0 = Instant::now();
        rtrans = ddot_same(&r);
        t_ddot += Instant::now() - t_0;

        let beta = rtrans / oldrtrans;

        let t_0 = Instant::now();
        yxpby(&r, beta, &mut p);
        t_waxpby += Instant::now() - t_0;

        normr = f32::sqrt(rtrans);

        #[cfg(feature = "verbose")]
        println!("Iteration = {} Residual = {}", k, normr);

        let t_0 = Instant::now();
        sparsemv(a, &p, &mut ap);
        t_sparsemv += Instant::now() - t_0;

        let t_0 = Instant::now();
        let mut alpha = ddot(&p, &ap);
        t_ddot += Instant::now() - t_0;

        alpha = rtrans / alpha;

        let t_0 = Instant::now();
        xxpby(x, alpha, &p);
        xxpby(&mut r, -alpha, &ap);
        t_waxpby += Instant::now() - t_0;

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
