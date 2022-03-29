use rayon::prelude::*;
use std::simd::Simd;

pub fn ddot(x: &[f32], y: &[f32]) -> f32 {
    x.par_iter().zip(y.par_iter()).map(|(x, y)| x * y).sum()
}

pub fn ddot_same(x: &[f32]) -> f32 {
    x.par_iter().map(|x| x * x).sum()
}

pub fn wxmy(x: &[f32], y: &[f32], w: &mut [f32]) {
    w.par_chunks_exact_mut(8)
        .zip(x.par_chunks_exact(8))
        .zip(y.par_chunks_exact(8))
        .for_each(|((w, x), y)| {
            let xv: Simd<f32, 8> = Simd::from_slice(x);
            let yv: Simd<f32, 8> = Simd::from_slice(y);
            w.copy_from_slice((xv - yv).as_array())
        });

    for i in (w.len() - (w.len() % 8))..w.len() {
        w[i] = x[i] - y[i]
    }
}

pub fn xxpby(x: &mut [f32], b: f32, y: &[f32]) {
    x.par_iter_mut()
        .zip(y.par_iter())
        .for_each(|(x, y)| *x += b * y);
}

pub fn yxpby(x: &[f32], b: f32, y: &mut [f32]) {
    y.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(y, x)| *y = x + b * *y);
}

pub fn sparsemv(a: &crate::Mesh, x: &[f32], y: &mut [f32]) {
    y.par_iter_mut()
        .zip(a.nnz.par_iter())
        .zip(a.values.par_iter())
        .zip(a.idx.par_iter())
        .for_each(|(((y, nnz), values), idx)| {
            let mut sum = 0.0;
            for j in 0..*nnz {
                sum += values[j] * x[idx[j]]
            }
            *y = sum
        });
}
