use core::arch::x86_64::*;
use rayon::prelude::*;

pub fn ddot(x: &[f32], y: &[f32]) -> f32 {
    let x = x.par_chunks_exact(8);
    let y = y.par_chunks_exact(8);
    x.remainder()
        .iter()
        .zip(y.remainder().iter())
        .fold(0.0, |sum, (x, y)| sum + x * y)
        + sum_m256(
            x.zip(y)
                .fold(
                    || unsafe { _mm256_setzero_ps() },
                    |sum, (x, y)| unsafe {
                        let xv = _mm256_loadu_ps(x.as_ptr());
                        let yv = _mm256_loadu_ps(y.as_ptr());
                        _mm256_fmadd_ps(xv, yv, sum)
                    },
                )
                .reduce(
                    || unsafe { _mm256_setzero_ps() },
                    |v1, v2| unsafe { _mm256_add_ps(v1, v2) },
                ),
        )
}

pub fn ddot_same(x: &[f32]) -> f32 {
    let x = x.par_chunks_exact(8);
    x.remainder().iter().fold(0.0, |sum, x| sum + x * x)
        + sum_m256(
            x.fold(
                || unsafe { _mm256_setzero_ps() },
                |sum, x| unsafe {
                    let xv = _mm256_loadu_ps(x.as_ptr());
                    _mm256_fmadd_ps(xv, xv, sum)
                },
            )
            .reduce(
                || unsafe { _mm256_setzero_ps() },
                |v1, v2| unsafe { _mm256_add_ps(v1, v2) },
            ),
        )
}

//not vectorised because the semantics are easy enough that llvm does it for us
pub fn wxmy(x: &[f32], y: &[f32], w: &mut [f32]) {
    w.par_iter_mut()
        .zip(x.par_iter().zip(y.par_iter()))
        .for_each(|(w, (x, y))| *w = x - y);
}

//have used explicit scalar fma tho to coerce it into playing nice
pub fn xxpby(x: &mut [f32], b: f32, y: &[f32]) {
    x.par_iter_mut()
        .zip(y.par_iter())
        .for_each(|(x, y)| *x = f32::mul_add(b, *y, *x));
}

pub fn yxpby(x: &[f32], b: f32, y: &mut [f32]) {
    y.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(y, x)| *y = f32::mul_add(b, *y, *x));
}

pub fn sparsemv(a: &crate::Mesh, x: &[f32], y: &mut [f32]) {
    y.par_iter_mut()
        .zip(a.values.par_iter())
        .zip(a.idx.par_iter())
        .for_each(|((y, values), idx)| {
            let values = values.chunks_exact(8);
            let idx = idx.chunks_exact(8);
            *y = values
                .remainder()
                .iter()
                .zip(idx.remainder().iter())
                .fold(0.0, |sum, (v, i)| sum + v * x[*i as usize])
                + sum_m256(values.zip(idx).fold(
                    unsafe { _mm256_setzero_ps() },
                    |sum, (val, i)| unsafe {
                        let vidx = _mm256_loadu_si256(i.as_ptr() as *const __m256i);
                        let vx = _mm256_i32gather_ps::<4>(x.as_ptr(), vidx);
                        let vrow = _mm256_loadu_ps(val.as_ptr());
                        _mm256_fmadd_ps(vrow, vx, sum)
                    },
                ));
        });
}

fn sum_m256(v: __m256) -> f32 {
    unsafe {
        let sum4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_movehdup_ps(sum2));
        _mm_cvtss_f32(sum1)
    }
}
