use core::arch::x86_64::*;
use rayon::prelude::*;

pub fn ddot(x: &[f32], y: &[f32]) -> f32 {
    let s = x
        .par_chunks_exact(8)
        .zip(y.par_chunks_exact(8))
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
        );
    let mut sum = sum_m256(s);
    for i in (x.len() - (x.len() % 8))..x.len() {
        sum += x[i] * y[i]
    }

    sum
}

pub fn ddot_same(x: &[f32]) -> f32 {
    let s = x
        .par_chunks_exact(8)
        .fold(
            || unsafe { _mm256_setzero_ps() },
            |sum, x| unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr());
                _mm256_fmadd_ps(xv, xv, sum)
            },
        )
        .reduce(
            || unsafe { _mm256_setzero_ps() },
            |v1, v2| unsafe { _mm256_add_ps(v1, v2) },
        );
    let mut sum = sum_m256(s);
    for xi in x.iter().skip(x.len() - (x.len() % 8)) {
        sum += xi
    }
    sum
}

pub fn wxmy(x: &[f32], y: &[f32], w: &mut [f32]) {
    w.par_iter_mut()
        .zip(x.par_iter().zip(y.par_iter()))
        .for_each(|(w, (x, y))| *w = x - y);
}

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
        .zip(a.nnz.par_iter())
        .zip(a.values.par_iter())
        .zip(a.idx.par_iter())
        .for_each(|(((y, nnz), values), idx)| unsafe {
            let mut sum = _mm256_setzero_ps();

            for i in (0..(*nnz / 8)).map(|i| i * 8) {
                let vidx = _mm256_loadu_si256(idx.as_ptr().add(i) as *const __m256i);
                let vx = _mm256_i32gather_ps::<4>(x.as_ptr(), vidx);
                let vrow = _mm256_loadu_ps(values.as_ptr().add(i));
                sum = _mm256_fmadd_ps(vx, vrow, sum);
            }
            let mut result = sum_m256(sum);

            //add remainder
            for i in (values.len() - values.len() % 8)..values.len() {
                result += values[i] * x[idx[i] as usize]
            }
            *y = result
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
