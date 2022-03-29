use core::arch::x86_64::*;
use rayon::prelude::*;

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
        .for_each(|((w, x), y)| unsafe {
            let xv = _mm256_loadu_ps(x.as_ptr());
            let yv = _mm256_loadu_ps(y.as_ptr());
            _mm256_storeu_ps(w.as_mut_ptr(), _mm256_sub_ps(xv, yv));
        });

    for i in (w.len() - (w.len() % 8))..w.len() {
        w[i] = x[i] - y[i]
    }
}

pub fn xxpby(x: &mut [f32], b: f32, y: &[f32]) {
    let beta = unsafe { _mm256_set1_ps(b) };
    x.par_chunks_exact_mut(8)
        .zip(y.par_chunks(8))
        .for_each(|(x, y)| unsafe {
            let xv = _mm256_loadu_ps(x.as_ptr());
            let yv = _mm256_loadu_ps(y.as_ptr());
            _mm256_storeu_ps(x.as_mut_ptr(), _mm256_fmadd_ps(yv, beta, xv));
        });

    for i in (x.len() - (x.len() % 8))..x.len() {
        x[i] += b * y[i];
    }
}

pub fn yxpby(x: &[f32], b: f32, y: &mut [f32]) {
    y.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(y, x)| *y = x + b * *y);

    let beta = unsafe { _mm256_set1_ps(b) };
    y.par_chunks_exact_mut(8)
        .zip(x.par_chunks(8))
        .for_each(|(y, x)| unsafe {
            let xv = _mm256_loadu_ps(x.as_ptr());
            let yv = _mm256_loadu_ps(y.as_ptr());
            _mm256_storeu_ps(y.as_mut_ptr(), _mm256_fmadd_ps(yv, beta, xv));
        });

    for i in (x.len() - (x.len() % 8))..x.len() {
        y[i] += x[i] + b * y[i];
    }
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

unsafe fn sum_m256(v: __m256) -> f32 {
    let sum4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_movehdup_ps(sum2));
    _mm_cvtss_f32(sum1)
}
