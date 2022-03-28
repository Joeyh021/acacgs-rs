pub fn ddot(x: &[f32], y: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * y[i];
    }
    sum
}

pub fn ddot_same(x: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in x {
        sum += i * i
    }
    sum
}

pub fn wxmy(x: &[f32], y: &[f32], w: &mut [f32]) {
    for i in 0..w.len() {
        w[i] = x[i] - y[i];
    }
}

pub fn xxpby(x: &mut [f32], b: f32, y: &[f32]) {
    for i in 0..x.len() {
        x[i] += b * y[i];
    }
}

pub fn yxpby(x: &[f32], b: f32, y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] = x[i] + b * y[i];
    }
}

pub fn sparsemv(a: &crate::Mesh, x: &[f32], y: &mut [f32]) {
    for i in 0..a.nrow {
        let mut sum = 0.0;
        let nnz = a.nnz[i];
        let values = &a.values[i];
        let idx = &a.idx[i];

        for j in 0..nnz {
            sum += values[j] * x[idx[j]];
        }
        y[i] = sum;
    }
}
