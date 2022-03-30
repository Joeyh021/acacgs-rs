#![allow(clippy::many_single_char_names)]
#![feature(portable_simd)]
mod conjugate_gradient;
mod kernels;
mod mesh;
use std::env::args;

use mesh::Mesh;
fn main() {
    if args().len() != 4 {
        eprintln!("Please specify size: x y z");
        return;
    }

    let size = args()
        .skip(1)
        .map(|s| s.parse::<isize>().expect("bad args"))
        .collect::<Vec<_>>();
    let size = (size[0], size[1], size[2]);

    let a = Mesh::generate(size);
    let nrow = a.nrow;
    let mut x: Vec<f32> = vec![0.0; nrow];
    let x_exact: Vec<f32> = vec![1.0; nrow];
    let b: Vec<f32> = a.nnz.iter().map(|i| 27.0 - (*i as f32 - 1.0)).collect();

    let results = conjugate_gradient::run(&a, &mut x, &b);

    let ops_ddot = results.iters * 4 * nrow as u32;
    let ops_waxpby = results.iters * 6 * nrow as u32;
    let ops_sparsemv = results.iters * 2 * nrow as u32;
    let ops = ops_ddot + ops_sparsemv + ops_waxpby;

    println!("===== Final Statistics =====");
    println!("Executable name:      {}", args().next().unwrap());
    println!("Dimensions:           {} {} {}", size.0, size.1, size.2);
    println!("Number of Iterations: {}", results.iters);
    println!("Final Residual:       {:e}", results.normr);
    println!("=== Time ===");
    println!("Total:           {:?}", results.t_total);
    println!("ddot Kernel:     {:?}", results.t_ddot);
    println!("waxpby Kernel:     {:?}", results.t_waxpby);
    println!("sparsemv Kernel:     {:?}", results.t_sparsemv);
    println!("=== MFLOP/s ===");
    println!("Total:           {:} floating point operations", ops);
    println!("ddot Kernel:     {:} floating point operations", ops_ddot);
    println!(
        "waxpby Kernel:     {:} floating point operations",
        ops_waxpby
    );
    println!(
        "sparsemv Kernel:     {:} floating point operations",
        ops_sparsemv
    );
    println!("=== Time ===");
    println!(
        "Total:           {:e} MFLOP/s",
        ops as f64 / results.t_total.as_secs_f64() / 1.0e6
    );
    println!(
        "ddot Kernel:     {:e} MFLOP/s",
        ops_ddot as f64 / results.t_ddot.as_secs_f64() / 1.0e6
    );
    println!(
        "waxpby Kernel:     {:e} MFLOP/s",
        ops_waxpby as f64 / results.t_waxpby.as_secs_f64() / 1.0e6
    );
    println!(
        "sparsemv Kernel:     {:e} MFLOP/s",
        ops_sparsemv as f64 / results.t_sparsemv.as_secs_f64() / 1.0e6
    );
    println!();
    println!(
        "Difference between computed and exact = {:e}",
        compute_residual(&x, &x_exact)
    );
}

fn compute_residual(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .fold(0.0, |local_residual, (i, j)| {
            f32::abs(i - j).max(local_residual)
        })
}
