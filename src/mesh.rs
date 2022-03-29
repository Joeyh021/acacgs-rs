pub struct Mesh {
    pub nrow: usize,
    pub nnz: Vec<usize>,
    pub values: Vec<Vec<f32>>,
    pub idx: Vec<Vec<u32>>,
}

impl Mesh {
    pub fn generate((nx, ny, nz): (isize, isize, isize)) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0);
        let nrow = (nx * ny * nz) as usize;

        let mut nnz: Vec<usize> = vec![0; nrow];
        let mut values: Vec<Vec<f32>> = vec![vec![]; nrow];
        let mut idx: Vec<Vec<usize>> = vec![vec![]; nrow];

        for iz in 0..nz as isize {
            for iy in 0..ny as isize {
                for ix in 0..nx as isize {
                    let current_row = TryInto::<usize>::try_into(iz * nx * ny + iy * nx + ix)
                        .expect("Index error: row was negative");
                    let mut nnz_in_row = 0;
                    for sz in -1..=1 {
                        for sy in -1..=1 {
                            for sx in -1..=1 {
                                let current_col =
                                    current_row as isize + sz * nx * ny + sy * nx + sx;
                                if (ix + sx >= 0)
                                    && (ix + sx < nx)
                                    && (iy + sy >= 0)
                                    && (iy + sy < ny)
                                    && (current_col >= 0 && current_col < nrow as isize)
                                {
                                    if current_row == current_col as usize {
                                        values[current_row].push(27.0);
                                    } else {
                                        values[current_row].push(-1.0);
                                    }
                                    idx[current_row].push(current_col as usize);
                                    nnz_in_row += 1;
                                }
                            }
                        }
                    }
                    nnz[current_row as usize] = nnz_in_row;
                }
            }
        }
        Self {
            nrow,
            nnz,
            values,
            idx: idx
                .into_iter()
                .map(|v| v.into_iter().map(|i| i as u32).collect())
                .collect(),
        }
    }
}
