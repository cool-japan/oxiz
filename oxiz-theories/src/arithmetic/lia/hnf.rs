//\! Hermite Normal Form for integer linear systems
/// Hermite Normal Form (HNF) transformation
///
/// Used for analyzing integer linear systems.
#[derive(Debug)]
pub struct HermiteNormalForm {
    /// The HNF matrix
    pub matrix: Vec<Vec<i64>>,
    /// The transformation matrix
    pub transform: Vec<Vec<i64>>,
}

impl HermiteNormalForm {
    /// Compute HNF of a matrix using column operations
    ///
    /// The Hermite Normal Form is useful for analyzing integer linear systems.
    /// For a matrix A, we find H and U such that A*U = H, where H is in HNF.
    pub fn compute(matrix: &[Vec<i64>]) -> Self {
        if matrix.is_empty() || matrix[0].is_empty() {
            return Self {
                matrix: Vec::new(),
                transform: Vec::new(),
            };
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        // Copy the input matrix
        let mut h: Vec<Vec<i64>> = matrix.to_vec();

        // Initialize transformation matrix as identity
        let mut u = vec![vec![0i64; cols]; cols];
        for (i, row) in u.iter_mut().enumerate().take(cols) {
            row[i] = 1;
        }

        // Column operations to achieve HNF
        for col in 0..cols.min(rows) {
            // Find pivot row (first non-zero entry in this column)
            let mut pivot_row = None;
            for (idx, h_row) in h.iter().enumerate().skip(col).take(rows - col) {
                if h_row[col] != 0 {
                    pivot_row = Some(col + idx);
                    break;
                }
            }

            if pivot_row.is_none() {
                continue;
            }

            let pivot_row = pivot_row.expect("pivot row must exist after is_none check");

            // Make pivot positive
            if h[pivot_row][col] < 0 {
                for c in 0..cols {
                    h[pivot_row][c] = -h[pivot_row][c];
                    u[col][c] = -u[col][c];
                }
            }

            // Reduce other entries in this row
            for other_col in (col + 1)..cols {
                if h[pivot_row][other_col] != 0 {
                    let pivot_val = h[pivot_row][col];
                    let other_val = h[pivot_row][other_col];

                    // Use extended GCD to reduce
                    let quotient = other_val / pivot_val;

                    // Column operation: col[other] -= quotient * col[col]
                    for h_row in h.iter_mut().take(rows) {
                        h_row[other_col] -= quotient * h_row[col];
                    }
                    // Copy u[col] to avoid borrow conflict
                    let u_col_copy: Vec<_> = u[col].clone();
                    for (u_other, u_col) in u[other_col].iter_mut().zip(u_col_copy.iter()) {
                        *u_other -= quotient * u_col;
                    }
                }
            }
        }

        Self {
            matrix: h,
            transform: u,
        }
    }
}

