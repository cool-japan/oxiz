//! Dense matrix operations for linear programming and linear algebra.
//!
//! This module provides a dense matrix implementation with operations needed
//! for linear programming solvers, including Gaussian elimination, LU decomposition,
//! and linear system solving.

use num_rational::Rational64;
use num_traits::Signed;
use std::fmt;

/// A dense matrix stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Rational64>,
}

impl Matrix {
    /// Creates a new matrix with the given dimensions, initialized to zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Rational64::from(0); rows * cols],
        }
    }

    /// Creates a new identity matrix of the given size.
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, Rational64::from(1));
        }
        matrix
    }

    /// Creates a new matrix from a Vec of row vectors.
    pub fn from_rows(rows: Vec<Vec<Rational64>>) -> Self {
        if rows.is_empty() {
            return Self::zeros(0, 0);
        }
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in rows {
            assert_eq!(row.len(), ncols, "All rows must have the same length");
            data.extend(row);
        }
        Self {
            rows: nrows,
            cols: ncols,
            data,
        }
    }

    /// Creates a new matrix from a flat array in row-major order.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<Rational64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Data size must match dimensions");
        Self { rows, cols, data }
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Gets the element at position (row, col).
    pub fn get(&self, row: usize, col: usize) -> Rational64 {
        assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col]
    }

    /// Sets the element at position (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: Rational64) {
        assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    /// Gets a reference to a row.
    pub fn row(&self, row: usize) -> &[Rational64] {
        assert!(row < self.rows);
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Swaps two rows.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        assert!(i < self.rows && j < self.rows);
        if i == j {
            return;
        }
        for k in 0..self.cols {
            let idx_i = i * self.cols + k;
            let idx_j = j * self.cols + k;
            self.data.swap(idx_i, idx_j);
        }
    }

    /// Adds a scalar multiple of one row to another: row_i += scalar * row_j
    pub fn add_row_multiple(&mut self, i: usize, j: usize, scalar: Rational64) {
        assert!(i < self.rows && j < self.rows);
        if scalar == Rational64::from(0) {
            return;
        }
        for k in 0..self.cols {
            let idx_i = i * self.cols + k;
            let idx_j = j * self.cols + k;
            let val_j = self.data[idx_j];
            self.data[idx_i] += scalar * val_j;
        }
    }

    /// Multiplies a row by a scalar.
    pub fn scale_row(&mut self, row: usize, scalar: Rational64) {
        assert!(row < self.rows);
        for k in 0..self.cols {
            let idx = row * self.cols + k;
            self.data[idx] *= scalar;
        }
    }

    /// Matrix-vector multiplication: returns A * v
    #[allow(clippy::needless_range_loop)]
    pub fn mul_vec(&self, v: &[Rational64]) -> Vec<Rational64> {
        assert_eq!(
            v.len(),
            self.cols,
            "Vector dimension must match matrix columns"
        );
        let mut result = vec![Rational64::from(0); self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }

    /// Matrix-matrix multiplication: returns A * B
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions must be compatible for multiplication"
        );
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Rational64::from(0);
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Performs Gaussian elimination with partial pivoting.
    /// Returns the row-echelon form and the rank of the matrix.
    pub fn gaussian_elimination(&self) -> (Matrix, usize) {
        let mut result = self.clone();
        let mut rank = 0;

        for col in 0..self.cols.min(self.rows) {
            // Find pivot
            let mut pivot_row = None;
            for row in rank..self.rows {
                if result.get(row, col) != Rational64::from(0) {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => continue, // Column is all zeros below current rank
            };

            // Swap rows if needed
            if pivot_row != rank {
                result.swap_rows(rank, pivot_row);
            }

            // Eliminate below
            let pivot = result.get(rank, col);
            for row in rank + 1..self.rows {
                let factor = result.get(row, col) / pivot;
                result.add_row_multiple(row, rank, -factor);
            }

            rank += 1;
        }

        (result, rank)
    }

    /// Performs LU decomposition with partial pivoting.
    /// Returns (L, U, P) where P is a permutation vector.
    pub fn lu_decomposition(&self) -> Option<(Matrix, Matrix, Vec<usize>)> {
        if self.rows != self.cols {
            return None; // LU decomposition requires square matrix
        }
        let n = self.rows;

        let mut l = Matrix::zeros(n, n);
        let mut u = self.clone();
        let mut p: Vec<usize> = (0..n).collect();

        for i in 0..n {
            // Partial pivoting: find row with largest element in column i
            let mut max_row = i;
            let mut max_val = u.get(i, i).abs();
            for k in i + 1..n {
                let val = u.get(k, i).abs();
                if val > max_val {
                    max_val = val;
                    max_row = k;
                }
            }

            // Swap rows in U and permutation vector
            if max_row != i {
                u.swap_rows(i, max_row);
                p.swap(i, max_row);
                // Also swap already computed parts of L
                for k in 0..i {
                    let tmp = l.get(i, k);
                    l.set(i, k, l.get(max_row, k));
                    l.set(max_row, k, tmp);
                }
            }

            let pivot = u.get(i, i);
            if pivot == Rational64::from(0) {
                return None; // Matrix is singular
            }

            l.set(i, i, Rational64::from(1));

            for k in i + 1..n {
                let factor = u.get(k, i) / pivot;
                l.set(k, i, factor);
                for j in i..n {
                    let val = u.get(k, j) - factor * u.get(i, j);
                    u.set(k, j, val);
                }
            }
        }

        Some((l, u, p))
    }

    /// Solves a linear system Ax = b using Gaussian elimination with back substitution.
    /// Returns None if the system has no unique solution.
    #[allow(clippy::needless_range_loop)]
    pub fn solve(&self, b: &[Rational64]) -> Option<Vec<Rational64>> {
        assert_eq!(
            b.len(),
            self.rows,
            "Right-hand side dimension must match matrix rows"
        );

        if self.rows != self.cols {
            return None; // Only solve square systems for now
        }

        // Augment matrix with b
        let mut aug = Matrix::zeros(self.rows, self.cols + 1);
        for i in 0..self.rows {
            for j in 0..self.cols {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, self.cols, b[i]);
        }

        // Forward elimination
        for col in 0..self.cols {
            // Find pivot
            let mut pivot_row = None;
            for row in col..self.rows {
                if aug.get(row, col) != Rational64::from(0) {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = pivot_row?; // No unique solution if None

            if pivot_row != col {
                aug.swap_rows(col, pivot_row);
            }

            let pivot = aug.get(col, col);
            for row in col + 1..self.rows {
                let factor = aug.get(row, col) / pivot;
                aug.add_row_multiple(row, col, -factor);
            }
        }

        // Back substitution
        let mut x = vec![Rational64::from(0); self.cols];
        for i in (0..self.rows).rev() {
            let mut sum = aug.get(i, self.cols);
            for j in i + 1..self.cols {
                sum -= aug.get(i, j) * x[j];
            }
            let pivot = aug.get(i, i);
            if pivot == Rational64::from(0) {
                return None; // Singular matrix
            }
            x[i] = sum / pivot;
        }

        Some(x)
    }

    /// Computes the determinant using LU decomposition.
    pub fn determinant(&self) -> Option<Rational64> {
        if self.rows != self.cols {
            return None;
        }

        let (_, u, p) = self.lu_decomposition()?;

        // Determinant of U is the product of diagonal elements
        let mut det = Rational64::from(1);
        for i in 0..self.rows {
            det *= u.get(i, i);
        }

        // Count swaps in permutation to get sign
        let mut swaps = 0;
        let mut visited = vec![false; p.len()];
        for i in 0..p.len() {
            if visited[i] {
                continue;
            }
            let mut j = i;
            let mut cycle_len = 0;
            while !visited[j] {
                visited[j] = true;
                j = p[j];
                cycle_len += 1;
            }
            if cycle_len > 1 {
                swaps += cycle_len - 1;
            }
        }

        if swaps % 2 == 1 {
            det = -det;
        }

        Some(det)
    }

    /// Performs QR decomposition using the modified Gram-Schmidt process.
    /// Returns (Q, R) where Q has orthogonal columns and R is upper triangular such that QR = A.
    ///
    /// # Note
    /// Since we're working with exact rational arithmetic and can't compute square roots,
    /// Q will not be orthonormal in the traditional sense. Instead, Q contains unnormalized
    /// orthogonal vectors, and R is adjusted accordingly so that QR = A holds exactly.
    ///
    /// Specifically: `R[j,j] = 1` and the j-th column of Q has squared norm stored implicitly.
    pub fn qr_decomposition(&self) -> Option<(Matrix, Matrix)> {
        let m = self.rows;
        let n = self.cols;

        if m < n {
            return None; // Need m >= n for standard QR
        }

        let mut q = Matrix::zeros(m, n);
        let mut r = Matrix::zeros(n, n);

        // Classical Gram-Schmidt: a_j = sum_{k=0}^j r[k,j] * q_k
        for j in 0..n {
            // Start with column j of A
            for i in 0..m {
                q.set(i, j, self.get(i, j));
            }

            // Subtract projections onto previous orthogonal vectors
            for k in 0..j {
                // Compute projection coefficient: <q_k, a_j> / <q_k, q_k>
                let mut dot_qk_aj = Rational64::from(0);
                let mut dot_qk_qk = Rational64::from(0);

                for i in 0..m {
                    dot_qk_aj += q.get(i, k) * self.get(i, j);
                    dot_qk_qk += q.get(i, k) * q.get(i, k);
                }

                if dot_qk_qk == Rational64::from(0) {
                    return None;
                }

                let proj_coeff = dot_qk_aj / dot_qk_qk;
                r.set(k, j, proj_coeff);

                // Subtract projection: q_j -= proj_coeff * q_k
                for i in 0..m {
                    let new_val = q.get(i, j) - proj_coeff * q.get(i, k);
                    q.set(i, j, new_val);
                }
            }

            // Compute norm squared of the orthogonalized q_j
            let mut norm_sq = Rational64::from(0);
            for i in 0..m {
                let val = q.get(i, j);
                norm_sq += val * val;
            }

            if norm_sq == Rational64::from(0) {
                // Columns are linearly dependent
                return None;
            }

            // Set r[j,j] = 1 (since we keep q_j unnormalized)
            r.set(j, j, Rational64::from(1));
        }

        Some((q, r))
    }

    /// Performs Cholesky decomposition for symmetric positive definite matrices.
    /// Returns L where A = L * L^T and L is lower triangular.
    ///
    /// # Returns
    /// None if the matrix is not square, symmetric, or positive definite.
    ///
    /// # Note
    /// This requires taking square roots, which may not be exact for rationals.
    /// The function will fail (return None) if any diagonal element during
    /// decomposition is not a perfect rational square.
    pub fn cholesky_decomposition(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None; // Must be square
        }

        let n = self.rows;

        // Check symmetry
        for i in 0..n {
            for j in i + 1..n {
                if self.get(i, j) != self.get(j, i) {
                    return None; // Not symmetric
                }
            }
        }

        let mut l = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.get(i, j);

                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }

                if i == j {
                    // Diagonal element
                    if sum <= Rational64::from(0) {
                        return None; // Not positive definite
                    }

                    // Need to take square root
                    // Check if sum is a perfect rational square
                    let sqrt_val = rational_sqrt(sum)?;
                    l.set(i, i, sqrt_val);
                } else {
                    // Off-diagonal element
                    let l_jj = l.get(j, j);
                    if l_jj == Rational64::from(0) {
                        return None;
                    }
                    l.set(i, j, sum / l_jj);
                }
            }
        }

        Some(l)
    }

    /// Computes the inverse of a square matrix using LU decomposition.
    /// Returns None if the matrix is singular (non-invertible).
    ///
    /// # Algorithm
    /// Uses LU decomposition with partial pivoting to solve AX = I,
    /// where I is the identity matrix. Each column of the inverse is
    /// computed by solving a system with the corresponding column of I.
    pub fn inverse(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None; // Only square matrices can be inverted
        }

        let n = self.rows;

        // Use LU decomposition
        let (l, u, p) = self.lu_decomposition()?;

        // Create result matrix (will be the inverse)
        let mut inv = Matrix::zeros(n, n);

        // Solve for each column of the inverse
        for col in 0..n {
            // Create the col-th column of identity matrix
            let mut b = vec![Rational64::from(0); n];
            b[col] = Rational64::from(1);

            // Apply permutation to b
            let mut pb = vec![Rational64::from(0); n];
            for i in 0..n {
                pb[i] = b[p[i]];
            }

            // Solve L*y = pb (forward substitution)
            let mut y = vec![Rational64::from(0); n];
            for i in 0..n {
                let mut sum = pb[i];
                for (j, &y_j) in y.iter().enumerate().take(i) {
                    sum -= l.get(i, j) * y_j;
                }
                y[i] = sum; // L has 1s on diagonal
            }

            // Solve U*x = y (backward substitution)
            let mut x = vec![Rational64::from(0); n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                for (j, &x_j) in x.iter().enumerate().skip(i + 1).take(n - i - 1) {
                    sum -= u.get(i, j) * x_j;
                }
                let u_ii = u.get(i, i);
                if u_ii == Rational64::from(0) {
                    return None; // Singular matrix
                }
                x[i] = sum / u_ii;
            }

            // Set the col-th column of inverse
            for (i, &x_i) in x.iter().enumerate() {
                inv.set(i, col, x_i);
            }
        }

        Some(inv)
    }

    /// Checks if this matrix is the identity matrix.
    pub fn is_identity(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                let expected = if i == j {
                    Rational64::from(1)
                } else {
                    Rational64::from(0)
                };
                if self.get(i, j) != expected {
                    return false;
                }
            }
        }
        true
    }
}

/// Computes the square root of a Rational64 if it's a perfect square.
/// Returns None if the rational is not a perfect square.
fn rational_sqrt(r: Rational64) -> Option<Rational64> {
    use num_integer::Roots;
    use num_traits::Zero;

    if r < Rational64::zero() {
        return None;
    }
    if r == Rational64::zero() {
        return Some(Rational64::zero());
    }

    let numer = r.numer();
    let denom = r.denom();

    // Check if both numerator and denominator are perfect squares
    let sqrt_numer = numer.sqrt();
    let sqrt_denom = denom.sqrt();

    if sqrt_numer * sqrt_numer == *numer && sqrt_denom * sqrt_denom == *denom {
        Some(Rational64::new(sqrt_numer, sqrt_denom))
    } else {
        None
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

/// A sparse matrix stored in Compressed Sparse Row (CSR) format.
/// Only non-zero elements are stored, making it efficient for sparse matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    /// Row pointers: row_ptr[i] points to the start of row i in col_indices and values.
    /// row_ptr has length rows + 1, with row_ptr[rows] = nnz (number of non-zeros).
    row_ptr: Vec<usize>,
    /// Column indices for each non-zero value.
    col_indices: Vec<usize>,
    /// Non-zero values.
    values: Vec<Rational64>,
}

impl SparseMatrix {
    /// Creates a new sparse matrix with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates a sparse identity matrix.
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        matrix.row_ptr = (0..=size).collect();
        matrix.col_indices = (0..size).collect();
        matrix.values = vec![Rational64::from(1); size];
        matrix
    }

    /// Creates a sparse matrix from a dense matrix.
    pub fn from_dense(dense: &Matrix) -> Self {
        let mut matrix = Self::new(dense.nrows(), dense.ncols());
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..dense.nrows() {
            matrix.row_ptr[i] = values.len();
            for j in 0..dense.ncols() {
                let val = dense.get(i, j);
                if val != Rational64::from(0) {
                    col_indices.push(j);
                    values.push(val);
                }
            }
        }
        matrix.row_ptr[dense.nrows()] = values.len();
        matrix.col_indices = col_indices;
        matrix.values = values;
        matrix
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Returns the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Gets the element at position (row, col).
    pub fn get(&self, row: usize, col: usize) -> Rational64 {
        assert!(row < self.rows && col < self.cols);
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        // Binary search for the column index
        for i in start..end {
            if self.col_indices[i] == col {
                return self.values[i];
            } else if self.col_indices[i] > col {
                break;
            }
        }
        Rational64::from(0)
    }

    /// Sets the element at position (row, col).
    /// Note: This is inefficient for CSR format. Consider building the matrix
    /// incrementally using a builder pattern or from a dense matrix.
    pub fn set(&mut self, row: usize, col: usize, value: Rational64) {
        assert!(row < self.rows && col < self.cols);

        if value == Rational64::from(0) {
            // Remove the element if it exists
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                if self.col_indices[i] == col {
                    self.col_indices.remove(i);
                    self.values.remove(i);
                    // Update row pointers
                    for r in row + 1..=self.rows {
                        self.row_ptr[r] -= 1;
                    }
                    return;
                }
            }
            return;
        }

        // Find or insert the element
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for i in start..end {
            if self.col_indices[i] == col {
                // Update existing element
                self.values[i] = value;
                return;
            } else if self.col_indices[i] > col {
                // Insert new element
                self.col_indices.insert(i, col);
                self.values.insert(i, value);
                // Update row pointers
                for r in row + 1..=self.rows {
                    self.row_ptr[r] += 1;
                }
                return;
            }
        }

        // Append to end of row
        self.col_indices.insert(end, col);
        self.values.insert(end, value);
        for r in row + 1..=self.rows {
            self.row_ptr[r] += 1;
        }
    }

    /// Sparse matrix-vector multiplication: returns A * v
    #[allow(clippy::needless_range_loop)]
    pub fn mul_vec(&self, v: &[Rational64]) -> Vec<Rational64> {
        assert_eq!(
            v.len(),
            self.cols,
            "Vector dimension must match matrix columns"
        );
        let mut result = vec![Rational64::from(0); self.rows];

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = Rational64::from(0);

            for k in start..end {
                sum += self.values[k] * v[self.col_indices[k]];
            }
            result[i] = sum;
        }

        result
    }

    /// Sparse matrix-matrix multiplication: returns A * B
    pub fn mul(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions must be compatible for multiplication"
        );

        let mut result = SparseMatrix::new(self.rows, other.cols);
        let mut row_data: Vec<Rational64> = vec![Rational64::from(0); other.cols];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..self.rows {
            result.row_ptr[i] = values.len();

            // Compute row i of result
            row_data.fill(Rational64::from(0));

            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for k in start..end {
                let a_val = &self.values[k];
                let k_col = self.col_indices[k];

                // Multiply with row k of B
                let b_start = other.row_ptr[k_col];
                let b_end = other.row_ptr[k_col + 1];

                for j in b_start..b_end {
                    let b_col = other.col_indices[j];
                    let b_val = &other.values[j];
                    row_data[b_col] += a_val * b_val;
                }
            }

            // Store non-zero elements
            for (j, val) in row_data.iter().enumerate() {
                if val != &Rational64::from(0) {
                    col_indices.push(j);
                    values.push(*val);
                }
            }
        }

        result.row_ptr[self.rows] = values.len();
        result.col_indices = col_indices;
        result.values = values;
        result
    }

    /// Transposes the sparse matrix.
    pub fn transpose(&self) -> SparseMatrix {
        let mut result = SparseMatrix::new(self.cols, self.rows);
        let mut col_counts = vec![0; self.cols];

        // Count elements per column (which become rows in transpose)
        for &col in &self.col_indices {
            col_counts[col] += 1;
        }

        // Build row pointers for transpose
        result.row_ptr[0] = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.cols {
            result.row_ptr[i + 1] = result.row_ptr[i] + col_counts[i];
        }

        // Allocate space
        result.col_indices = vec![0; self.nnz()];
        result.values = vec![Rational64::from(0); self.nnz()];

        // Fill in values
        let mut current_pos = result.row_ptr.clone();
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for k in start..end {
                let col = self.col_indices[k];
                let pos = current_pos[col];
                result.col_indices[pos] = i;
                result.values[pos] = self.values[k];
                current_pos[col] += 1;
            }
        }

        result
    }

    /// Converts the sparse matrix to a dense matrix.
    pub fn to_dense(&self) -> Matrix {
        let mut dense = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                dense.set(i, self.col_indices[k], self.values[k]);
            }
        }
        dense
    }
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SparseMatrix({}x{}, {} non-zeros):",
            self.rows,
            self.cols,
            self.nnz()
        )?;
        for i in 0..self.rows {
            write!(f, "  Row {}: ", i)?;
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                write!(f, "({}, {}) ", self.col_indices[k], self.values[k])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.get(0, 0), Rational64::from(0));
        assert_eq!(m.get(1, 2), Rational64::from(0));
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), Rational64::from(1));
        assert_eq!(m.get(1, 1), Rational64::from(1));
        assert_eq!(m.get(2, 2), Rational64::from(1));
        assert_eq!(m.get(0, 1), Rational64::from(0));
        assert_eq!(m.get(1, 2), Rational64::from(0));
    }

    #[test]
    fn test_matrix_from_rows() {
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        assert_eq!(m.get(0, 0), Rational64::from(1));
        assert_eq!(m.get(0, 1), Rational64::from(2));
        assert_eq!(m.get(1, 0), Rational64::from(3));
        assert_eq!(m.get(1, 1), Rational64::from(4));
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        m.swap_rows(0, 1);
        assert_eq!(m.get(0, 0), Rational64::from(3));
        assert_eq!(m.get(0, 1), Rational64::from(4));
        assert_eq!(m.get(1, 0), Rational64::from(1));
        assert_eq!(m.get(1, 1), Rational64::from(2));
    }

    #[test]
    fn test_matrix_add_row_multiple() {
        let mut m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        m.add_row_multiple(1, 0, Rational64::from(2));
        assert_eq!(m.get(0, 0), Rational64::from(1));
        assert_eq!(m.get(0, 1), Rational64::from(2));
        assert_eq!(m.get(1, 0), Rational64::from(5)); // 3 + 2*1
        assert_eq!(m.get(1, 1), Rational64::from(8)); // 4 + 2*2
    }

    #[test]
    fn test_matrix_mul_vec() {
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        let v = vec![Rational64::from(5), Rational64::from(6)];
        let result = m.mul_vec(&v);
        assert_eq!(result[0], Rational64::from(17)); // 1*5 + 2*6
        assert_eq!(result[1], Rational64::from(39)); // 3*5 + 4*6
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        let b = Matrix::from_rows(vec![
            vec![Rational64::from(5), Rational64::from(6)],
            vec![Rational64::from(7), Rational64::from(8)],
        ]);
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), Rational64::from(19)); // 1*5 + 2*7
        assert_eq!(c.get(0, 1), Rational64::from(22)); // 1*6 + 2*8
        assert_eq!(c.get(1, 0), Rational64::from(43)); // 3*5 + 4*7
        assert_eq!(c.get(1, 1), Rational64::from(50)); // 3*6 + 4*8
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_rows(vec![
            vec![
                Rational64::from(1),
                Rational64::from(2),
                Rational64::from(3),
            ],
            vec![
                Rational64::from(4),
                Rational64::from(5),
                Rational64::from(6),
            ],
        ]);
        let t = m.transpose();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t.get(0, 0), Rational64::from(1));
        assert_eq!(t.get(1, 0), Rational64::from(2));
        assert_eq!(t.get(2, 0), Rational64::from(3));
        assert_eq!(t.get(0, 1), Rational64::from(4));
        assert_eq!(t.get(1, 1), Rational64::from(5));
        assert_eq!(t.get(2, 1), Rational64::from(6));
    }

    #[test]
    fn test_gaussian_elimination() {
        let m = Matrix::from_rows(vec![
            vec![
                Rational64::from(2),
                Rational64::from(1),
                Rational64::from(-1),
            ],
            vec![
                Rational64::from(-3),
                Rational64::from(-1),
                Rational64::from(2),
            ],
            vec![
                Rational64::from(-2),
                Rational64::from(1),
                Rational64::from(2),
            ],
        ]);
        let (ref_form, rank) = m.gaussian_elimination();
        assert_eq!(rank, 3);
        // Check that it's in row echelon form (upper triangular)
        assert_ne!(ref_form.get(0, 0), Rational64::from(0));
        assert_eq!(ref_form.get(1, 0), Rational64::from(0));
        assert_eq!(ref_form.get(2, 0), Rational64::from(0));
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve: 2x + y - z = 8
        //       -3x - y + 2z = -11
        //       -2x + y + 2z = -3
        let a = Matrix::from_rows(vec![
            vec![
                Rational64::from(2),
                Rational64::from(1),
                Rational64::from(-1),
            ],
            vec![
                Rational64::from(-3),
                Rational64::from(-1),
                Rational64::from(2),
            ],
            vec![
                Rational64::from(-2),
                Rational64::from(1),
                Rational64::from(2),
            ],
        ]);
        let b = vec![
            Rational64::from(8),
            Rational64::from(-11),
            Rational64::from(-3),
        ];
        let x = a.solve(&b).unwrap();
        assert_eq!(x[0], Rational64::from(2));
        assert_eq!(x[1], Rational64::from(3));
        assert_eq!(x[2], Rational64::from(-1));
    }

    #[test]
    fn test_lu_decomposition() {
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(4), Rational64::from(3)],
            vec![Rational64::from(6), Rational64::from(3)],
        ]);
        let (l, u, _p) = m.lu_decomposition().unwrap();

        // L should be lower triangular with ones on diagonal
        assert_eq!(l.get(0, 0), Rational64::from(1));
        assert_eq!(l.get(1, 1), Rational64::from(1));
        assert_eq!(l.get(0, 1), Rational64::from(0));

        // U should be upper triangular
        assert_eq!(u.get(1, 0), Rational64::from(0));
    }

    #[test]
    fn test_determinant() {
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        let det = m.determinant().unwrap();
        assert_eq!(det, Rational64::from(-2)); // 1*4 - 2*3 = -2
    }

    #[test]
    fn test_determinant_3x3() {
        let m = Matrix::from_rows(vec![
            vec![
                Rational64::from(6),
                Rational64::from(1),
                Rational64::from(1),
            ],
            vec![
                Rational64::from(4),
                Rational64::from(-2),
                Rational64::from(5),
            ],
            vec![
                Rational64::from(2),
                Rational64::from(8),
                Rational64::from(7),
            ],
        ]);
        let det = m.determinant().unwrap();
        assert_eq!(det, Rational64::from(-306));
    }

    #[test]
    fn test_sparse_matrix_identity() {
        let m = SparseMatrix::identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.get(0, 0), Rational64::from(1));
        assert_eq!(m.get(1, 1), Rational64::from(1));
        assert_eq!(m.get(2, 2), Rational64::from(1));
        assert_eq!(m.get(0, 1), Rational64::from(0));
        assert_eq!(m.get(1, 2), Rational64::from(0));
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = Matrix::from_rows(vec![
            vec![
                Rational64::from(1),
                Rational64::from(0),
                Rational64::from(2),
            ],
            vec![
                Rational64::from(0),
                Rational64::from(3),
                Rational64::from(0),
            ],
            vec![
                Rational64::from(4),
                Rational64::from(0),
                Rational64::from(5),
            ],
        ]);
        let sparse = SparseMatrix::from_dense(&dense);
        assert_eq!(sparse.nnz(), 5); // Five non-zero elements
        assert_eq!(sparse.get(0, 0), Rational64::from(1));
        assert_eq!(sparse.get(0, 2), Rational64::from(2));
        assert_eq!(sparse.get(1, 1), Rational64::from(3));
        assert_eq!(sparse.get(2, 0), Rational64::from(4));
        assert_eq!(sparse.get(2, 2), Rational64::from(5));
        assert_eq!(sparse.get(0, 1), Rational64::from(0));
    }

    #[test]
    fn test_sparse_to_dense() {
        let sparse = SparseMatrix::identity(3);
        let dense = sparse.to_dense();
        assert_eq!(dense.get(0, 0), Rational64::from(1));
        assert_eq!(dense.get(1, 1), Rational64::from(1));
        assert_eq!(dense.get(2, 2), Rational64::from(1));
        assert_eq!(dense.get(0, 1), Rational64::from(0));
    }

    #[test]
    fn test_sparse_mul_vec() {
        let dense = Matrix::from_rows(vec![
            vec![
                Rational64::from(1),
                Rational64::from(0),
                Rational64::from(2),
            ],
            vec![
                Rational64::from(0),
                Rational64::from(3),
                Rational64::from(0),
            ],
        ]);
        let sparse = SparseMatrix::from_dense(&dense);
        let v = vec![
            Rational64::from(1),
            Rational64::from(2),
            Rational64::from(3),
        ];
        let result = sparse.mul_vec(&v);
        assert_eq!(result[0], Rational64::from(7)); // 1*1 + 0*2 + 2*3 = 7
        assert_eq!(result[1], Rational64::from(6)); // 0*1 + 3*2 + 0*3 = 6
    }

    #[test]
    fn test_sparse_transpose() {
        let dense = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
            vec![Rational64::from(5), Rational64::from(6)],
        ]);
        let sparse = SparseMatrix::from_dense(&dense);
        let transposed = sparse.transpose();
        assert_eq!(transposed.nrows(), 2);
        assert_eq!(transposed.ncols(), 3);
        assert_eq!(transposed.get(0, 0), Rational64::from(1));
        assert_eq!(transposed.get(0, 1), Rational64::from(3));
        assert_eq!(transposed.get(0, 2), Rational64::from(5));
        assert_eq!(transposed.get(1, 0), Rational64::from(2));
        assert_eq!(transposed.get(1, 1), Rational64::from(4));
        assert_eq!(transposed.get(1, 2), Rational64::from(6));
    }

    #[test]
    fn test_sparse_mul() {
        let a_dense = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);
        let b_dense = Matrix::from_rows(vec![
            vec![Rational64::from(5), Rational64::from(6)],
            vec![Rational64::from(7), Rational64::from(8)],
        ]);
        let a = SparseMatrix::from_dense(&a_dense);
        let b = SparseMatrix::from_dense(&b_dense);
        let c = a.mul(&b);
        let c_dense = c.to_dense();
        assert_eq!(c_dense.get(0, 0), Rational64::from(19)); // 1*5 + 2*7
        assert_eq!(c_dense.get(0, 1), Rational64::from(22)); // 1*6 + 2*8
        assert_eq!(c_dense.get(1, 0), Rational64::from(43)); // 3*5 + 4*7
        assert_eq!(c_dense.get(1, 1), Rational64::from(50)); // 3*6 + 4*8
    }

    #[test]
    fn test_qr_decomposition_simple() {
        // Test QR decomposition with a simple 2x2 matrix
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(1)],
            vec![Rational64::from(0), Rational64::from(1)],
        ]);

        let (q, r) = m.qr_decomposition().unwrap();

        // Verify QR = A by multiplying Q and R
        let reconstructed = q.mul(&r);

        // Check that reconstructed matrix equals original
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert_eq!(
                    m.get(i, j),
                    reconstructed.get(i, j),
                    "QR reconstruction failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qr_decomposition_3x2() {
        // Test with a tall matrix (m > n)
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(0)],
            vec![Rational64::from(1), Rational64::from(1)],
            vec![Rational64::from(0), Rational64::from(1)],
        ]);

        let (q, r) = m.qr_decomposition().unwrap();

        // Verify dimensions
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 2);
        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 2);

        // Verify QR = A
        let reconstructed = q.mul(&r);
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert_eq!(m.get(i, j), reconstructed.get(i, j));
            }
        }
    }

    #[test]
    fn test_cholesky_decomposition_simple() {
        // Test with a simple 2x2 positive definite matrix
        // A = [[4, 2],
        //      [2, 3]]
        // L = [[2, 0],
        //      [1, sqrt(2)]]
        // But sqrt(2) is not rational, so this matrix won't decompose exactly

        // Instead, use a matrix with rational square roots
        // A = [[4, 2],
        //      [2, 2]]
        // L = [[2, 0],
        //      [1, 1]]
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(4), Rational64::from(2)],
            vec![Rational64::from(2), Rational64::from(2)],
        ]);

        let l = m.cholesky_decomposition().unwrap();

        // Verify L * L^T = A
        let lt = l.transpose();
        let reconstructed = l.mul(&lt);

        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert_eq!(m.get(i, j), reconstructed.get(i, j));
            }
        }

        // Verify L is lower triangular
        assert_eq!(l.get(0, 1), Rational64::from(0));
    }

    #[test]
    fn test_cholesky_decomposition_identity() {
        // Identity matrix should have Cholesky decomposition = identity
        let m = Matrix::identity(3);
        let l = m.cholesky_decomposition().unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(l.get(i, j), m.get(i, j));
            }
        }
    }

    #[test]
    fn test_cholesky_decomposition_3x3() {
        // Test with a 3x3 positive definite matrix
        // A = [[9, 3, 3],
        //      [3, 5, 1],
        //      [3, 1, 5]]
        // This has a rational Cholesky decomposition
        let m = Matrix::from_rows(vec![
            vec![
                Rational64::from(9),
                Rational64::from(3),
                Rational64::from(3),
            ],
            vec![
                Rational64::from(3),
                Rational64::from(5),
                Rational64::from(1),
            ],
            vec![
                Rational64::from(3),
                Rational64::from(1),
                Rational64::from(5),
            ],
        ]);

        let l = m.cholesky_decomposition().unwrap();

        // Verify L * L^T = A
        let lt = l.transpose();
        let reconstructed = l.mul(&lt);

        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert_eq!(
                    m.get(i, j),
                    reconstructed.get(i, j),
                    "Cholesky reconstruction failed at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Verify L is lower triangular
        for i in 0..3 {
            for j in i + 1..3 {
                assert_eq!(l.get(i, j), Rational64::from(0));
            }
        }
    }

    #[test]
    fn test_cholesky_decomposition_not_symmetric() {
        // Non-symmetric matrix should fail
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(4), Rational64::from(1)],
            vec![Rational64::from(2), Rational64::from(3)],
        ]);

        assert!(m.cholesky_decomposition().is_none());
    }

    #[test]
    fn test_cholesky_decomposition_not_positive_definite() {
        // Symmetric but not positive definite matrix should fail
        let m = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(2), Rational64::from(1)],
        ]);

        // This matrix has eigenvalues 3 and -1, so it's not positive definite
        assert!(m.cholesky_decomposition().is_none());
    }

    #[test]
    fn test_rational_sqrt_perfect_squares() {
        assert_eq!(
            rational_sqrt(Rational64::from(0)),
            Some(Rational64::from(0))
        );
        assert_eq!(
            rational_sqrt(Rational64::from(1)),
            Some(Rational64::from(1))
        );
        assert_eq!(
            rational_sqrt(Rational64::from(4)),
            Some(Rational64::from(2))
        );
        assert_eq!(
            rational_sqrt(Rational64::from(9)),
            Some(Rational64::from(3))
        );
        assert_eq!(
            rational_sqrt(Rational64::from(16)),
            Some(Rational64::from(4))
        );
        assert_eq!(
            rational_sqrt(Rational64::from(25)),
            Some(Rational64::from(5))
        );

        // Test rational perfect squares
        assert_eq!(
            rational_sqrt(Rational64::new(1, 4)),
            Some(Rational64::new(1, 2))
        );
        assert_eq!(
            rational_sqrt(Rational64::new(9, 16)),
            Some(Rational64::new(3, 4))
        );
    }

    #[test]
    fn test_rational_sqrt_not_perfect_squares() {
        assert!(rational_sqrt(Rational64::from(2)).is_none());
        assert!(rational_sqrt(Rational64::from(3)).is_none());
        assert!(rational_sqrt(Rational64::from(5)).is_none());
        assert!(rational_sqrt(Rational64::new(1, 2)).is_none());
        assert!(rational_sqrt(Rational64::new(2, 3)).is_none());
        assert!(rational_sqrt(Rational64::from(-1)).is_none()); // Negative
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        // Test with a simple 2x2 matrix
        // A = [[1, 2],
        //      [3, 4]]
        // A^-1 = [[-2, 1],
        //         [3/2, -1/2]]
        let a = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(3), Rational64::from(4)],
        ]);

        let a_inv = a.inverse().unwrap();

        // Verify A * A^-1 = I
        let product = a.mul(&a_inv);
        assert!(product.is_identity(), "A * A^-1 should be identity");

        // Verify A^-1 * A = I
        let product2 = a_inv.mul(&a);
        assert!(product2.is_identity(), "A^-1 * A should be identity");
    }

    #[test]
    fn test_matrix_inverse_identity() {
        // Identity matrix inverse should be itself
        let i = Matrix::identity(3);
        let i_inv = i.inverse().unwrap();
        assert!(i_inv.is_identity());
    }

    #[test]
    fn test_matrix_inverse_3x3() {
        // Test with a 3x3 matrix
        let a = Matrix::from_rows(vec![
            vec![
                Rational64::from(2),
                Rational64::from(1),
                Rational64::from(1),
            ],
            vec![
                Rational64::from(1),
                Rational64::from(2),
                Rational64::from(1),
            ],
            vec![
                Rational64::from(1),
                Rational64::from(1),
                Rational64::from(2),
            ],
        ]);

        let a_inv = a.inverse().unwrap();

        // Verify A * A^-1 = I
        let product = a.mul(&a_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j {
                    Rational64::from(1)
                } else {
                    Rational64::from(0)
                };
                assert_eq!(
                    product.get(i, j),
                    expected,
                    "A * A^-1 failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_matrix_inverse_singular() {
        // Singular matrix (determinant = 0) should not have an inverse
        let a = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(2)],
            vec![Rational64::from(2), Rational64::from(4)], // Second row is 2x first row
        ]);

        assert!(
            a.inverse().is_none(),
            "Singular matrix should not have inverse"
        );
    }

    #[test]
    fn test_matrix_inverse_non_square() {
        // Non-square matrix should not have an inverse
        let a = Matrix::from_rows(vec![
            vec![
                Rational64::from(1),
                Rational64::from(2),
                Rational64::from(3),
            ],
            vec![
                Rational64::from(4),
                Rational64::from(5),
                Rational64::from(6),
            ],
        ]);

        assert!(
            a.inverse().is_none(),
            "Non-square matrix should not have inverse"
        );
    }

    #[test]
    fn test_is_identity() {
        let i = Matrix::identity(3);
        assert!(i.is_identity());

        let not_i = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(0)],
            vec![Rational64::from(0), Rational64::from(2)],
        ]);
        assert!(!not_i.is_identity());

        let also_not_i = Matrix::from_rows(vec![
            vec![Rational64::from(1), Rational64::from(1)],
            vec![Rational64::from(0), Rational64::from(1)],
        ]);
        assert!(!also_not_i.is_identity());
    }
}
