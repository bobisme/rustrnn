extern crate rand;

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, Index};
use utils::{Sigmoid, sigmoid, sigmoid_derivative};
use std::fmt;

#[derive(PartialEq, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: (0..(rows * cols)).map(|_| 0.0).collect(),
        }
    }

    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        let data: Vec<f64> = (0..(rows * cols))
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();
        Matrix {
            rows,
            cols,
            data: data,
        }
    }

    pub fn from(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        Matrix {
            rows,
            cols,
            data: data.to_vec(),
        }
    }
}

pub trait Transpose {
    fn transpose(self) -> Matrix;
}

impl<'a> Transpose for &'a Matrix {
    fn transpose(self) -> Matrix {
        let mut out = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let out_index = out.get_index((col, row));
                out.data[out_index] = self[(row, col)];
            }
        }
        out
    }
}

impl Transpose for Matrix {
    fn transpose(self) -> Matrix {
        (&self).transpose()
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

trait FlatIndex<Idx>
    where Idx: ?Sized
{
    fn get_index(&self, index: Idx) -> usize;
}

impl FlatIndex<(usize, usize)> for Matrix {
    fn get_index(&self, (row, col): (usize, usize)) -> usize {
        self.cols * row + col
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[self.get_index((row, col))]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner_slices: Vec<_> = (0..self.rows)
            .map(|row| {
                     let range = (self.cols * row)..(self.cols * row + self.cols);
                     let slice = &(self.data)[range];
                     slice.iter().map(|&x| format!("{:.3}", x)).collect()
                 })
            .map(|x: Vec<String>| format!(" {}", x.join(", ")))
            .collect();
        write!(f, "[\n{}\n]", inner_slices.join(",\n")).unwrap();
        Ok(())
    }
}

pub fn element_mul(lhs: &Matrix, rhs: &Matrix) -> Result<Matrix, String> {
    if lhs.cols != rhs.cols {
        return Err(format!("cols do not match: {} vs {}", lhs.cols, rhs.cols));
    }
    if lhs.rows != rhs.rows {
        return Err(format!("rows do not match: {} vs {}", lhs.rows, rhs.rows));
    }
    Ok(Matrix {
           rows: lhs.rows,
           cols: rhs.cols,
           data: lhs.data
               .iter()
               .zip(rhs.data.iter())
               .map(|(x, y)| x * y)
               .collect(),
       })
}

impl<'a> Mul<f64> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Self::Output {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| x * rhs).collect(),
        }
    }
}

impl Mul<f64> for Matrix {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

pub trait Dot<RHS> {
    fn dot(self, rhs: RHS) -> Result<Matrix, String>;
}

impl<'a> Dot<&'a Matrix> for &'a Matrix {
    fn dot(self, rhs: &Matrix) -> Result<Matrix, String> {
        if self.cols != rhs.rows {
            let err = format!("Left cols don't match right rows: [{}×{}] ⋅ [{}×{}]",
                              self.rows,
                              self.cols,
                              rhs.rows,
                              rhs.cols);
            return Err(err);
        }
        let mut out = Matrix::new(self.rows, rhs.cols);
        let (lhs_cols, lhs_rows) = (self.cols, self.rows);
        let rhs_cols = rhs.cols;
        for out_row in 0..lhs_rows {
            for out_col in 0..rhs_cols {
                let out_index = out.get_index((out_row, out_col));
                out.data[out_index] = (0..lhs_cols)
                    .map(|i| self[(out_row, i)] * rhs[(i, out_col)])
                    .sum();
            }
        }
        Ok(out)

    }
}

impl<'a> Dot<&'a Matrix> for Matrix {
    fn dot(self, rhs: &Matrix) -> Result<Matrix, String> {
        (&self).dot(rhs)
    }
}

impl<'a> Dot<Matrix> for &'a Matrix {
    fn dot(self, rhs: Matrix) -> Result<Matrix, String> {
        self.dot(&rhs)
    }
}

impl Dot<Matrix> for Matrix {
    fn dot(self, rhs: Matrix) -> Result<Matrix, String> {
        (&self).dot(&rhs)
    }
}

impl<'a> Mul<&'a Matrix> for &'a Matrix {
    type Output = Result<Matrix, String>;
    fn mul(self, rhs: &'a Matrix) -> Self::Output {
        element_mul(self, rhs)
    }
}

impl<'a> Mul<Matrix> for &'a Matrix {
    type Output = Result<Matrix, String>;
    fn mul(self, rhs: Matrix) -> Self::Output {
        element_mul(self, &rhs)
    }
}

impl<'a> Mul<&'a Matrix> for Matrix {
    type Output = Result<Matrix, String>;
    fn mul(self, rhs: &'a Matrix) -> Self::Output {
        &self * rhs
    }
}

impl<'a> Mul<Matrix> for Matrix {
    type Output = Result<Matrix, String>;
    fn mul(self, rhs: Matrix) -> Self::Output {
        &self * &rhs
    }
}

macro_rules! impl_mat_add {
    ($lhs:ty, $rhs:ty) => (

        impl<'a> Add<$rhs> for $lhs {
            type Output = Matrix;
            fn add(self, rhs: $rhs) -> Self::Output {
                if self.cols != rhs.cols || self.rows != rhs.rows {
                    panic!("can't add these. rows and cols must be the same");
                }
                Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    data: self.data
                        .iter()
                        .zip(rhs.data.iter())
                        .map(|(&l, &r)| l + r)
                        .collect(),
                }
            }
        }

        )
}

impl_mat_add!(Matrix, Matrix);
impl_mat_add!(Matrix, &'a Matrix);
impl_mat_add!(&'a Matrix, &'a Matrix);
impl_mat_add!(&'a Matrix, Matrix);

macro_rules! impl_mat_addassign {
    ($rhs:ty) => (
        impl<'a> AddAssign<$rhs> for Matrix {
            fn add_assign(&mut self, rhs: $rhs) {
                for i in 0..self.data.len() {
                    self.data[i] += rhs.data[i];
                }
            }
        }
        )
}

impl_mat_addassign!(&'a Matrix);
impl_mat_addassign!(Matrix);

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.data.len() {
            self.data[i] *= rhs;
        }
    }
}

impl<'a> Sub<&'a Matrix> for Matrix {
    type Output = Self;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.cols != rhs.cols || self.rows != rhs.rows {
            panic!("can't add these. rows and cols must be the same");
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(&l, &r)| l - r)
                .collect(),
        }
    }
}

impl Sigmoid for Matrix {
    fn sigmoid(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| sigmoid(x)).collect(),
        }
    }

    fn sigmoid_derivative(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| sigmoid_derivative(x)).collect(),
        }
    }
}

#[cfg(test)]
mod describe_matrix {
    use super::*;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => ({
            let (a, b) = (&$a, &$b);
            assert!((*a - *b).abs() < 1.0e-6, "{} ≉ {}", *a, *b);
        })
    }

    #[test]
    fn it_can_test_for_equivalence() {
        let m1 = Matrix::new(2, 2);
        let m3 = Matrix::new_random(2, 2);
        assert!(m1 != m3);
    }

    #[test]
    fn it_can_multiply_by_a_scalar() {
        let m1 = Matrix::from(2, 2, &vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = m1 * 2.0;
        assert_eq!(m2.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn it_can_be_indexed() {
        let m = Matrix::from(3, 2, &vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
        assert_eq!(m[(2, 0)], 5.0);
        assert_eq!(m[(2, 1)], 6.0);
    }

    #[test]
    fn it_can_be_sigmoided() {
        let m1 = &Matrix::from(2, 2, &vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = m1.sigmoid();
        assert_approx_eq!(m2[(0, 0)], 0.7310585786300048792511592418218362744);
        assert_approx_eq!(m2[(0, 1)], 0.8807970779778824440597291413023967952);
    }

    #[test]
    fn it_can_be_transposed() {
        let m = Matrix::from(3, 2, &vec![1., 2., 3., 4., 5., 6.]);
        let transposed = m.transpose();
        println!("{:?}", transposed);
        assert_eq!(transposed[(0, 0)], 1.0);
        assert_eq!(transposed[(0, 1)], 3.0);
        assert_eq!(transposed[(0, 2)], 5.0);
        assert_eq!(transposed[(1, 0)], 2.0);
        assert_eq!(transposed[(1, 1)], 4.0);
        assert_eq!(transposed[(1, 2)], 6.0);
    }

    #[cfg(test)]
    mod multiplying_two_matricies {
        use super::*;

        #[test]
        fn it_can_calculate_dot_product_with_dot() {
            let m1 = Matrix::from(2, 3, &vec![1., 2., 3., 4., 5., 6.]);
            let m2 = Matrix::from(3, 2, &vec![1., 2., 1., 2., 1., 2.]);
            let dot = m1.dot(&m2).unwrap();
            assert_approx_eq!(dot[(0, 0)], 6.);
            assert_approx_eq!(dot[(0, 1)], 12.);
            assert_approx_eq!(dot[(1, 0)], 15.);
            assert_approx_eq!(dot[(1, 1)], 30.);
        }

        #[test]
        fn it_can_multiply_struct_and_struct() {
            let m1 = Matrix::from(2, 2, &vec![2., 2., 2., 2.]);
            let m2 = Matrix::from(2, 2, &vec![3., 3., 3., 3.]);
            m1.dot(m2).unwrap();
        }

        #[test]
        fn it_can_multiply_struct_and_ref() {
            let m1 = Matrix::from(2, 2, &vec![2., 2., 2., 2.]);
            let m2 = &Matrix::from(2, 2, &vec![3., 3., 3., 3.]);
            m1.dot(m2).unwrap();
        }

        #[test]
        fn it_can_multiply_ref_and_struct() {
            let m1 = &Matrix::from(2, 2, &vec![2., 2., 2., 2.]);
            let m2 = Matrix::from(2, 2, &vec![3., 3., 3., 3.]);
            m1.dot(m2).unwrap();
        }

        #[test]
        fn it_can_multiply_ref_and_ref() {
            let m1 = &Matrix::from(2, 2, &vec![2., 2., 2., 2.]);
            let m2 = &Matrix::from(2, 2, &vec![3., 3., 3., 3.]);
            m2.dot(m1).unwrap();
        }

        #[test]
        #[should_panic]
        fn it_panics_if_the_matrices_arent_the_right_size() {
            let m1 = Matrix::new(4, 2);
            let m2 = Matrix::new(3, 5);
            m1.dot(&m2).unwrap();
        }

    }

    #[cfg(test)]
    mod adding_two_matrices {
        use super::*;

        #[test]
        fn it_can_add_two_matrices() {
            let m1 = Matrix::from(2, 2, &vec![1., 2., 3., 4.]);
            let m2 = Matrix::from(2, 2, &vec![1., 2., 1., 2.]);
            let out = m1 + &m2;
            assert_approx_eq!(out[(0, 0)], 2.);
            assert_approx_eq!(out[(0, 1)], 4.);
            assert_approx_eq!(out[(1, 0)], 4.);
            assert_approx_eq!(out[(1, 1)], 6.);
        }

        #[test]
        #[should_panic]
        fn it_panics_if_the_matrices_arent_the_same_size() {
            let m1 = Matrix::new(2, 2);
            let m2 = Matrix::new(2, 3);
            m1 + &m2;
        }
    }

    #[cfg(test)]
    mod elementwise_multiplying_two_matrices {
        use super::*;

        #[test]
        fn it_can_multiply_two_matrices() {
            let m1 = Matrix::from(2, 2, &vec![1., 2., 3., 4.]);
            let m2 = Matrix::from(2, 2, &vec![1., 2., 1., 2.]);
            let out = (m1 * &m2).unwrap();
            assert_approx_eq!(out[(0, 0)], 1.);
            assert_approx_eq!(out[(0, 1)], 4.);
            assert_approx_eq!(out[(1, 0)], 3.);
            assert_approx_eq!(out[(1, 1)], 8.);
        }

        #[test]
        #[should_panic]
        fn it_panics_if_the_matrices_arent_the_same_size() {
            let m1 = Matrix::new(2, 2);
            let m2 = Matrix::new(2, 3);
            (m1 * &m2).unwrap();
        }
    }

    #[cfg(test)]
    mod subtracting_two_matrices {
        use super::*;

        #[test]
        fn it_can_subtract_two_matrices() {
            let m1 = Matrix::from(2, 2, &vec![1., 2., 3., 4.]);
            let m2 = Matrix::from(2, 2, &vec![1., 2., 1., 2.]);
            let out = m1 - &m2;
            assert_approx_eq!(out[(0, 0)], 0.);
            assert_approx_eq!(out[(0, 1)], 0.);
            assert_approx_eq!(out[(1, 0)], 2.);
            assert_approx_eq!(out[(1, 1)], 2.);
        }

        #[test]
        #[should_panic]
        fn it_panics_if_the_matrices_arent_the_same_size() {
            let m1 = Matrix::new(2, 2);
            let m2 = Matrix::new(2, 3);
            m1 - &m2;
        }
    }
}
