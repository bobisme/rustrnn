extern crate rand;

use std::ops::{Mul, Index};
use utils::{Sigmoid, sigmoid, sigmoid_derivative};

#[derive(PartialEq, Debug)]
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
        let data = (0..(rows * cols)).map(|_| rand::random()).collect();
        Matrix {
            rows,
            cols,
            data: data,
        }
    }

    fn from(rows: usize, cols: usize, data: &Vec<f64>) -> Matrix {
        Matrix {
            rows,
            cols,
            data: data.clone(),
        }
    }

    // pub fn exp(&self) -> Matrix {
    //     Matrix {
    //         rows: self.rows,
    //         cols: self.cols,
    //         data: self.data.clone().iter().map(|x| E.powf(*x)).collect(),
    //     }
    // }
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

// impl fmt::Debug for Matrix {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         self.data.fmt(f)
//     }
// }

impl Mul<f64> for Matrix {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| x * rhs).collect(),
        }
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Self;
    fn mul(self, rhs: Matrix) -> Self::Output {
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
        out
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

trait Transpose {
    fn transpose(&self) -> Self;
}

impl Transpose for Matrix {
    fn transpose(&self) -> Matrix {
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
        let m2 = Matrix::new(2, 2);
        assert_eq!(m1, m2);
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

    #[test]
    fn it_can_calculate_dot_product() {
        let m1 = Matrix::from(2, 3, &vec![1., 2., 3., 4., 5., 6.]);
        let m2 = Matrix::from(3, 2, &vec![1., 2., 1., 2., 1., 2.]);
        let dot = m1 * m2;
        assert_approx_eq!(dot[(0, 0)], 6.);
        assert_approx_eq!(dot[(0, 1)], 12.);
        assert_approx_eq!(dot[(1, 0)], 15.);
        assert_approx_eq!(dot[(1, 1)], 30.);
    }
}
