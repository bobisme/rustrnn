#[cfg(test)]
mod tests {
    use utils::*;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => ({
            let (a, b) = (&$a, &$b);
            assert!((*a - *b).abs() < 1.0e-6, "{} ≉ {}", *a, *b);
        })
    }

    mod describe_sigmoid {
        use super::*;

        #[test]
        fn it_should_work() {
            assert_approx_eq!(sigmoid(1.0), 0.73105857863000);
        }
    }

    #[test]
    fn test_sigmoid_derivative() {
        assert_approx_eq!(sigmoid_derivative(1.0), 0.0);
        assert_approx_eq!(sigmoid_derivative(0.5), 0.25);
    }
}
