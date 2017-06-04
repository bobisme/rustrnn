pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}


pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

pub trait Sigmoid {
    fn sigmoid(&self) -> Self;
    fn sigmoid_derivative(&self) -> Self;
}
