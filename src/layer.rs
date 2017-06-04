use matrix::Matrix;

pub struct Layer {
    input_size: usize,
    size: usize,
    pub weights: Matrix,
}

impl Layer {
    pub fn new(input_size: usize, size: usize) -> Layer {
        Layer {
            input_size: input_size,
            size: size,
            weights: Matrix::new_random(input_size, size),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod describe_new {
        use super::*;

        #[test]
        fn it_creates_a_new_layer() {
            let l = Layer::new(2, 3);
            assert_eq!(l.input_size(), 2);
            assert_eq!(l.size(), 3);
        }
    }
}
