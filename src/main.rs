extern crate rand;

use rand::distributions::{Range,IndependentSample};

mod utils;
#[macro_use]
mod utils_tests;
mod layer;
mod matrix;

use utils::{Sigmoid};
use matrix::{Matrix, Transpose};

// const LEARNING_RATE: f64 = 0.1;
const INPUT_SIZE: usize = 2;
const HIDDEN_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 1;
const BIT_COUNT: usize = 8;


fn get_bits(i: i32) -> Vec<i32> {
    (0..BIT_COUNT).rev().map(|k| (i & (1 << k)) >> k).collect()
}

fn forward_hidden_layer<'a, 'b>(
    input: &'a Matrix, hidden_weights: &'a Matrix,
    previous_hidden: &'b Matrix,
    recurrent_weights: &'b Matrix) -> Matrix {
    // hidden layer forward
    let current_op = match input * hidden_weights {
        Ok(v) => v, Err(e) => panic!(e),
    };
    // recurrent step
    let recurrent_op = match previous_hidden * recurrent_weights {
        Ok(v) => v, Err(e) => panic!(e),
    };
    (current_op + &recurrent_op).sigmoid()
}


struct IntGenerator {
    rng: rand::ThreadRng,
    range: Range<i32>,
}

impl IntGenerator {
    pub fn new(from: i32, to: i32) -> IntGenerator {
        IntGenerator {
            rng: rand::thread_rng(),
            range: Range::new(from, to),
        }
    }
    pub fn next_number(&mut self) -> i32 {
        let rng = &mut self.rng;
        self.range.ind_sample(rng)
    }
}


fn main() {
    let mut int_gen = IntGenerator::new(0, 128);

    let mut hidden_layer_weights = Matrix::new_random(INPUT_SIZE, HIDDEN_SIZE);
    let mut output_layer_weights = Matrix::new_random(HIDDEN_SIZE, OUTPUT_SIZE);
    let mut recurrent_layer_weights = Matrix::new_random(
        HIDDEN_SIZE, HIDDEN_SIZE);

    let mut hidden_layer_update = Matrix::new(INPUT_SIZE, HIDDEN_SIZE);
    let mut output_layer_update = Matrix::new(HIDDEN_SIZE, OUTPUT_SIZE);
    let mut recurrent_layer_update = Matrix::new(HIDDEN_SIZE, HIDDEN_SIZE);

    for iteration in 0..1 {
        let a = int_gen.next_number();
        let a_bits = get_bits(a);
        let b = int_gen.next_number();
        let b_bits = get_bits(b);
        let c = a + b;
        let c_bits = get_bits(c);

        let mut prediction: [i32; BIT_COUNT] = [0; BIT_COUNT];
        let mut overall_error: f64 = 0.0;

        let mut output_deltas: Vec<Matrix> = vec![];
        let mut hidden_layer_history: Vec<Matrix> = Default::default();
        hidden_layer_history.push(Matrix::new(1, HIDDEN_SIZE));

        let bits_iter = (0..BIT_COUNT).rev()
            .map(|i| (i, a_bits[i], b_bits[i], c_bits[i]));

        for (bit_pos, a_bit, b_bit, c_bit) in bits_iter {
            println!("a: {}, b: {}", a_bit, b_bit);
            let input_layer = Matrix::from(
                1, INPUT_SIZE, &[a_bit as f64, b_bit as f64]);
            let labels = Matrix::from(1, 1, &[c_bit as f64]);

            let hidden_layer = forward_hidden_layer(
                &input_layer, &hidden_layer_weights,
                hidden_layer_history.last().unwrap(),
                &recurrent_layer_weights);

            // output layer forward
            let output_layer_op = match &hidden_layer * &output_layer_weights {
                Ok(v) => v, Err(e) => panic!(e)
            };
            let output_layer = output_layer_op.sigmoid();

            let output_error = labels - &output_layer;
            overall_error += output_error[(0, 0)].abs();

            // set the predicted bit
            prediction[bit_pos] = output_layer[(0, 0)].round() as i32;

            // save the deltaas for each timestep
            let output_derivative = output_layer.sigmoid_derivative();
            let output_delta = match output_error * &output_derivative {
                Ok(v) => v, Err(e) => panic!(e),
            };
            output_deltas.push(output_delta);

            // store the output of the hidden layer for the next timestep
            hidden_layer_history.push(hidden_layer.clone());
        }

        let mut next_hidden_layer_delta = Matrix::new(1, HIDDEN_SIZE);
        for i in 0..BIT_COUNT {
            let inputs = Matrix::from(
                2, 1, &[a_bits[i] as f64, b_bits[i] as f64]);
            let hidden_layer_index = hidden_layer_history.len() - i - 1;
            let hidden_layer = &hidden_layer_history[hidden_layer_index];
            let prev_hidden_layer = &hidden_layer_history[
                hidden_layer_index - 1];

            let output_delta = &output_deltas[output_deltas.len() - i - 1];
            let recurrent_dot = {
                let delta = &next_hidden_layer_delta;
                let weights = &recurrent_layer_weights;
                let trans = weights.transpose();
                match delta * trans {
                    Ok(v) => v, Err(e) => panic!(e),
                }
            };
            let hidden_layer_delta = {
                let weights = &hidden_layer_weights;
                let trans = weights.transpose();
                println!("{:?}\n{:?}", output_delta, trans);
                let output_delta_dot = match output_delta * trans {
                    Ok(v) => v, Err(e) => panic!(e)
                };
                let hidden_differential = hidden_layer.sigmoid_derivative();
                (recurrent_dot + &output_delta_dot) * &hidden_differential
            }.unwrap();

            output_layer_update +=
                match hidden_layer.transpose() * output_delta {
                    Ok(v) => v, Err(e) => panic!(e)
                };
            recurrent_layer_update +=
                match prev_hidden_layer.transpose() * &hidden_layer_delta {
                    Ok(v) => v, Err(e) => panic!(e)
                };
            hidden_layer_update +=
                match inputs.transpose() * &hidden_layer_delta {
                    Ok(v) => v, Err(e) => panic!(e)
                };

            next_hidden_layer_delta = hidden_layer_delta;
        }

        hidden_layer_weights += &hidden_layer_update;
        output_layer_weights += &output_layer_update;
        recurrent_layer_weights += &recurrent_layer_update;

        hidden_layer_weights *= 0.0;
        output_layer_weights *= 0.0;
        recurrent_layer_weights *= 0.0;

        if iteration % 1000 == 0 {
            println!("Error: {}", overall_error);
            println!("Pred: {:?}", prediction);
            println!("True: {:?}", c_bits);
        }
    }
}
