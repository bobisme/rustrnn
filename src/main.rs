extern crate rand;

use rand::Rng;
use rand::distributions::{Range,IndependentSample};

mod utils;
#[macro_use]
mod utils_tests;
mod layer;
mod matrix;

use utils::{sigmoid, sigmoid_derivative};
use layer::{Layer};
use matrix::{Matrix};

// const LEARNING_RATE: f64 = 0.1;
const INPUT_SIZE: usize = 2;
const HIDDEN_SIZE: usize = 16;
// const OUTPUT_SIZE: u32 = 1;
const BIT_COUNT: usize = 8;


fn get_bits(i: i32) -> Vec<i32> {
    (0..BIT_COUNT).rev().map(|k| (i & (1 << k)) >> k).collect()
}


fn main() {
    let mut rng = rand::thread_rng();
    let x = 2.0;
    println!("{}", sigmoid(x));
    println!("{}", sigmoid_derivative(x));
    println!("{}", sigmoid_derivative(sigmoid(x)));
    println!("{}", rng.gen::<f64>());

    let l = Layer::new(INPUT_SIZE, HIDDEN_SIZE);
    println!("Layer is {} big", l.size() * l.input_size());
    println!("thing: {:?}", l.weights);
    println!("{:?}", Matrix::new(2, 3));

    let int_range = Range::new(0, 128);

    for _ in 0..1 {
        let a: i32 = int_range.ind_sample(&mut rng);
        let a_bits = get_bits(a);
        let b: i32 = int_range.ind_sample(&mut rng);
        let b_bits = get_bits(b);
        let c = a + b;
        let c_bits = get_bits(c);

        // let prediction: [i32; BIT_COUNT] = [0; BIT_COUNT];
        // let overall_error = 0.0;

        // let hidden_layer_deltas = ?;
        let mut input_layer_history: Vec<Vec<i32>> = Default::default();
        input_layer_history.push(vec![0, BIT_COUNT as i32]);

        for (a, b) in a_bits.iter().zip(b_bits.iter()).rev() {
            println!("a: {}, b: {}", a, b)
        }

        println!("{} + {} = {}", a, b, c);
        println!("{:b} + {:b} = {:b}", a, b, c);
        println!("{:?} + {:?} = {:?}", a_bits, b_bits, c_bits);
        println!("{:?}", (23 & (1 << 2)) >> 2);
    }
}
