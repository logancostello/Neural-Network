mod layer;
mod neural_network;
mod data_point;

use crate::data_point::DataPoint;
use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;
use rand::Rng;

fn main() {
    // Test if neural network can learn to identify 2d points separated by a linear line
    let layer = Layer::new(2, 2);
    let mut neural_network = NeuralNetwork::new(vec![layer]);
    let training_data = generate_linear_test(100, -1.0, 4.0);
    println!("Original loss: {}", neural_network.loss(&training_data));
    for i in 0..500 {
        neural_network.learn(&training_data, 0.2);
        println!("{i}: Updated loss: {}", neural_network.loss(&training_data));
    }
}

// Generates n points with two populations, indicating whether they are above or below the given line
// Meant to be a very simple test for the neural network
fn generate_linear_test(n: usize, slope: f32, intercept: f32) -> Vec<DataPoint> {
    let mut training_data: Vec<DataPoint> = vec![];
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let x = rng.gen_range(0.0..10.0);
        let y = rng.gen_range(0.0..10.0);

        let expected_output: Vec<f32> = if y >= slope * x + intercept {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };

        training_data.push(DataPoint::new(vec![x, y], expected_output));
    }
    training_data
}
