mod layer;
mod neural_network;
mod data_point;

use crate::data_point::DataPoint;
use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;
use rand::Rng;

fn main() {
    let learn_rate = 0.25;
    let nodes_per_layer = vec![2, 5, 2];
    run_2d_curved_test(learn_rate, nodes_per_layer);
}

// Generates n points with two populations, indicating whether they are above or below the given line
// Meant to be a very simple test for the neural network
fn generate_linear_test(n: usize, slope: f32, intercept: f32) -> Vec<DataPoint> {
    let mut training_data: Vec<DataPoint> = vec![];
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let x = rng.gen_range(0.0..10.0);
        let y = rng.gen_range(0.0..10.0);

        let expected_output: Vec<usize> = if y >= slope * x + intercept {
            vec![1, 0]
        } else {
            vec![0, 1]
        };

        training_data.push(DataPoint::new(vec![x, y], expected_output));
    }
    training_data
}

// Generates n points with two populations seperated by a curved 2d line
fn generate_curved_test(n: usize) -> Vec<DataPoint> {
    let mut training_data: Vec<DataPoint> = vec![];
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let x: f32 = rng.gen_range(0.0..10.0);
        let y: f32 = rng.gen_range(0.0..10.0);

        let expected_output: Vec<usize> = if y >= x.powf(2.0) {
            vec![1, 0]
        } else {
            vec![0, 1]
        };

        training_data.push(DataPoint::new(vec![x, y], expected_output));
    }
    training_data
}

// Test if neural network can learn to identify 2d points separated by a linear line
fn run_2d_linear_test(learn_rate: f32, nodes_per_layer: Vec<usize>) {
    let mut neural_network = NeuralNetwork::new(nodes_per_layer);
    let training_data = generate_linear_test(100, -1.0, 4.0);
    let mut num = 1;
    println!("Loss: {}, Accuracy: {}", neural_network.loss(&training_data), neural_network.accuracy(&training_data));
    while neural_network.accuracy(&training_data) < 1.0 {
        neural_network.learn(&training_data, learn_rate);
        println!("{num}: Loss: {}, Accuracy: {}", neural_network.loss(&training_data), neural_network.accuracy(&training_data));
        num += 1;
    }
}

// Test if neural network can learn to identify 2d points separated by a curved line
fn run_2d_curved_test(learn_rate: f32, nodes_per_layer: Vec<usize>) {
    let mut neural_network = NeuralNetwork::new(nodes_per_layer);
    let training_data = generate_curved_test(500);
    let mut num = 1;
    println!("Loss: {}, Accuracy: {}", neural_network.loss(&training_data), neural_network.accuracy(&training_data));
    while neural_network.accuracy(&training_data) < 1.0 {
        neural_network.learn(&training_data, learn_rate);
        println!("{num}: Loss: {}, Accuracy: {}", neural_network.loss(&training_data), neural_network.accuracy(&training_data));
        num += 1;
    }
}
