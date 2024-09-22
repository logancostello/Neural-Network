mod layer;
mod neural_network;
mod data_point;
mod data_set;

use crate::data_point::DataPoint;
use crate::layer::Layer;
use crate::neural_network::NeuralNetwork;
use crate::data_set::DataSet;
use rand::Rng;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::error::Error;
use ndarray::{Array2, Array1};


fn main() {
    let learn_rate = 0.01;
    let batch_size = 64;
    let nodes_per_layer = vec![784, 128, 64, 10];
    match get_mnist() {
        Some(mut dataset) => run(learn_rate, nodes_per_layer, batch_size, &mut dataset),
        None => println!("No data"),
    }

    // let learn_rate = 0.25;
    // let batch_size = 50;
    // let nodes_per_layer = vec![2, 5, 2];
    // run_2d_curved_test(learn_rate, nodes_per_layer, batch_size)
}

// Run network on given data
fn run(learn_rate: f64, nodes_per_layer: Vec<usize>, batch_size: usize, dataset: &mut DataSet) {
    let mut neural_network = NeuralNetwork::new(nodes_per_layer);
    let mut num = 1;
    println!("Epoch {num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
    while neural_network.accuracy(&dataset.train) < 1.0 {
        neural_network.learn(&mut dataset.train, learn_rate, batch_size);
        println!("Epoch {num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
        num += 1;
    }
}

// Generates n points with two populations, indicating whether they are above or below the given line
// Meant to be a very simple test for the neural network
fn generate_linear_test(n: usize, slope: f64, intercept: f64) -> Vec<DataPoint> {
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
        let x: f64 = rng.gen_range(0.0..10.0);
        let y: f64 = rng.gen_range(0.0..10.0);

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
fn run_2d_linear_test(learn_rate: f64, nodes_per_layer: Vec<usize>, batch_size: usize) {
    let mut neural_network = NeuralNetwork::new(nodes_per_layer);
    let data = generate_linear_test(100, -1.0, 4.0);
    let mut dataset = DataSet::new(data, 0.2);
    let mut num = 1;
    println!("{num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
    while neural_network.accuracy(&dataset.train) < 1.0 {
        neural_network.learn(&mut dataset.train, learn_rate, batch_size);
        println!("{num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
        num += 1;
    }
}

// Test if neural network can learn to identify 2d points separated by a curved line
fn run_2d_curved_test(learn_rate: f64, nodes_per_layer: Vec<usize>, batch_size: usize) {
    let mut neural_network = NeuralNetwork::new(nodes_per_layer);
    let data = generate_curved_test(500);
    let mut dataset = DataSet::new(data, 0.2);
    let mut num = 1;
    println!("Epoch {num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
    while neural_network.accuracy(&dataset.train) < 1.0 {
        neural_network.learn(&mut dataset.train, learn_rate, batch_size);
        println!("Epoch {num}. Loss: {:.6}, Train: {:.4}, Test: {:.4}", neural_network.loss(&dataset.train), neural_network.accuracy(&dataset.train), neural_network.accuracy(&dataset.test));
        num += 1;
    }
}

// Reads the MNIST dataset files, making them ready for the network to read
fn get_mnist() -> Option<DataSet> {
    let train_images = get_mnist_images("mnist/train-images-idx3-ubyte");
    let test_images = get_mnist_images("mnist/t10k-images-idx3-ubyte");
    let train_labels = get_mnist_labels("mnist/train-labels-idx1-ubyte");
    let test_labels = get_mnist_labels("mnist/t10k-labels-idx1-ubyte");


    match (train_images, test_images, train_labels, test_labels) {
        (Ok(mut train_images), Ok(mut test_images), Ok(mut train_labels), Ok(mut test_labels)) => {
            println!("Successfully read mnist");
            let mut train_data: Vec<DataPoint> = vec![];
            let mut test_data: Vec<DataPoint> = vec![];
            while train_images.len() > 0 {
                let input = train_images.pop().unwrap().iter().map(|&x| x as f64).collect();
                let expected_output = train_labels.pop().unwrap().iter().map(|&x| x as usize).collect();
                train_data.push(DataPoint::new(input, expected_output))
            }

            while test_images.len() > 0 {
                let input = test_images.pop().unwrap().iter().map(|&x| x as f64).collect();
                let expected_output = test_labels.pop().unwrap().iter().map(|&x| x as usize).collect();
                test_data.push(DataPoint::new(input, expected_output))
            }
            Some(DataSet{train: train_data, test: test_data})
        }
        (Err(e), _, _, _) => {
            println!("Error reading training images: {}", e);
            None
        }
        (_, Err(e), _, _) => {
            println!("Error reading test images: {}", e);
            None
        },
        (_, _, Err(e), _) => {
            println!("Error reading training labels: {}", e);
            None
        },
        (_, _, _, Err(e)) => {
            println!("Error reading test labels: {}", e);
            None
        },
    }
}

// Reads the MNIST images
fn get_mnist_images(file_path: &str) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    // Open the file
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    // Read the header
    let mut header = [0u8; 16];
    reader.read_exact(&mut header)?;

    // Unpack header
    let magic_number = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]);

    // Validate magic number
    if magic_number != 2051 {
        return Err("Invalid magic number".into());
    }

    // Read image data
    let num_images = num_images as usize;
    let rows = rows as usize;
    let cols = cols as usize;
    let mut images = vec![vec![0; rows * cols]; num_images];

    for image in images.iter_mut() {
        reader.read_exact(image)?;
    }

    Ok(images)
}

// Read the mnist labels
fn get_mnist_labels(file_path: &str) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    // Open the file
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);

    // Read the header
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;

    // Unpack header
    let magic_number = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);

    // Validate magic number
    if magic_number != 2049 {
        return Err("Invalid magic number for label file".into());
    }

    // Read label data
    let num_labels = num_labels as usize;
    let mut labels = vec![0; num_labels];

    reader.read_exact(&mut labels)?;

    // Convert labels to one-hot encoding
    let one_hot_labels: Vec<Vec<u8>> = labels
        .into_iter()
        .map(|label| {
            let mut one_hot = vec![0; 10];
            one_hot[label as usize] = 1;
            one_hot
        })
        .collect();

    Ok(one_hot_labels)
}