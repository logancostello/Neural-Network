use ndarray::{Array2, Array1};

// Data that the network can be trained on
pub struct DataPoint {
    pub inputs: Array1<f64>,
    pub expected_outputs: Array1<usize>
}

impl DataPoint {

    // Create new DataPoint
    pub fn new(inputs: Vec<f64>, expected_outputs: Vec<usize>) -> Self {
        DataPoint { 
            inputs: Array1::from_vec(inputs),
             expected_outputs: Array1::from_vec(expected_outputs)
        }
    }
}