// Data that the network can be trained on
pub struct DataPoint {
    pub inputs: Vec<f64>,
    pub expected_outputs: Vec<usize>
}

impl DataPoint {

    // Create new DataPoint
    pub fn new(inputs: Vec<f64>, expected_outputs:Vec<usize>) -> Self {
        DataPoint { inputs, expected_outputs }
    }
}