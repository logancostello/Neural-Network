// Data that the network can be trained on
pub struct DataPoint {
    pub inputs: Vec<f32>,
    pub expected_outputs: Vec<f32>
}

impl DataPoint {

    // Create new DataPoint
    pub fn new(inputs: Vec<f32>, expected_outputs:Vec<f32>) -> Self {
        DataPoint { inputs, expected_outputs }
    }
}