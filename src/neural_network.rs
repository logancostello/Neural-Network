use crate::layer::Layer;

struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {

    // Create new Neural Network
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    // Classify the inputs by running them through the network
    pub fn classify(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut inputs_for_next_layer: Vec<f32> = inputs.clone();
        for layer in &self.layers {
            // The outputs of one layer are the inputs for the next layer
            inputs_for_next_layer = layer.calculate_outputs(&inputs_for_next_layer);
        }
        inputs_for_next_layer
    }
} 