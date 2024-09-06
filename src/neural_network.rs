use crate::layer::Layer;
use crate::data_point::DataPoint;

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {

    // Create new Neural Network
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    // Run the inputs through the network to get the outputs
    pub fn calculate_outputs(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut inputs_for_next_layer: Vec<f32> = inputs.clone();
        for layer in &self.layers {
            // The outputs of one layer are the inputs for the next layer
            inputs_for_next_layer = layer.calculate_outputs(&inputs_for_next_layer);
        }
        inputs_for_next_layer
    }

    // Calculate the loss for a given datapoint
    pub fn loss(&self, data: &Vec<DataPoint>) -> f32 {
        let mut loss: f32 = 0.0;

        // Get loss for each DataPoint
        for dp in data {
            let outputs: Vec<f32> = self.calculate_outputs(&dp.inputs);
            
            // Add error of each node
            for i in 0..outputs.len() {
                let error = outputs[i] - dp.expected_outputs[i];
                loss += error * error;
            }
        }

        // Return average loss for consistency across varying amounts of data
        loss / data.len() as f32
    }

    // Update all weights and biases in all layers
    pub fn apply_all_gradients(&mut self, learn_rate: f32) {
        for mut layer in &mut self.layers {
            layer.apply_gradients(learn_rate);
        }
    }
} 

// Indicate class by returning the index of the greatest output
pub fn classify(nn_outputs: &Vec<f32>) -> usize {
    nn_outputs.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;

    // Test two inputs to two outputs
    #[test]
    fn test_classify_1() {
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![5.0, 6.0];
        let layer = Layer::new_manual(weights, biases);
        let network = NeuralNetwork::new(vec![layer]);
        let inputs: Vec<f32> = vec![1.0, 2.0];
        assert_eq!(1, classify(&network.calculate_outputs(&inputs)));
    } 

    // Test loss function
    #[test]
    fn test_loss_1() {
        let data = vec![DataPoint::new(vec![0.0, 0.0], vec![0.0, 0.0])];
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![20.0, 20.0];
        let layer = Layer::new_manual(weights, biases);
        let network = NeuralNetwork::new(vec![layer]);

        assert_eq!(network.loss(&data), 2.0);
    }
}