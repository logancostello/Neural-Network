use crate::layer::Layer;
use crate::data_point::DataPoint;

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {

    // Creates a new Neural Network with specifically given layers
    pub fn new_manual(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    // Creates a new Neural Network from list of nodes per layer
    pub fn new(nodes_per_layer: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = vec![];
        for i in 1..nodes_per_layer.len() {
            layers.push(Layer::new(nodes_per_layer[i - 1], nodes_per_layer[i]))
        }
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

    // Calculate the loss for a given dataset
    pub fn loss(&self, data: &Vec<DataPoint>) -> f32 {
        let mut loss: f32 = 0.0;

        // Get loss for each DataPoint
        for dp in data {
            let outputs: Vec<f32> = self.calculate_outputs(&dp.inputs);
            
            // Add error of each node
            for i in 0..outputs.len() {
                let error = outputs[i] - dp.expected_outputs[i] as f32;
                loss += error * error;
            }
        }

        // Return average loss for consistency across varying amounts of data
        loss / data.len() as f32
    }

    // Calculate accuracy for a given dataset
    pub fn accuracy(&self, data: &Vec<DataPoint>) -> f32 {
        let mut num_correct = 0.0;

        // Check expected class is the same as the predicted class
        for dp in data {
            let outputs = self.calculate_outputs(&dp.inputs);
            let predicted_class: usize = classify(&outputs);
            if dp.expected_outputs[predicted_class] == 1 {
                num_correct += 1.0;
            }
        }
        num_correct / data.len() as f32
    }

    // Run a single iteration of Gradient Descent
    pub fn learn(&mut self, training_data: &Vec<DataPoint>, learn_rate: f32) {
        let h: f32 = 0.0001; // A small step to get the slope
        let original_loss: f32 = self.loss(training_data);
        
        for l in 0..self.layers.len() { // Looping over the index avoids dealing with borrow checker
            
            // Calculate gradient for weights
            for i in 0..self.layers[l].nodes_out {
                for j in 0..self.layers[l].nodes_in {
                    self.layers[l].weights[i][j] += h;
                    self.layers[l].loss_gradient_weights[i][j] = (self.loss(training_data) - original_loss) / h;
                    self.layers[l].weights[i][j] -= h;
                }
            }

            // Calculate gradient for biases
            for i in 0..self.layers[l].nodes_out {
                self.layers[l].biases[i] += h;
                self.layers[l].loss_gradient_biases[i] = (self.loss(training_data) - original_loss) / h;
                self.layers[l].biases[i] -= h;
            }
        }
        self.apply_all_gradients(learn_rate);
    }

    // Update all weights and biases in all layers
    pub fn apply_all_gradients(&mut self, learn_rate: f32) {
        for layer in &mut self.layers {
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
        let network = NeuralNetwork::new_manual(vec![layer]);
        let inputs: Vec<f32> = vec![1.0, 2.0];
        assert_eq!(1, classify(&network.calculate_outputs(&inputs)));
    } 

    // Test loss function
    #[test]
    fn test_loss_1() {
        let data = vec![DataPoint::new(vec![0.0, 0.0], vec![0, 0])];
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![20.0, 20.0];
        let layer = Layer::new_manual(weights, biases);
        let network = NeuralNetwork::new_manual(vec![layer]);

        assert_eq!(network.loss(&data), 2.0);
    }
}