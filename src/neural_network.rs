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
    pub fn classify(&self, inputs: &Vec<f32>) -> usize {
        let mut inputs_for_next_layer: Vec<f32> = inputs.clone();
        for layer in &self.layers {
            // The outputs of one layer are the inputs for the next layer
            inputs_for_next_layer = layer.calculate_outputs(&inputs_for_next_layer);
        }
        // Indicate class by returning the index of the greatest output
        inputs_for_next_layer.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
    }
} 


#[cfg(test)]
mod tests {
    use super::*;

    // Test two inputs to two outputs
    #[test]
    fn test_classify_1() {
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![5.0, 6.0];
        let layer = Layer::new(weights, biases);
        let network = NeuralNetwork::new(vec![layer]);
        let inputs: Vec<f32> = vec![1.0, 2.0];
        assert_eq!(1, network.classify(&inputs));
    } 
}