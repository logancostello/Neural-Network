// Represents a single layer in a neural network
pub struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    loss_gradient_weights: Vec<Vec<f32>>,
    loss_gradient_biases: Vec<f32>
}

impl Layer {

    // Create new Layer with manually set weights and biases
    pub fn new_manual(weights: Vec<Vec<f32>>, biases:Vec<f32>) -> Self {
        Layer {
            loss_gradient_weights: vec![vec![0.0; weights[0].len()]; weights.len()],
            loss_gradient_biases: vec![0.0; biases.len()],
            weights,
            biases
        }
    }

    // Create new Layer with randomly initialized weights
    pub fn new (num_nodes_in: usize, num_nodes_out: usize) -> Self {
        Layer {
            weights: initialize_weights(num_nodes_in, num_nodes_out),
            biases: vec![0.0; num_nodes_out],
            loss_gradient_weights: vec![vec![0.0; num_nodes_in]; num_nodes_out],
            loss_gradient_biases: vec![0.0; num_nodes_out]
        }

    }

    // Loop through all of the inputs and calculate the outputs
    pub fn calculate_outputs(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut activations: Vec<f32> = Vec::new();
        for i in 0..inputs.len() {
            let mut weighted_input = self.biases[i];
            for j in 0..self.weights.len() {
                weighted_input += inputs[j] * self.weights[i][j];
            }
            activations.push(activation_function(weighted_input));
        }
        activations
    }
}

// Using the Sigmoid Activation Function
fn activation_function(weighted_input: f32) -> f32 {
    1.0 / (1.0 + (-weighted_input).exp())
}

// Initialize weights using Xavier/Glorot Initialization
fn initialize_weights(num_nodes_in: usize, num_nodes_out: usize) ->  Vec<Vec<f32>> {
    let limit: f32 = (6.0 / (num_nodes_in + num_nodes_out) as f32).sqrt();
    let mut weights: Vec<Vec<f32>> = vec![vec![0.0; num_nodes_in]; num_nodes_out];
    for i in 0..num_nodes_out {
        for j in 0..num_nodes_in {
            weights[i][j] = rand::random::<f32>() * 2.0 * limit - limit;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test two inputs to two outputs
    #[test]
    fn test_calculate_outputs_1() {
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![5.0, 6.0];
        let layer = Layer::new_manual(weights, biases);
        let inputs: Vec<f32> = vec![1.0, 2.0];
        let outputs = layer.calculate_outputs(&inputs);
        assert_eq!(outputs[0] > 0.99, true);
        assert_eq!(outputs[1] > 0.99, true);
    } 

    // Test activation function has effect
    #[test]
    fn test_calculate_outputs_2() {
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![0.0, 0.0];
        let layer = Layer::new_manual(weights, biases);
        let inputs: Vec<f32> = vec![-1.0, -2.0];
        let outputs = layer.calculate_outputs(&inputs);
        assert_eq!(outputs[0] < 0.1, true);
        assert_eq!(outputs[1] < 0.1, true);
    } 
}