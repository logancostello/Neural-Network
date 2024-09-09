// Represents a single layer in a neural network
pub struct Layer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub loss_gradient_weights: Vec<Vec<f32>>,
    pub loss_gradient_biases: Vec<f32>,
    pub is_hidden: bool,
    pub weighted_inputs: Vec<f32>, // stored for backpropagation
    pub propagated_values: Vec<f32> // stored for backpropagation
}

impl Layer {

    // Create new Layer with randomly initialized weights
    pub fn new(num_nodes_in: usize, num_nodes_out: usize, is_hidden: bool) -> Self {
        Layer {
            weights: initialize_weights(num_nodes_in, num_nodes_out, is_hidden),
            biases: vec![0.0; num_nodes_out],
            loss_gradient_weights: vec![vec![0.0; num_nodes_in]; num_nodes_out],
            loss_gradient_biases: vec![0.0; num_nodes_out],
            nodes_in: num_nodes_in,
            nodes_out: num_nodes_out,
            is_hidden: is_hidden,
            weighted_inputs: vec![0.0; num_nodes_in],
            propagated_values: vec![0.0; num_nodes_out]
        }

    }

    // Loop through all of the inputs and calculate the outputs
    pub fn calculate_outputs(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut activations: Vec<f32> = Vec::new();
        for i in 0..self.nodes_out {
            let mut output = self.biases[i];
            for j in 0..self.nodes_in {
                output += inputs[j] * self.weights[i][j];
            }
            activations.push(self.activation_function(output));
        }
        self.weighted_inputs = inputs.clone(); // save the inputs for backpropagation
        activations
    }

    // Adjust weights by the gradient times the learn rate
    pub fn apply_gradients(&mut self, learn_rate: f32) {
        for i in 0..self.nodes_out {
            self.biases[i] -= learn_rate * self.loss_gradient_biases[i];
            for j in 0..self.nodes_in {
                self.weights[i][j] -= learn_rate * self.loss_gradient_weights[i][j];
            }
        }
    }

    // Use ReLU for hidden layers, Sigmoid for final layer
    pub fn activation_function(&self, weighted_input: f32) -> f32 {
        if self.is_hidden {
            return (weighted_input + weighted_input.abs()) / 2.0
        }
        1.0 / (1.0 + (-weighted_input).exp())
    }
}

// Use He Initialization for hidden layers, Xavier/Glorot Initialization for final layer
fn initialize_weights(num_nodes_in: usize, num_nodes_out: usize, is_hidden: bool) ->  Vec<Vec<f32>> {
    let limit: f32 = if is_hidden {
        (2.0 / num_nodes_in as f32).sqrt() // He 
    } else {
        (6.0 / (num_nodes_in + num_nodes_out) as f32).sqrt() // Xavier/Glorot
    };
    let mut weights: Vec<Vec<f32>> = vec![vec![0.0; num_nodes_in]; num_nodes_out];
    for i in 0..num_nodes_out {
        for j in 0..num_nodes_in {
            weights[i][j] = rand::random::<f32>() * 2.0 * limit - limit;
        }
    }
    weights
}



