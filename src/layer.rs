// Represents a single layer in a neural network
pub struct Layer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub is_hidden: bool,

    // Stored for backpropagation
    pub inputs: Vec<f64>, 
    pub outputs: Vec<f64>,
    pub propagated_values: Vec<f64>,
    pub loss_gradient_weights: Vec<Vec<f64>>,
    pub loss_gradient_biases: Vec<f64>
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
            inputs: vec![0.0; num_nodes_in],
            outputs: vec![0.0; num_nodes_out],
            propagated_values: vec![0.0; num_nodes_out]
        }
    }

    // Loop through all of the inputs and calculate the outputs
    pub fn calculate_outputs(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut activations: Vec<f64> = Vec::new();
        let mut outputs: Vec<f64> = Vec::new();
        for i in 0..self.nodes_out {
            let mut output = self.biases[i];
            for j in 0..self.nodes_in {
                output += inputs[j] * self.weights[i][j];
            }
            outputs.push(output);
            activations.push(self.activation_function(output));
        }
        // Save the inputs for backpropagation
        self.inputs = inputs.clone(); 
        self.outputs = outputs;
        activations
    }

    // Adjust weights by the gradient times the learn rate. Reset gradients afterwords
    pub fn apply_gradients(&mut self, learn_rate: f64, batch_size: usize) {
        for i in 0..self.nodes_out {
            self.biases[i] -= learn_rate * (self.loss_gradient_biases[i] / batch_size as f64);
            self.loss_gradient_biases[i] = 0.0;
            for j in 0..self.nodes_in {
                self.weights[i][j] -= learn_rate * (self.loss_gradient_weights[i][j] / batch_size as f64);
                self.loss_gradient_weights[i][j] = 0.0;
            }
        }
    }

    // Use ReLU for hidden layers, Sigmoid for final layer
    pub fn activation_function(&self, output: f64) -> f64 {
        if self.is_hidden {
            return (output + output.abs()) / 2.0
        }
        1.0 / (1.0 + (-output).exp())
    }

    // Derivative with respect to the output (pre-activation value)
    pub fn activation_derivative(&self, output: f64) -> f64 {
        if self.is_hidden {
            return (output + output.abs()) / (2.0 * output.abs()) 
        }
        (-output).exp() / (1.0 + (-output).exp()).powf(2.0)
    }

    // The first step in backpropagation is updating the gradient of the final layer
    pub fn update_final_layer_gradient(&mut self, predicted: &Vec<f64>, expected: &Vec<usize>) {
        for i in 0..self.nodes_out {

            // Calculate and store values that will be propagated
            let loss_derivative = loss_derivative(predicted[i], expected[i] as f64);
            let activation_derivative = self.activation_derivative(self.outputs[i]);
            self.propagated_values[i] = loss_derivative * activation_derivative;

            // Update gradient of biases (derivative of biases is 1)
            self.loss_gradient_biases[i] += self.propagated_values[i];

            // Update gradient of weights (derivative of weights is the input value)
            for j in 0..self.nodes_in {
                self.loss_gradient_weights[i][j] += self.propagated_values[i] * self.inputs[j];
            }
        }
    }
}

// Use He Initialization for hidden layers, Xavier/Glorot Initialization for final layer
fn initialize_weights(num_nodes_in: usize, num_nodes_out: usize, is_hidden: bool) ->  Vec<Vec<f64>> {
    let limit: f64 = if is_hidden {
        (2.0 / num_nodes_in as f64).sqrt() // He 
    } else {
        (6.0 / (num_nodes_in + num_nodes_out) as f64).sqrt() // Xavier/Glorot
    };
    let mut weights: Vec<Vec<f64>> = vec![vec![0.0; num_nodes_in]; num_nodes_out];
    for i in 0..num_nodes_out {
        for j in 0..num_nodes_in {
            weights[i][j] = rand::random::<f64>() * 2.0 * limit - limit;
        }
    }
    weights
}

// With respect to the calculated output
pub fn loss_derivative(predicted: f64, expected: f64) -> f64 {
    // 2.0 * (predicted - expected)
    predicted - expected
}