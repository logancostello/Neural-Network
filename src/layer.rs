use ndarray::{Array2, Array1};

// Represents a single layer in a neural network
pub struct Layer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub is_hidden: bool,
    pub loss_gradient_weights: Array2<f64>,
    pub loss_gradient_biases: Array1<f64>
}

impl Layer {

    // Create new Layer with randomly initialized weights
    pub fn new(num_nodes_in: usize, num_nodes_out: usize, is_hidden: bool) -> Self {
        Layer {
            weights: initialize_weights(num_nodes_in, num_nodes_out, is_hidden),
            biases: Array1::zeros(num_nodes_out),
            loss_gradient_weights: Array2::zeros((num_nodes_out, num_nodes_in)),
            loss_gradient_biases: Array1::zeros(num_nodes_out),
            nodes_in: num_nodes_in,
            nodes_out: num_nodes_out,
            is_hidden: is_hidden,
        }
    }

    // Loop through all of the inputs and calculate the outputs
    // Additionally return the inputs and outputs for backpropagation
    pub fn calculate_outputs(&self, inputs: &Vec<f64>) -> (Vec<f64>, (Vec<f64>, Vec<f64>)) {
        let mut activations: Vec<f64> = vec![0.0; self.nodes_out];
        let mut outputs: Vec<f64> = vec![0.0; self.nodes_out];
        for i in 0..self.nodes_out {
            let mut output = self.biases[i];
            for j in 0..self.nodes_in {
                output += inputs[j] * self.weights[(i, j)];
            }
            outputs[i] = output;
            activations[i] = self.activation_function(output);
        }
        // Save the inputs for backpropagation
        (activations, (inputs.clone(), outputs))
    }

    // Adjust weights by the gradient times the learn rate. Reset gradients afterwords
    pub fn apply_gradients(&mut self, learn_rate: f64, batch_size: usize) {
        for i in 0..self.nodes_out {
            self.biases[i] -= learn_rate * (self.loss_gradient_biases[i] / batch_size as f64);
            self.loss_gradient_biases[i] = 0.0;
            for j in 0..self.nodes_in {
                self.weights[(i, j)] -= learn_rate * (self.loss_gradient_weights[(i, j)] / batch_size as f64);
                self.loss_gradient_weights[(i, j)] = 0.0;
            }
        }
    }

    // Use ReLU for hidden layers, Sigmoid for final layer
    pub fn activation_function(&self, output: f64) -> f64 {
        if self.is_hidden {
            return (output + output.abs()) / 2.0
        }
        // Handle negative values that cause Nan
        if output < -709.78 {
            return 0.0; 
        }
    
        1.0 / (1.0 + (-output).exp())
    }

    // Derivative with respect to the output (pre-activation value)
    pub fn activation_derivative(&self, output: f64) -> f64 {
        if self.is_hidden {
            return if output > 0.0 {1.0} else {0.0}
        }
        let sigmoid_output = 1.0 / (1.0 + (-output).exp());
        sigmoid_output * (1.0 - sigmoid_output)
    }

    // The first step in backpropagation is updating the gradient of the final layer
    pub fn update_final_layer_gradient(&mut self, predicted: &Vec<f64>, expected: &Vec<usize>, inputs: &Vec<f64>, outputs: &Vec<f64>) -> Vec<f64> {
        let mut propagated_values: Vec<f64> = vec![0.0; self.biases.len()];
        for i in 0..self.nodes_out {

            // Calculate and store values that will be propagated
            let loss_derivative = loss_derivative(predicted[i], expected[i] as f64);
            let activation_derivative = self.activation_derivative(outputs[i]);
            propagated_values[i] = loss_derivative * activation_derivative;

            // Update gradient of biases (derivative of biases is 1)
            self.loss_gradient_biases[i] += propagated_values[i];

            // Update gradient of weights (derivative of weights is the input value)
            for j in 0..self.nodes_in {
                self.loss_gradient_weights[(i, j)] += propagated_values[i] * inputs[j];
            }
        }
        // Return propagated values for the next layers to use
        propagated_values
    }
}

// Use He Initialization for hidden layers, Xavier/Glorot Initialization for final layer
fn initialize_weights(num_nodes_in: usize, num_nodes_out: usize, is_hidden: bool) ->  Array2<f64> {
    let limit: f64 = if is_hidden {
        (2.0 / num_nodes_in as f64).sqrt() // He 
    } else {
        (6.0 / (num_nodes_in + num_nodes_out) as f64).sqrt() // Xavier/Glorot
    };
    let mut weights: Array2<f64> = Array2::zeros((num_nodes_out, num_nodes_in));

    for i in 0..num_nodes_out {
        for j in 0..num_nodes_in {
            weights[(i, j)] = rand::random::<f64>() * 2.0 * limit - limit;
        }
    }
    weights
}

// With respect to the calculated output
pub fn loss_derivative(predicted: f64, expected: f64) -> f64 {
    predicted - expected
}