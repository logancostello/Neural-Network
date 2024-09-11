use crate::layer::Layer;
use crate::data_point::DataPoint;

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {

    // Creates a new Neural Network from list of nodes per layer
    pub fn new(nodes_per_layer: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = vec![];
        for i in 1..nodes_per_layer.len() {
            let is_hidden: bool = if i == nodes_per_layer.len() - 1 { false } else { true };
            layers.push(Layer::new(nodes_per_layer[i - 1], nodes_per_layer[i], is_hidden))
        }
        NeuralNetwork { layers }
    }

    // Run the inputs through the network to get the outputs
    pub fn calculate_outputs(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut inputs_for_next_layer: Vec<f64> = inputs.clone();
        for layer in &mut self.layers {
            // The outputs of one layer are the inputs for the next layer
            inputs_for_next_layer = layer.calculate_outputs(&inputs_for_next_layer);
        }
        inputs_for_next_layer
    }

    // Calculate the loss for a given dataset
    pub fn loss(&mut self, data: &Vec<DataPoint>) -> f64 {
        let mut loss: f64 = 0.0;

        // Get loss for each DataPoint
        for dp in data {
            let outputs: Vec<f64> = self.calculate_outputs(&dp.inputs);
            
            // Add error of each node
            for i in 0..outputs.len() {
                let error = outputs[i] - dp.expected_outputs[i] as f64;
                loss += error * error;
            }
        }

        // Return average loss for consistency across varying amounts of data
        loss / data.len() as f64
    }

    // Calculate accuracy for a given dataset
    pub fn accuracy(&mut self, data: &Vec<DataPoint>) -> f64 {
        let mut num_correct = 0.0;

        // Check expected class is the same as the predicted class
        for dp in data {
            let outputs = self.calculate_outputs(&dp.inputs);
            let predicted_class: usize = classify(&outputs);
            if dp.expected_outputs[predicted_class] == 1 {
                num_correct += 1.0;
            }
        }
        num_correct / data.len() as f64
    }

    // Run a single iteration of Gradient Descent
    pub fn learn2(&mut self, training_data: &Vec<DataPoint>, learn_rate: f64) {
        let h: f64 = 0.0001; // A small step to get the slope
        let original_loss: f64 = self.loss(training_data);
        
        for l in 0..self.layers.len() { // Looping over the index avoids issues raised by the borrow checker
            
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
        // self.apply_all_gradients(learn_rate);
    }

    // Run a single iteration of Gradient Descent via backpropagation
    pub fn learn(&mut self, training_data: &Vec<DataPoint>, learn_rate: f64) {
        for dp in training_data {
            self.update_gradients(&dp);
        }
        self.apply_all_gradients(learn_rate, training_data.len());
    }

    // Update all weights and biases in all layers
    pub fn apply_all_gradients(&mut self, learn_rate: f64, batch_size: usize) {
        for layer in &mut self.layers {
            layer.apply_gradients(learn_rate, batch_size);
        }
    }

    // Update gradients using backpropagation for a single point
    pub fn update_gradients(&mut self, datapoint: &DataPoint) {

        // Run the point through the network, storing the information we need for backpropagation
        let predicted = self.calculate_outputs(&datapoint.inputs);

        // Update gradient of the final layer
        let final_layer_index: usize = self.layers.len() - 1;
        self.layers[final_layer_index].update_final_layer_gradient(&predicted, &datapoint.expected_outputs);
        
        // Update the gradients of all of the hidden layers 
        for index in (0..final_layer_index).rev() {
            self.update_hidden_layer_gradient(index);
        }
    }

    // Update the gradients of the given layer by using the propagated values from the following layers
    // Ideally this could be a Layer method, there becomes ownership issues when the layer needs the values from the following layer
    pub fn update_hidden_layer_gradient(&mut self, layer_index: usize) {
        for i in 0..self.layers[layer_index].nodes_out {

            // Calculate and store values that will be propagated
            let activation_derivative = self.layers[layer_index].activation_derivative(self.layers[layer_index].outputs[i]);
            let mut following_layer_values = 0.0;
            for j in 0..self.layers[layer_index + 1].nodes_out {
                following_layer_values += self.layers[layer_index + 1].propagated_values[j] * self.layers[layer_index + 1].weights[j][i];
            }
            self.layers[layer_index].propagated_values[i] = activation_derivative * following_layer_values;

            // Update gradient of biases (derivative of biases is 1)
            self.layers[layer_index].loss_gradient_biases[i] += self.layers[layer_index].propagated_values[i];

            // Update gradient of weights (derivative of weights is the input value)
            for j in 0..self.layers[layer_index].nodes_in {
                self.layers[layer_index].loss_gradient_weights[i][j] += self.layers[layer_index].propagated_values[i] * self.layers[layer_index].inputs[j];
            }
        }
    }

} 

// Indicate class by returning the index of the greatest output
pub fn classify(nn_outputs: &Vec<f64>) -> usize {
    // println!("{} {}", nn_outputs[0], nn_outputs[1]);
    nn_outputs.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
}