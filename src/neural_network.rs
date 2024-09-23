use crate::layer::Layer;
use crate::data_point::DataPoint;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use ndarray::{Array2, Array1, Axis};


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
    // Additionally record the inputs and outputs of each layer for backpropagation
    pub fn calculate_outputs(&self, inputs: &Array1<f64>) -> (Array1<f64>, Vec<(Array1<f64>, Array1<f64>)>) {
        let mut history: Vec<(Array1<f64>, Array1<f64>)> = Vec::new();
        let mut inputs_for_next_layer: Array1<f64> = inputs.clone();
        for layer in &self.layers {
            // The outputs of one layer are the inputs for the next layer
            let (outputs, layer_history) = layer.calculate_outputs(inputs_for_next_layer);
            inputs_for_next_layer = outputs;
            history.push(layer_history);
        }
        (inputs_for_next_layer, history)
    }

    // Multiclass cross entropy loss
    pub fn loss(&self, data: &Vec<DataPoint>) -> f64 {
        let total_loss: f64 = data.par_iter().map(|dp| {
            let (outputs, _) = self.calculate_outputs(&dp.inputs);
            let mut loss = 0.0;
    
            // Add loss of each node
            for i in 0..outputs.len() {
                if outputs[i] == 0.0 {
                    loss -= dp.expected_outputs[i] as f64 * (f64::EPSILON).ln();
                } else {
                    loss -= dp.expected_outputs[i] as f64 * outputs[i].ln();
                }
            }
            loss
        }).sum();
    
        // Return average loss for consistency across varying amounts of data
        total_loss / data.len() as f64
    }
    
    // Calculate accuracy for a given dataset
    pub fn accuracy(&self, data: &Vec<DataPoint>) -> f64 {
        // Uses a thread for each datapoint for efficiency when the data is large
        let num_correct: f64 = data.par_iter().map(|dp| {
            let (outputs, _) = self.calculate_outputs(&dp.inputs);
            let predicted_class: usize = classify(&outputs);
            if dp.expected_outputs[predicted_class] == 1 {
                1.0
            } else {
                0.0
            }
        }).sum();
    
        num_correct / data.len() as f64
    }
    
    // Run a single iteration of Gradient Descent via backpropagation
    pub fn learn(&mut self, training_data: &mut Vec<DataPoint>, learn_rate: f64, batch_size: usize) {

        training_data.shuffle(&mut thread_rng());
        let mini_batches: Vec<&[DataPoint]> = training_data.chunks(batch_size).collect();
    
        // Wrap `self` in Arc<Mutex<...>> for safe shared access
        let self_arc = Arc::new(Mutex::new(self));
    
        for mini_batch in mini_batches {
            // Process each data point in the mini_batch in parallel
            mini_batch.par_iter().for_each(|datapoint| {
                let self_clone = Arc::clone(&self_arc); // Clone the Arc to share it with the threads
                
                let mut instance = match self_clone.lock() {
                    Ok(instance) => instance,
                    Err(poisoned) => {
                        eprintln!("Mutex poisoned: {}", poisoned);
                        return; // Skip this iteration
                    }
                };

                instance.update_gradients(&datapoint);
            });
    
            // Lock the mutex again to apply gradients after all updates
            {
                let mut instance = self_arc.lock().unwrap();
                instance.apply_all_gradients(learn_rate, training_data.len());
            }
        }
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
        let (predicted, mut history) = self.calculate_outputs(&datapoint.inputs);

        // Update gradient of the final layer
        let final_layer_index: usize = self.layers.len() - 1;
        let (inputs, outputs) = history.pop().unwrap();
        let mut propagated = self.layers[final_layer_index].update_final_layer_gradient(&predicted, &datapoint.expected_outputs, &inputs, &outputs);
        
        // Update the gradients of all of the hidden layers 
        for index in (0..final_layer_index).rev() {
            let (inputs, outputs) = history.pop().unwrap();
            propagated = self.update_hidden_layer_gradient(index, &propagated, &inputs, &outputs);
        }
    }

    // Update the gradients of the given layer by using the propagated values from the following layers
    // Ideally this could be a Layer method, there becomes ownership issues when the layer needs the values from the following layer
    pub fn update_hidden_layer_gradient(&mut self, layer_index: usize, prev_propagated_values: &Array1<f64>, inputs: &Array1<f64>, outputs: &Array1<f64>) -> Array1<f64> {

        // Calculate and store values that will be propagated
        let activation_derivatives = outputs.mapv(|o| self.layers[layer_index].activation_derivative(o));
        let following_layer_values = self.layers[layer_index + 1].weights.t().dot(prev_propagated_values);
        let propagated_values: Array1<f64> = activation_derivatives * following_layer_values;

        // Update gradient of biases (derivative of biases is 1)
        self.layers[layer_index].loss_gradient_biases += &propagated_values;

        // Update gradient of weights (derivative of weights is the input value)
        let inputs_reshaped = inputs.view().insert_axis(Axis(0)); 
        let propagated_values_reshaped = propagated_values.view().insert_axis(Axis(1)); 
        self.layers[layer_index].loss_gradient_weights += &(propagated_values_reshaped.dot(&inputs_reshaped)); 
        
        // Return propagated values for next layer
        propagated_values
    }
} 

// Indicate class by returning the index of the greatest output
pub fn classify(nn_outputs: &Array1<f64>) -> usize {
    nn_outputs.iter().enumerate().max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
}