// Represents a single layer in a neural network
pub struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>
}

impl Layer {

    // Create new Layer
    pub fn new(weights: Vec<Vec<f32>>, biases:Vec<f32>) -> Self {
        Layer{ weights, biases }
    }

    // Loop through all of the inputs and calculate the outputs
    pub fn calculate_outputs(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut outputs: Vec<f32> = Vec::new();
        for i in 0..inputs.len() {
            let mut output = self.biases[i];
            for j in 0..self.weights.len() {
                output += inputs[j] * self.weights[i][j];
            }
            outputs.push(output);
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test two inputs to two outputs
    #[test]
    fn test_calculate_outputs() {
        let weights: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let biases: Vec<f32> = vec![5.0, 6.0];
        let layer = Layer::new(weights, biases);
        let inputs: Vec<f32> = vec![1.0, 2.0];
        let outputs = layer.calculate_outputs(&inputs);
        assert_eq!(outputs, vec![10.0, 17.0]);
    } 
}