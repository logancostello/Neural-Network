mod layer;
mod neural_network;

const WEIGHT_1_1: u32 = 5;
const WEIGHT_2_1: u32 = 3;
const WEIGHT_1_2: u32 = 7;
const WEIGHT_2_2: u32 = 1;

const BIAS_1: u32 = 3;
const BIAS_2: u32 = 2;

fn main() {

    println!("{}", classify(0, 0));   

}

fn classify(x: u32, y: u32) -> u32 {
    let output1 = x * WEIGHT_1_1 + y * WEIGHT_2_1 + BIAS_1;
    let output2 = x * WEIGHT_1_2 + y * WEIGHT_2_2 + BIAS_2;

    if output1 >= output2 { return 0 } else { 1 }

}
