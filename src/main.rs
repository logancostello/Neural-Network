const weight_1_1: u32 = 5;
const weight_2_1: u32 = 3;
const weight_1_2: u32 = 7;
const weight_2_2: u32 = 1;

const bias_1: u32 = 3;
const bias_2: u32 = 4;

fn main() {

    println!("{}", classify(3, 2));   

}

fn classify(x: u32, y: u32) -> u32 {
    let output1 = x * weight_1_1 + y * weight_2_1 + bias_1;
    let output2 = x * weight_1_2 + y * weight_2_2 + bias_2;

    if output1 >= output2 { return 0 } else { 1 }

}
