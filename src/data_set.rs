use crate::data_point::DataPoint;

pub struct DataSet {
    pub train: Vec<DataPoint>,
    pub test: Vec<DataPoint>
}

impl DataSet {

    // Input data must be random
    pub fn new (mut data: Vec<DataPoint>, test_pct: f32) -> Self {
        let split_index = (data.len() as f32 * test_pct).ceil() as usize;
        let train = data.split_off(split_index);

        DataSet { train, test: data }
    }
}