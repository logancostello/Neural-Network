struct DataSet {
    pub train: Vec<DataPoint>,
    pub test: Vec<DataPoint>
}

impl DataSet {

    // Input data must be random
    pub fn new (data: Vec<DataPoint>, test_pct: f32) -> Self {
        let split_index = (vec.len() as f32 * percent).ceil() as usize;
        let (train, test) = vec.split_at(split_index);

        DataSet { train, test }
    }
}