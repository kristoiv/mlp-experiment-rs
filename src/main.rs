use rand::distributions::Distribution;
use rand_distr::Normal;
use std::ops::{Div, Mul, Neg, Sub};

mod data_loader;
mod mnist;

fn identity(size: usize) -> Vec<Vec<f32>> {
    let mut v = Vec::with_capacity(size);
    v.resize_with(size, || {
        let mut v1 = Vec::with_capacity(size);
        v1.resize(size, 0.0_f32);
        v1
    });
    v.into_iter()
        .enumerate()
        .map(|it| {
            let mut row = it.1;
            row[it.0] = 1.0;
            row
        })
        .collect()
}

fn main() {
    // Load training data
    println!("Loading training dataset");
    let (y_train, x_train) = mnist::load(true).unwrap();
    let (y_test, x_test) = mnist::load(false).unwrap();
    println!(
        "Loaded! training_labels={}, training_images={}, test_labels={}, test_images={}",
        y_train.len(),
        x_train.len(),
        y_test.len(),
        x_test.len(),
    );

    // Define model parameters
    let input_dim: usize = 784;
    let hidden_dim: usize = 32;
    let output_dim: usize = 10;

    // Initialize weights and biases
    let mut weights1 = Tensor::randn(input_dim, hidden_dim) / (input_dim as f32).sqrt();
    //println!("weights1={weights1:#?}");
    let mut bias1 = Tensor::zeros(1, hidden_dim);
    //println!("bias1={bias1:#?}");
    let mut weights2 = Tensor::randn(hidden_dim, output_dim) / (hidden_dim as f32).sqrt();
    //println!("weights2={weights2:#?}");
    let mut bias2 = Tensor::zeros(1, output_dim);
    //println!("bias2={bias2:#?}");

    println!(
        "shape(weights1)={:#?}, shape(bias1)={:#?}, shape(weights2)={:#?}, shape(bias2)={:#?}",
        weights1.1, bias1.1, weights2.1, bias2.1,
    );

    // Train model using mini-batches
    const BATCH_SIZE: usize = 128;
    let num_batches: usize = (x_train.len() as f32 / BATCH_SIZE as f32).ceil() as usize;
    let learning_rate: f32 = 0.01;
    println!("batch_size={BATCH_SIZE}, num_batches={num_batches}, learning_rate={learning_rate}");

    let mut dl_train = data_loader::DataLoader::<BATCH_SIZE, true, u8>::new(y_train, x_train);
    let mut dl_test = data_loader::DataLoader::<10000, false, u8>::new(y_test, x_test);

    let mut first0 = true;
    let mut first1 = true;
    let mut loss = None;
    for epoch in 0..100 {
        for (mbatch, (y_train, x_train)) in dl_train.iter().enumerate() {
            // Preprocess data (from u8 to f32 between 0.0 and 1.0)
            let x_batch = Tensor::new(
                x_train
                    .into_iter()
                    .flat_map(|img| {
                        img.into_iter()
                            .map(|px| (*px as f32) / 255.0)
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<f32>>(),
                (BATCH_SIZE, 28 * 28),
                false,
            );
            let y_batch = Tensor::new(
                y_train
                    .into_iter()
                    .flat_map(|lbl| identity(10)[*lbl as usize].clone())
                    .collect::<Vec<f32>>(),
                (BATCH_SIZE, 10),
                false,
            );

            if first0 {
                first0 = false;
                println!(
                    "shape(batch_x)={:#?}, shape(batch_y)={:#?}, batch_y={:#?}",
                    x_batch.1,
                    y_batch.1,
                    &y_batch.0[0..10],
                );
            }

            // Forward pass
            let hidden_layer_x = x_batch.dot(&weights1); // TODO: + bias1;
            let hidden_layer = hidden_layer_x.relu();
            let output = (hidden_layer.dot(&weights2)/*TODO: + bias2*/).softmax();

            if first1 {
                first1 = false;
                println!(
                    "shape(hidden_layer_x)={:#?}, shape(output)={:#?}",
                    hidden_layer_x.1, output.1,
                );
            }

            // Compute loss and gradients
            loss = Some(output.cross_entropy_loss(y_batch.clone()));
            let d_output = output - y_batch;
            let d_hidden_layer = d_output.dot(&weights2.transpose()) * hidden_layer_x.d_relu_dx();
            let d_weights2 = hidden_layer.transpose().dot(&d_output);
            let d_bias2 = d_output.sum(Axis::Columns, true);
            let d_weights1 = x_batch.transpose().dot(&d_hidden_layer);
            let d_bias1 = d_hidden_layer.sum(Axis::Columns, true);

            // Update weights and biases
            weights1 = weights1 - (learning_rate * d_weights1 / BATCH_SIZE as f32);
            bias1 = bias1 - (learning_rate * d_bias1 / BATCH_SIZE as f32);
            weights2 = weights2 - (learning_rate * d_weights2 / BATCH_SIZE as f32);
            bias2 = bias2 - (learning_rate * d_bias2 / BATCH_SIZE as f32);

            println!(
                "Epoch {}/{}, MiniBatch {}/{}, Loss: {:#?}, Timestamp: {}",
                epoch + 1,
                100,
                mbatch + 1,
                BATCH_SIZE,
                loss,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            );
            //break;
        }
        //break;

        /*println!(
            "Epoch {}, Loss: {:#?}, Timestamp: {}",
            epoch + 1,
            loss,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );*/

        // break;
    }

    for (y_test, x_test) in dl_test.iter() {
        // Preprocess data (from u8 to f32 between 0.0 and 1.0)
        let x_batch = Tensor::new(
            x_test
                .into_iter()
                .flat_map(|img| {
                    img.into_iter()
                        .map(|px| (*px as f32) / 255.0)
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<f32>>(),
            (BATCH_SIZE, 28 * 28),
            false,
        );
        let y_batch = Tensor::new(
            y_test
                .into_iter()
                .flat_map(|lbl| identity(10)[*lbl as usize].clone())
                .collect::<Vec<f32>>(),
            (BATCH_SIZE, 10),
            false,
        );

        // Evaluate model on test set
        let hidden_layer = (x_batch.dot(&weights1)/*+ bias1*/).relu();
        let output = (hidden_layer.dot(&weights2)/*+ bias2*/).softmax();
        let accuracy = (output
            .argmax(Axis::Rows, false)
            .eq(&y_batch.argmax(Axis::Rows, false)))
        .mean(Axis::None, false)
        .0[0];
        println!("Test Accuracy: {accuracy}");
    }
}

#[derive(Debug, Clone)]
enum Axis {
    None,
    Rows,
    #[allow(dead_code)]
    Columns,
}

#[derive(Debug, Clone)]
struct Tensor(Vec<f32>, (usize, usize), bool);

impl Tensor {
    /// Compose a new tensor from primitives
    fn new(inner: Vec<f32>, shape: (usize, usize), transposed: bool) -> Self {
        Tensor(inner, shape, transposed)
    }

    /// Zeroed out 2-dimensional tensor
    fn zeros(rows: usize, columns: usize) -> Self {
        let mut v = Vec::with_capacity(rows * columns);
        v.resize(rows * columns, 0.0);
        Tensor(v, (rows, columns), false)
    }

    /// 2-dimensional tensor with values sampled from normal distribution centered on 0.0 with a std. div. of 1.0
    fn randn(rows: usize, columns: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut v = Vec::with_capacity(rows * columns);
        v.resize_with(rows * columns, || normal.sample(&mut rand::thread_rng()));
        Tensor(v, (rows, columns), false)
    }

    /// Index into tensor by row and column
    fn at(&self, row: usize, column: usize) -> Option<&f32> {
        match self.2 {
            true => self.0.get(column * self.1 .0 + row),
            false => self.0.get(row * self.1 .1 + column),
        }
    }

    /// Index mutably into tensor by row and column
    fn at_mut(&mut self, row: usize, column: usize) -> Option<&mut f32> {
        match self.2 {
            true => self.0.get_mut(column * self.1 .0 + row),
            false => self.0.get_mut(row * self.1 .1 + column),
        }
    }

    /// e^(f32) for each value in tensor
    fn exp(&self) -> Self {
        let mut t = self.clone();
        t.0.iter_mut().for_each(|it| {
            *it = (*it).exp();
        });
        t
    }

    /// log(f32) for each value in tensor
    fn log(&self) -> Self {
        let mut t = self.clone();
        t.0.iter_mut().for_each(|it| {
            *it = (*it).ln();
        });
        t
    }

    /// Transpose matrix
    fn transpose(&self) -> Self {
        let mut t = self.clone();
        t.1 = (t.1 .1, t.1 .0);
        t.2 = true;
        t
    }

    /// Apply sigmoid activation function to tensor
    #[allow(dead_code)]
    fn sigmoid(&self) -> Self {
        let mut t = self.clone();
        t.0.iter_mut().for_each(|it| {
            *it = 1.0_f32 / (1.0_f32 + (-*it).exp());
        });
        t
    }

    /// Apply relu activation function to tensor
    fn relu(&self) -> Self {
        let mut t = self.clone();
        t.0.iter_mut().for_each(|it| {
            *it = (*it).max(0.0);
        });
        t
    }

    /// Calculate derivative of relu with respect to x
    #[allow(dead_code)]
    fn d_relu_dx(&self) -> Self {
        let mut t = self.clone();
        t.0.iter_mut().for_each(|it| {
            *it = if *it > 0.0 { 1.0 } else { 0.0 };
        });
        t
    }

    // Dot product (matrix mul)
    fn dot(&self, rhs: &Self) -> Self {
        if self.1 .1 != rhs.1 .0 {
            panic!("unexpected number of dimensions in dot product (lhs_shape=({}, {}), rhs_shape=({}, {}))", self.1.0, self.1.1, rhs.1.0, rhs.1.1);
        }
        let mut out = Tensor::zeros(self.1 .0, rhs.1 .1);
        for row in 0..out.1 .0 {
            for i in 0..self.1 .1 {
                for column in 0..out.1 .1 {
                    let a = *self.at(row, i).unwrap();
                    let b = *rhs.at(i, column).unwrap();
                    //println!("row={row}, column={column}, i={i}, lhs=({row}, {i}), rhs=({i}, {column}), a={a}, b={b}, a*b={}", a*b);
                    *out.at_mut(row, column).unwrap() += a * b;
                }
            }
        }
        out
    }

    fn argmax(&self, axis: Axis, _keep_dims: bool) -> Self {
        match axis {
            Axis::Rows => {
                let mut out = Tensor::zeros(self.1 .0, 1);
                for row in 0..self.1 .0 {
                    let mut max: Option<(usize, f32)> = None;
                    for column in 0..self.1 .1 {
                        let v = *self.at(row, column).unwrap();
                        match max {
                            Some(v_old) => {
                                if v_old.1 < v {
                                    max = Some((column, v));
                                }
                            }
                            None => max = Some((column, v)),
                        }
                    }
                    *out.at_mut(row, 0).unwrap() = max.unwrap().0 as f32;
                }
                out
            }
            Axis::None => todo!(),
            Axis::Columns => todo!(),
        }
    }

    // Calculate max value along one of the axies
    fn max(&self, axis: Axis, _keep_dims: bool) -> Self {
        match axis {
            Axis::Rows => {
                let mut out = Tensor::zeros(self.1 .0, 1);
                for row in 0..self.1 .0 {
                    let mut max = None;
                    for column in 0..self.1 .1 {
                        let v = *self.at(row, column).unwrap();
                        match max {
                            Some(v_old) => {
                                if v_old < v {
                                    max = Some(v);
                                }
                            }
                            None => max = Some(v),
                        }
                    }
                    *out.at_mut(row, 0).unwrap() = max.unwrap();
                }
                out
            }
            Axis::None => todo!(),
            Axis::Columns => todo!(),
        }
    }

    fn eq(&self, other: &Self) -> Tensor {
        if self.1 .0 != other.1 .0 || self.1 .1 != 1 || other.1 .1 != 1 {
            panic!("unexpected number of dimensions in divison (lhs_shape=({}, {}), rhs_shape=({}, {}))", self.1.0, self.1.1, other.1.0, other.1.1);
        }
        let mut out = Tensor::zeros(self.1 .0, 1);
        for row in 0..self.1 .0 {
            let v0 = *self.at(row, 0).unwrap();
            let v1 = *other.at(row, 0).unwrap();
            *out.at_mut(row, 0).unwrap() = if v0 == v1 { 1.0 } else { 0.0 };
        }
        out
    }

    // Calculate sum of value along one of the axies
    fn sum(&self, axis: Axis, _keep_dims: bool) -> Self {
        match axis {
            Axis::Rows => {
                let mut out = Tensor::zeros(self.1 .0, 1);
                for row in 0..self.1 .0 {
                    let mut sum = 0.0;
                    for column in 0..self.1 .1 {
                        sum += *self.at(row, column).unwrap();
                    }
                    *out.at_mut(row, 0).unwrap() = sum;
                }
                out
            }
            Axis::Columns => {
                let mut out = Tensor::zeros(1, self.1 .1);
                for column in 0..self.1 .1 {
                    let mut sum = 0.0;
                    for row in 0..self.1 .0 {
                        sum += *self.at(row, column).unwrap();
                    }
                    *out.at_mut(0, column).unwrap() = sum;
                }
                out
            }
            Axis::None => todo!(),
        }
    }

    // Calculate mean of value along one of the axies
    fn mean(&self, axis: Axis, _keep_dims: bool) -> Self {
        match axis {
            Axis::None => {
                let mut out = Tensor::zeros(1, 1);
                let mut sum = None;
                let denominator = (self.1 .0 * self.1 .1) as f32;
                for row in 0..self.1 .0 {
                    for column in 0..self.1 .1 {
                        let v = *self.at(row, column).unwrap();
                        match sum {
                            Some(v_old) => {
                                sum = Some(v + v_old);
                            }
                            None => sum = Some(v),
                        }
                    }
                }
                *out.at_mut(0, 0).unwrap() = sum.unwrap() / denominator;
                out
            }
            Axis::Rows => todo!(),
            Axis::Columns => todo!(),
        }
    }

    /// Apply softmax activation function to tensor
    fn softmax(&self) -> Self {
        let e_t = (self.clone() - self.clone().max(Axis::Rows, true)).exp();
        e_t.clone() / e_t.sum(Axis::Rows, true)
    }

    /// Cross entropy loss function for estimating loss
    fn cross_entropy_loss(&self, labels: Tensor) -> Self {
        -((labels * self.log())
            .sum(Axis::Rows, false)
            .mean(Axis::None, false))
    }
}

impl Div<f32> for Tensor {
    type Output = Self;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.0.iter_mut().for_each(|it| *it = *it / rhs);
        self
    }
}

impl Div for Tensor {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        if rhs.1 .0 != self.1 .0 || rhs.1 .1 != 1 {
            panic!("unexpected number of dimensions in divison (lhs_shape=({}, {}), rhs_shape=({}, {}))", self.1.0, self.1.1, rhs.1.0, rhs.1.1);
        }
        for row in 0..self.1 .0 {
            let r = rhs.at(row, 0).unwrap();
            for column in 0..self.1 .1 {
                *self.at_mut(row, column).unwrap() /= *r;
            }
        }
        self
    }
}

impl Mul<f32> for Tensor {
    type Output = Self;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.0.iter_mut().for_each(|it| *it = *it * rhs);
        self
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, mut rhs: Tensor) -> Self::Output {
        rhs.0.iter_mut().for_each(|it| *it = *it * self);
        rhs
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        if rhs.1 .0 != self.1 .0 || rhs.1 .1 != self.1 .1 {
            panic!("unexpected number of dimensions in divison (lhs_shape=({}, {}), rhs_shape=({}, {}))", self.1.0, self.1.1, rhs.1.0, rhs.1.1);
        }
        for row in 0..self.1 .0 {
            for column in 0..self.1 .1 {
                *self.at_mut(row, column).unwrap() *= *rhs.at(row, column).unwrap();
            }
        }
        self
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        if self.1 .0 == rhs.1 .0 && self.1 .1 == rhs.1 .1 {
            for row in 0..self.1 .0 {
                for column in 0..self.1 .1 {
                    let r = rhs.at(row, column).unwrap();
                    *self.at_mut(row, column).unwrap() -= *r;
                }
            }
        } else if rhs.1 .0 == self.1 .0 && rhs.1 .1 == 1 {
            for row in 0..self.1 .0 {
                let r = rhs.at(row, 0).unwrap();
                for column in 0..self.1 .1 {
                    *self.at_mut(row, column).unwrap() -= *r;
                }
            }
        } else {
            panic!("unexpected number of dimensions in substraction (lhs_shape=({}, {}), rhs_shape=({}, {}))", self.1.0, self.1.1, rhs.1.0, rhs.1.1);
        }
        self
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for row in 0..self.1 .0 {
            for column in 0..self.1 .1 {
                let n = self.at(row, column).unwrap().neg();
                *self.at_mut(row, column).unwrap() = n;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_test() {
        let t = Tensor::zeros(2, 2);
        assert_eq!(0.0, t.0.iter().sum::<f32>());
        //println!("{t:#?}");
    }

    #[test]
    fn randn_test() {
        let t = Tensor::randn(2, 2);
        assert_ne!(0.0, t.0.iter().sum::<f32>());
        //println!("{t:#?}");
    }

    #[test]
    fn sigmoid_test() {
        let t = Tensor::new(
            vec![-0.63926487, 0.59283878, 0.20237229, 0.58973558],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.sigmoid();
        assert!((0.34541274 - t_t.0[0]).abs() < 0.01);
        assert!((0.64401623 - t_t.0[1]).abs() < 0.01);
        assert!((0.55042111 - t_t.0[2]).abs() < 0.01);
        assert!((0.64330447 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn relu_test() {
        let t = Tensor::new(
            vec![-0.63926487, 0.59283878, 0.20237229, 0.58973558],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.relu();
        assert!((0.0 - t_t.0[0]).abs() < 0.01);
        assert!((0.59283878 - t_t.0[1]).abs() < 0.01);
        assert!((0.20237229 - t_t.0[2]).abs() < 0.01);
        assert!((0.58973558 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn d_relu_dx_test() {
        let t = Tensor::new(
            vec![-0.63926487, 0.59283878, 0.20237229, 0.58973558],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.d_relu_dx();
        assert_eq!(0.0, t_t.0[0]);
        assert_eq!(1.0, t_t.0[1]);
        assert_eq!(1.0, t_t.0[2]);
        assert_eq!(1.0, t_t.0[3]);
        //println!("{t_t:#?}");
    }

    #[test]
    fn max_test() {
        let t = Tensor::new(
            vec![-0.63926487, 0.59283878, 0.20237229, 0.58973558],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.max(Axis::Rows, true);
        assert_eq!(2, t_t.0.len());
        assert!((0.59283876 - t_t.0[0]).abs() < 0.01);
        assert!((0.58973557 - t_t.0[1]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn sub_test() {
        let t1 = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        let t2 = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        //println!("{t1:#?}, {t2:#?}");
        let t_t = t1 - t2;
        assert!((0.0 - t_t.0[0]).abs() < 0.01);
        assert!((0.0 - t_t.0[1]).abs() < 0.01);
        assert!((0.0 - t_t.0[2]).abs() < 0.01);
        assert!((0.0 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn sub_1dim_test() {
        let t1 = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        let t2 = Tensor::new(vec![1.0252992, 0.41544288], (2, 1), false);
        //println!("{t1:#?}, {t2:#?}");
        let t_t = t1 - t2;
        assert!((-0.15306509 - t_t.0[0]).abs() < 0.01);
        assert!((0.0 - t_t.0[1]).abs() < 0.01);
        assert!((-1.2684548 - t_t.0[2]).abs() < 0.01);
        assert!((0.0 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn exp_test() {
        let t = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.exp();
        assert!((2.39224937 - t_t.0[0]).abs() < 0.01);
        assert!((2.78792948 - t_t.0[1]).abs() < 0.01);
        assert!((0.42612954 - t_t.0[2]).abs() < 0.01);
        assert!((1.51504157 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn div_1dim_test() {
        let t1 = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        let t2 = Tensor::new(vec![1.89753328, -0.43756901], (2, 1), false);
        //println!("{t1:#?}, {t2:#?}");
        let t_t = t1 / t2;
        assert!((0.45966734 - t_t.0[0]).abs() < 0.01);
        assert!((0.54033266 - t_t.0[1]).abs() < 0.01);
        assert!((1.94943395 - t_t.0[2]).abs() < 0.01);
        assert!((-0.94943395 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn softmax_test() {
        let t = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        //println!("{t:#?}");
        let t_t = t.softmax();
        assert!((0.46180826 - t_t.0[0]).abs() < 0.01);
        assert!((0.53819174 - t_t.0[1]).abs() < 0.01);
        assert!((0.21952189 - t_t.0[2]).abs() < 0.01);
        assert!((0.78047811 - t_t.0[3]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn cross_entropy_loss_test() {
        let t1 = Tensor::new(
            vec![0.87223408, 1.0252992, -0.85301189, 0.41544288],
            (2, 2),
            false,
        );
        let t2 = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
        //println!("{t1:#?}, {t2:#?}");
        let t_t1 = t1.softmax();
        //println!("{t_t1:#?}");
        let t_t2 = t_t1.cross_entropy_loss(t2);
        assert_eq!(t_t2.1.clone(), (1, 1));
        assert!((0.5102270402592233 - t_t2.0[0]).abs() < 0.01);
        //println!("{t_t2:#?}");
    }

    #[test]
    fn dot_test() {
        let t1 = Tensor::new(vec![2.0, 2.0, 0.0, 3.0, 0.0, 4.0], (3, 2), false);
        let t2 = Tensor::new(vec![2.0, 1.0, 2.0, 3.0, 2.0, 4.0], (2, 3), false);
        //println!("{t1:#?}, {t2:#?}");
        let t_t = t1.dot(&t2);
        assert!((10.0 - t_t.0[0]).abs() < 0.01);
        assert!((6.0 - t_t.0[1]).abs() < 0.01);
        assert!((12.0 - t_t.0[2]).abs() < 0.01);
        assert!((9.0 - t_t.0[3]).abs() < 0.01);
        assert!((6.0 - t_t.0[4]).abs() < 0.01);
        assert!((12.0 - t_t.0[5]).abs() < 0.01);
        assert!((12.0 - t_t.0[6]).abs() < 0.01);
        assert!((8.0 - t_t.0[7]).abs() < 0.01);
        assert!((16.0 - t_t.0[8]).abs() < 0.01);
        //println!("{t_t:#?}");
    }

    #[test]
    fn transpose_test() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), false);
        //println!("{t:#?}");
        let t_t = t.transpose();
        assert_eq!((3, 2), t_t.1);
        assert!((1.0 - *t_t.at(0, 0).unwrap()).abs() < 0.01);
        assert!((4.0 - *t_t.at(0, 1).unwrap()).abs() < 0.01);
        assert!((2.0 - *t_t.at(1, 0).unwrap()).abs() < 0.01);
        assert!((5.0 - *t_t.at(1, 1).unwrap()).abs() < 0.01);
        assert!((3.0 - *t_t.at(2, 0).unwrap()).abs() < 0.01);
        assert!((6.0 - *t_t.at(2, 1).unwrap()).abs() < 0.01);
        //println!("{t_t:#?}");
    }
}
