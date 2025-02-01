use rand::prelude::SliceRandom;

pub struct DataLoader<const BATCH_SIZE: usize, const SHUFFLE: bool, DTYPE = f32> {
    labels: Vec<DTYPE>,
    images: Vec<Vec<DTYPE>>,
}

impl<const BATCH_SIZE: usize, const SHUFFLE: bool, DTYPE: Clone>
    DataLoader<BATCH_SIZE, SHUFFLE, DTYPE>
{
    pub fn new(labels: Vec<DTYPE>, images: Vec<Vec<DTYPE>>) -> Self {
        assert_eq!(
            labels.len(),
            images.len(),
            "labels and images must have the same length vectors"
        );

        Self { labels, images }
    }

    pub fn iter(&mut self) -> DataLoaderEpochIterator<BATCH_SIZE, DTYPE> {
        if SHUFFLE {
            // Shuffle internals if SHUFFLE == true
            let mut indices: Vec<usize> = (0..self.labels.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            let mut t0 = Vec::with_capacity(self.labels.len());
            let mut t1 = Vec::with_capacity(self.images.len());
            for &idx in &indices {
                t0.push(self.labels[idx].clone());
                t1.push(self.images[idx].clone());
            }

            self.labels = t0;
            self.images = t1;
        }

        DataLoaderEpochIterator::<BATCH_SIZE, DTYPE> {
            vec1: self.labels.as_slice(),
            vec2: &self.images.as_slice(),
            batch: 0,
        }
    }
}

pub struct DataLoaderEpochIterator<'a, const BATCH_SIZE: usize, DTYPE> {
    vec1: &'a [DTYPE],
    vec2: &'a [Vec<DTYPE>],
    batch: usize,
}

impl<'a, const BATCH_SIZE: usize, DTYPE> DataLoaderEpochIterator<'a, BATCH_SIZE, DTYPE> {
    #[allow(dead_code)]
    pub fn new(vec1: &'a [DTYPE], vec2: &'a [Vec<DTYPE>]) -> Self {
        Self {
            vec1,
            vec2,
            batch: 0,
        }
    }
}

impl<'a, const BATCH_SIZE: usize, DTYPE> Iterator
    for DataLoaderEpochIterator<'a, BATCH_SIZE, DTYPE>
{
    type Item = (&'a [DTYPE], &'a [Vec<DTYPE>]);

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.vec1.len();
        let offset = self.batch * BATCH_SIZE;

        if offset + BATCH_SIZE <= len {
            let slice1 = &self.vec1[offset..(offset + BATCH_SIZE)];
            let slice2 = &self.vec2[offset..(offset + BATCH_SIZE)];
            self.batch += 1;
            Some((slice1, slice2))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.vec1.len() / BATCH_SIZE;
        (len, Some(len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_shuffle_dataload() {
        let labels = vec![1_u8, 2, 3, 4];
        let images = vec![
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
        ];

        let mut dl = DataLoader::<2, false, u8>::new(labels, images);
        for (labels, images) in dl.iter() {
            assert_eq!(labels.len(), 2, "batch size should be 2");
            assert_eq!(images.len(), 2, "batch size should be 2");

            let label0 = labels[0] as usize;
            let pixel0 = images[0][label0 - 1];

            if label0 == 1 {
                assert_eq!(pixel0, 1);
            } else {
                assert_eq!(pixel0, 3);
            }
        }
    }

    #[test]
    fn shuffle_dataload() {
        let labels = vec![1_u8, 2, 3, 4];
        let images = vec![
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
        ];

        let mut dl = DataLoader::<2, true, u8>::new(labels, images);
        for (labels, images) in dl.iter() {
            assert_eq!(labels.len(), 2, "batch size should be 2");
            assert_eq!(images.len(), 2, "batch size should be 2");

            let label0 = labels[0] as usize;
            let pixel0 = images[0][label0 - 1];
            assert_eq!(pixel0, label0 as u8);
        }
    }

    #[test]
    fn shuffle_on_each_epoch() {
        let labels = vec![1_u8, 2, 3, 4];
        let images = vec![
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
            vec![1_u8, 2, 3, 4],
        ];

        let mut dl = DataLoader::<2, true, u8>::new(labels, images);
        let some0: Vec<u8> = dl.iter().map(|it| it.0.to_vec()).nth(0).unwrap();
        let some1: Vec<u8> = dl.iter().map(|it| it.0.to_vec()).nth(0).unwrap();
        assert_ne!(some0, some1);
    }
}
