use std::{fs, io::Read};

use byteorder::{BigEndian, ReadBytesExt};
use image::ExtendedColorType;

const TRAIN_LABELS_FILENAME: &str = "data/train-labels-idx1-ubyte";
const TRAIN_IMAGES_FILENAME: &str = "data/train-images-idx3-ubyte";
const T10K_LABELS_FILENAME: &str = "data/t10k-labels-idx1-ubyte";
const T10K_IMAGES_FILENAME: &str = "data/t10k-images-idx3-ubyte";

pub fn load(training: bool) -> Result<(Vec<u8>, Vec<Vec<u8>>), Box<dyn std::error::Error>> {
    let labels = {
        let mut file = fs::File::open(if training {
            TRAIN_LABELS_FILENAME
        } else {
            T10K_LABELS_FILENAME
        })?;

        let magic = file.read_u32::<BigEndian>()? as usize;
        let size = file.read_u32::<BigEndian>()? as usize;

        if magic != 2049 {
            return Err(format!("Magic number mismatch, expected 2049, got {}", magic).into());
        }

        let mut buf = Vec::<u8>::with_capacity(size);
        buf.resize(size, 0);
        file.read_exact(&mut buf)?;
        buf
    };

    let images = {
        let mut file = fs::File::open(if training {
            TRAIN_IMAGES_FILENAME
        } else {
            T10K_IMAGES_FILENAME
        })?;

        let magic = file.read_u32::<BigEndian>()? as usize;
        let size = file.read_u32::<BigEndian>()? as usize;
        let rows = file.read_u32::<BigEndian>()? as usize;
        let cols = file.read_u32::<BigEndian>()? as usize;

        if magic != 2051 {
            return Err(format!("Magic number mismatch, expected 2051, got {}", magic).into());
        }

        let mut buf = Vec::<u8>::with_capacity(rows * cols);
        buf.resize(rows * cols, 0);

        let mut images: Vec<Vec<u8>> = vec![];
        for _ in 0..size {
            file.read_exact(&mut buf)?;
            images.push(buf.clone());
        }
        images
    };

    Ok((labels, images))
}

#[allow(dead_code)]
pub fn output_image(filename: &str, image: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    image::save_buffer(filename, image, 28, 28, ExtendedColorType::L8).map_err(|it| it.into())
}
