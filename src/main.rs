use std::collections::HashMap;
/**
 * wget https://huggingface.co/giangndm/yolo11-onnx/resolve/main/yolo11n_640.onnx
*/
use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use oxionnx::{OptLevel, Session, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let img_path = Path::new("test.jpeg");
    let img: DynamicImage = image::open(img_path).expect("Failed to open image file");

    // Resize image to 640x640
    let resized_img = img.resize_exact(640, 640, FilterType::Triangle);

    // Create a flat Vector for the normalized Float32 pixel values
    let mut input_data = Vec::with_capacity(1 * 3 * 640 * 640);

    // Extract RGB values and normalize them
    let rgb_img = resized_img.to_rgb8();

    // YOLO expects CHW format (All Reds, then all Greens, then all Blues)
    for c in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                let pixel = rgb_img.get_pixel(x, y);
                let pixel_value = pixel[c] as f32 / 255.0;
                input_data.push(pixel_value);
            }
        }
    }

    // Reshape the flat vector into a 4D Array: [Batch, Channels, Height, Width]
    let input_tensor = Array4::from_shape_vec((1, 3, 640, 640), input_data)
        .expect("Failed to shape tensor properly");

    // ====

    // Load model
    let session = Session::builder()
        .with_optimization_level(OptLevel::All)
        .with_memory_pool(true)
        .with_parallel_execution(true)
        .with_profiling()
        .load("yolo11n_640.onnx".as_ref())?;

    println!("Model loaded successfully!");

    println!("Inputs: {:?}", session.input_names());
    println!("Metadata: {:?}", session.metadata());

    // Prepare input
    let mut inputs = HashMap::new();
    inputs.insert("images", Tensor::from_ndarray(input_tensor));

    // Run inference
    let outputs = session.run(&inputs)?;
    println!("{:?}", outputs);

    Ok(())
}
