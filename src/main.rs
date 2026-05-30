// wget https://huggingface.co/giangndm/yolo11-onnx/resolve/main/yolo11n_640.onnx

use std::collections::HashMap;
use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use oxionnx::{OptLevel, Session, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

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
                // Indexing is: (c * 640 * 640) + (y * 640) + x
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

    // Print the actual names and order of the model\\\'s outputs
    for (i, tensor_info) in session.output_info().iter().enumerate() {
        println!(
            "Output Index [{}]: Name = {}, Type = {:?}, Shape = {:?}, Dim Params = {:?}",
            i, tensor_info.name, tensor_info.dtype, tensor_info.shape, tensor_info.dim_params
        );
    }

    // Prepare input
    let mut inputs = HashMap::new();
    inputs.insert("images", Tensor::from_ndarray(input_tensor));

    // Run inference
    let outputs = session.run(&inputs)?;

    println!("Inference complete!");

    let output_tensor = outputs
        .get("output0")
        .ok_or("Output tensor \"output0\" not found")?;

    println!("Output tensor: {:?}", output_tensor);

    // You can add further processing here, e.g., NMS, drawing bounding boxes

    Ok(())
}
