use std::collections::HashMap;
use std::path::Path;

use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array4, Axis};
use oxionnx::{OptLevel, Session, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Load image
    let img_path = Path::new("test.jpeg");
    let img: DynamicImage = image::open(img_path).expect("Failed to open image file");

    // Resize image to 640x640
    let resized_img = img.resize_exact(640, 640, FilterType::Triangle);

    // Create a flat Vector for the normalized Float32 pixel values
    let mut input_data = Vec::with_capacity(3 * 640 * 640);

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
        .load("yolov8n.onnx".as_ref())?;

    println!("Model loaded successfully!");

    println!("Inputs: {:?}", session.input_names());
    println!("Metadata: {:?}", session.metadata());

    // Print the actual names and order of the model's inputs
    for (i, tensor_info) in session.input_info().iter().enumerate() {
        println!(
            "Input Index [{}]: Name = {}, Type = {:?}, Shape = {:?}, Dim Params = {:?}",
            i, tensor_info.name, tensor_info.dtype, tensor_info.shape, tensor_info.dim_params
        );
    }

    // Print the actual names and order of the model's outputs
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

    println!("Output tensor shape: {:?}", output_tensor.shape);

    // Convert the oxionnx Tensor back to an ndarray for parsing
    let output_ndarray = output_tensor.to_ndarray();

    // Remove the batch dimension -> shape becomes [84, 8400]
    let output_matrix = output_ndarray.index_axis(Axis(0), 0);

    let num_elements = output_matrix.shape()[0]; // 84
    let num_candidates = output_matrix.shape()[1]; // 8400
    let num_classes = num_elements - 4; // 80

    // Define confidence threshold
    let conf_threshold = 0.80;

    println!("Parsing bounding boxes (Threshold: {})...", conf_threshold);

    for col in 0..num_candidates {
        // Extract confidence scores for all 80 classes for this candidate box
        let mut max_score = 0.0;
        let mut class_id = 0;

        for class_idx in 0..num_classes {
            let score = output_matrix[[4 + class_idx, col]];
            if score > max_score {
                max_score = score;
                class_id = class_idx;
            }
        }

        // Filter out predictions below the defined confidence threshold
        if max_score > conf_threshold {
            // Extract bounding box geometry coordinates
            let xc = output_matrix[[0, col]];
            let yc = output_matrix[[1, col]];
            let w = output_matrix[[2, col]];
            let h = output_matrix[[3, col]];

            // Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
            let x1 = xc - (w / 2.0);
            let y1 = yc - (h / 2.0);
            let x2 = xc + (w / 2.0);
            let y2 = yc + (h / 2.0);

            println!(
                "Found Object -> Class ID: {}, Confidence: {:.4}, BBox: [x1: {:.1}, y1: {:.1}, x2: {:.1}, y2: {:.1}]",
                class_id, max_score, x1, y1, x2, y2
            );
        }
    }

    Ok(())
}
