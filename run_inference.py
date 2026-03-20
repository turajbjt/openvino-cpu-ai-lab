"""
run_inference.py

Batch image inference using OpenVINO CPU.

Features:
- Supports multiple images in a single batch
- Top-5 predictions per image
- Configurable CPU threads and streams
- Warm-up for accurate timing

Usage:
python run_inference.py --model models/resnet50.xml --inputs data/img1.jpg data/img2.jpg --threads 4 --streams 1
"""

import argparse
import os
import time
import cv2
import numpy as np
from openvino.runtime import Core

def load_model(model_path: str, num_threads: int = 4, num_streams: int = 1):
    """
    Load and compile an OpenVINO IR model for CPU inference.

    Args:
        model_path (str): Path to the .xml OpenVINO IR model
        num_threads (int): Number of CPU threads
        num_streams (int): Number of CPU throughput streams

    Returns:
        compiled_model: Compiled OpenVINO model ready for inference
    """
    ie = Core()
    # Set CPU-specific performance options
    ie.set_property({
        "CPU_THREADS_NUM": num_threads,
        "CPU_THROUGHPUT_STREAMS": num_streams
    })

    # Load and compile the model for CPU
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_image(image_path: str, input_shape):
    """
    Read and preprocess a single image for model input.

    Args:
        image_path (str): Path to the image file
        input_shape (tuple): Model input shape (batch, channels, height, width)

    Returns:
        np.ndarray: Preprocessed image array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    
    # Resize image to model input dimensions
    image_resized = cv2.resize(image, (input_shape[3], input_shape[2]))
    
    # Convert HWC -> CHW format
    image_transposed = image_resized.transpose((2, 0, 1))
    return image_transposed

def load_batch(image_paths: list, input_shape):
    """
    Preprocess and stack multiple images into a batch array.

    Args:
        image_paths (list): List of image file paths
        input_shape (tuple): Model input shape

    Returns:
        np.ndarray: Batch of preprocessed images (batch_size, channels, height, width)
    """
    batch = []
    for img_path in image_paths:
        batch.append(preprocess_image(img_path, input_shape))
    return np.stack(batch, axis=0)  # Create a batch along axis 0

def main():
    """
    Main function to run batch inference.
    Parses command-line arguments, loads the model, runs inference,
    and prints top-5 predictions per image.
    """
    parser = argparse.ArgumentParser(description="Batch Image Inference with OpenVINO")
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR model (.xml)")
    parser.add_argument("--inputs", nargs="+", required=True, help="Path(s) to input image(s)")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="Number of CPU throughput streams")
    args = parser.parse_args()

    # Load and compile the model
    compiled_model = load_model(args.model, args.threads, args.streams)
    input_layer = compiled_model.inputs[0]  # Get input layer details

    # Load and preprocess all images as a batch
    input_data = load_batch(args.inputs, input_layer.shape)

    # Warm-up inference to stabilize performance
    _ = compiled_model([input_data])[compiled_model.outputs[0]]

    # Actual inference with timing
    start_time = time.time()
    output = compiled_model([input_data])[compiled_model.outputs[0]]
    end_time = time.time()

    # Report total inference time for batch
    print(f"Inference time for batch of {len(args.inputs)} images: {(end_time - start_time)*1000:.2f} ms")

    # Process output per image
    for i, out in enumerate(output):
        top5_idx = out.argsort()[-5:][::-1]  # Get top-5 class indices
        print(f"\nImage: {args.inputs[i]}")
        print("Top-5 predictions:")
        for idx in top5_idx:
            print(f"  Class {idx}: {out[idx]:.4f}")

if __name__ == "__main__":
    main()
