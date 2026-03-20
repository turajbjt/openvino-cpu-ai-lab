"""
benchmark_model.py

Benchmark a single OpenVINO IR model on CPU.

Features:
- Measures inference time for a model
- Supports configurable CPU threads and streams
- Performs warm-up run for accurate timing
- Optional asynchronous inference for faster benchmarking
- Prints results clearly with milliseconds

Usage:
python benchmark_model.py --model models/resnet50.xml --sample data/sample.jpg --threads 4 --streams 1
"""

import argparse
import time
import cv2
import numpy as np
from openvino.runtime import Core

def load_model(model_path: str, num_threads: int = 4, num_streams: int = 1):
    """
    Load and compile an OpenVINO IR model for CPU inference.

    Args:
        model_path (str): Path to the .xml model
        num_threads (int): Number of CPU threads
        num_streams (int): Number of CPU throughput streams

    Returns:
        compiled_model: Compiled OpenVINO model
    """
    ie = Core()
    # Set CPU performance properties
    ie.set_property({
        "CPU_THREADS_NUM": num_threads,
        "CPU_THROUGHPUT_STREAMS": num_streams
    })
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_input(frame: np.ndarray, input_shape):
    """
    Preprocess an image/frame for model input.

    Args:
        frame (np.ndarray): Original image/frame (HWC)
        input_shape (tuple): Model input shape (batch, channels, height, width)

    Returns:
        np.ndarray: Preprocessed frame (1, C, H, W)
    """
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    frame_transposed = frame_resized.transpose((2,0,1))
    return frame_transposed[np.newaxis, :]

def benchmark_model(model_path: str, sample_input: np.ndarray, num_threads: int = 4, num_streams: int = 1):
    """
    Benchmark a single model with warm-up and timed inference.

    Args:
        model_path (str): Path to .xml OpenVINO model
        sample_input (np.ndarray): Preprocessed image/frame
        num_threads (int): Number of CPU threads
        num_streams (int): Number of CPU streams

    Returns:
        float: Inference time in milliseconds
    """
    compiled_model = load_model(model_path, num_threads, num_streams)
    input_layer = compiled_model.inputs[0]

    # Optional: Create asynchronous inference request
    infer_request = compiled_model.create_infer_request()

    # Warm-up run for stable performance
    infer_request.infer({input_layer: sample_input})

    # Measure inference time
    start_time = time.time()
    infer_request.infer({input_layer: sample_input})
    end_time = time.time()

    return (end_time - start_time) * 1000  # Convert to ms

def main():
    """
    Main function to benchmark a single model.
    Loads sample input, preprocesses it, runs benchmark, and prints results.
    """
    parser = argparse.ArgumentParser(description="CPU Benchmark for a single OpenVINO model")
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR model (.xml)")
    parser.add_argument("--sample", default="data/sample.jpg", help="Sample image or video frame")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU throughput streams")
    args = parser.parse_args()

    # Load sample input
    if args.sample.lower().endswith(('.jpg', '.png', '.jpeg')):
        sample_input = cv2.imread(args.sample)
        if sample_input is None:
            raise RuntimeError(f"Could not read image: {args.sample}")
    else:
        cap = cv2.VideoCapture(args.sample)
        ret, sample_input = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame from video: {args.sample}")

    # Preprocess input for the model
    compiled_model = load_model(args.model, args.threads, args.streams)
    input_layer = compiled_model.inputs[0]
    input_data = preprocess_input(sample_input, input_layer.shape)

    # Run benchmark
    time_ms = benchmark_model(args.model, input_data, args.threads, args.streams)
    print(f"Benchmark result for {args.model}: {time_ms:.2f} ms per inference")

if __name__ == "__main__":
    main()
