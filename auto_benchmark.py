"""
auto_benchmark.py

Automatically benchmark all OpenVINO IR models in a folder using a sample image or video.

Features:
- Iterates over all .xml models in a specified folder
- Preprocesses a sample image or first video frame
- Runs a warm-up inference to stabilize performance
- Measures CPU inference time (ms per model)
- Sorts and prints results by speed
- Supports configurable CPU threads and throughput streams

Usage:
python auto_benchmark.py --models_dir models --sample data/sample.jpg --threads 4 --streams 1
"""

import os
import time
import argparse
import cv2
import numpy as np
from openvino.runtime import Core

def load_model(model_path: str, num_threads: int = 4, num_streams: int = 1):
    """
    Load and compile an OpenVINO IR model for CPU inference.

    Args:
        model_path (str): Path to .xml model file
        num_threads (int): Number of CPU threads
        num_streams (int): Number of CPU throughput streams

    Returns:
        compiled_model: OpenVINO compiled model ready for inference
    """
    ie = Core()
    ie.set_property({
        "CPU_THREADS_NUM": num_threads,
        "CPU_THROUGHPUT_STREAMS": num_streams
    })
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_input(frame: np.ndarray, input_shape):
    """
    Preprocess an image or video frame for the model.

    Args:
        frame (np.ndarray): Original frame/image (HWC)
        input_shape (tuple): Model input shape (batch, channels, height, width)

    Returns:
        np.ndarray: Preprocessed frame with batch dimension (1, C, H, W)
    """
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    frame_transposed = frame_resized.transpose((2, 0, 1))  # HWC -> CHW
    return frame_transposed[np.newaxis, :]

def benchmark_model(model_path: str, sample_input: np.ndarray, num_threads: int = 4, num_streams: int = 1):
    """
    Benchmark a single model on the sample input.

    Args:
        model_path (str): Path to OpenVINO IR model
        sample_input (np.ndarray): Preprocessed image/frame
        num_threads (int): CPU threads
        num_streams (int): CPU streams

    Returns:
        float: Inference time in milliseconds
    """
    compiled_model = load_model(model_path, num_threads, num_streams)
    input_layer = compiled_model.inputs[0]

    # Warm-up inference to stabilize CPU performance
    _ = compiled_model([sample_input])[compiled_model.outputs[0]]

    # Measure actual inference time
    start_time = time.time()
    _ = compiled_model([sample_input])[compiled_model.outputs[0]]
    end_time = time.time()

    # Return time in milliseconds
    return (end_time - start_time) * 1000

def main():
    """
    Main function to iterate over all models in the folder and benchmark them.
    Prints sorted results by inference speed.
    """
    parser = argparse.ArgumentParser(description="Auto Benchmark OpenVINO models on CPU")
    parser.add_argument("--models_dir", default="models", help="Folder containing .xml OpenVINO models")
    parser.add_argument("--sample", default="data/sample.jpg", help="Sample image or video for benchmarking")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU throughput streams")
    args = parser.parse_args()

    # Load sample input (image or first frame from video)
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

    # Store results for all models
    results = []

    # Iterate over models
    for file in os.listdir(args.models_dir):
        if file.endswith(".xml"):
            model_path = os.path.join(args.models_dir, file)
            try:
                # Preprocess sample input for the model
                compiled_model = load_model(model_path, args.threads, args.streams)
                input_layer = compiled_model.inputs[0]
                input_data = preprocess_input(sample_input, input_layer.shape)

                # Benchmark model
                time_ms = benchmark_model(model_path, input_data, args.threads, args.streams)
                results.append((file, time_ms))
            except Exception as e:
                print(f"Error benchmarking {file}: {e}")

    # Sort results by fastest inference
    results.sort(key=lambda x: x[1])

    # Print benchmark summary
    print("\n=== CPU Benchmark Results (ms per inference) ===")
    for model_name, time_ms in results:
        print(f"{model_name}: {time_ms:.2f} ms")

if __name__ == "__main__":
    main()
