import os
import time
import argparse
import cv2
import numpy as np
from openvino.runtime import Core

## Features
# Automatically finds all .xml models in models/
# Supports images or videos as input
# Configurable CPU threads and streams
# Warm-up run for accurate timing
# Prints sorted inference times for easy comparison
# Can be used for quick profiling or adding to README benchmarks


def load_model(model_path, num_threads=4, num_streams=1):
    ie = Core()
    ie.set_property({"CPU_THREADS_NUM": num_threads, "CPU_THROUGHPUT_STREAMS": num_streams})
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_input(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    frame_transposed = frame_resized.transpose((2,0,1))
    return frame_transposed[np.newaxis, :]

def benchmark_model(model_path, sample_input, num_threads=4, num_streams=1):
    compiled_model = load_model(model_path, num_threads, num_streams)
    input_layer = compiled_model.inputs[0]

    input_data = preprocess_input(sample_input, input_layer.shape)

    # Warm-up
    for _ in range(5):
        _ = compiled_model([input_data])[compiled_model.outputs[0]]

    # Measure inference time
    start_time = time.time()
    _ = compiled_model([input_data])[compiled_model.outputs[0]]
    end_time = time.time()

    return (end_time - start_time) * 1000  # ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models", help="Folder with OpenVINO IR models (.xml)")
    parser.add_argument("--sample", default="data/sample.jpg", help="Sample image/video for benchmarking")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU streams")
    args = parser.parse_args()

    # Load sample input
    if args.sample.lower().endswith(('.jpg','.png','.jpeg')):
        sample_input = cv2.imread(args.sample)
    else:
        cap = cv2.VideoCapture(args.sample)
        ret, sample_input = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read sample video frame from {args.sample}")

    # Benchmark all models
    results = []
    for file in os.listdir(args.models_dir):
        if file.endswith(".xml"):
            model_path = os.path.join(args.models_dir, file)
            try:
                time_ms = benchmark_model(model_path, sample_input, args.threads, args.streams)
                results.append((file, time_ms))
            except Exception as e:
                print(f"Error benchmarking {file}: {e}")

    # Print sorted results
    results.sort(key=lambda x: x[1])
    print("\n=== CPU Benchmark Results (ms per inference) ===")
    for model_name, time_ms in results:
        print(f"{model_name}: {time_ms:.2f} ms")

if __name__ == "__main__":
    main()
