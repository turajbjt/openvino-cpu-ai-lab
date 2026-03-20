import argparse
import time
import cv2
import numpy as np
from openvino.runtime import Core

## Image Inference with CPU Optimizations [with Top-5 Predictions and FPS]
# FPS (inference time in ms)
# Top-5 predictions instead of just top-1
# Automatic batching for single image
# CU threads and streams configurable via command-line
# Supports any model with compatible input shape

def load_model(model_path, num_threads=4, num_streams=1):
    ie = Core()
    ie.set_property({"CPU_THREADS_NUM": num_threads, "CPU_THROUGHPUT_STREAMS": num_streams})
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[3], input_shape[2]))
    image_transposed = image_resized.transpose((2, 0, 1))  # HWC -> CHW
    return image_transposed[np.newaxis, :]  # Add batch dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR model (.xml)")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU streams")
    args = parser.parse_args()

    compiled_model = load_model(args.model, args.threads, args.streams)
    input_layer = compiled_model.inputs[0]

    input_data = preprocess_image(args.input, input_layer.shape)

    # Measure inference time
    start_time = time.time()
    output = compiled_model([input_data])[compiled_model.outputs[0]]
    end_time = time.time()

    top5_idx = output[0].argsort()[-5:][::-1]
    print(f"Inference time: {(end_time-start_time)*1000:.2f} ms")
    print("Top-5 predictions:")
    for i in top5_idx:
        print(f"  Class {i}: {output[0][i]:.4f}")

if __name__ == "__main__":
    main()
