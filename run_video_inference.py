"""
run_video_inference.py

Real-time video or webcam inference using OpenVINO CPU.

Features:
- Asynchronous inference for improved FPS
- Displays Top-5 predictions per frame
- Configurable CPU threads and streams
- Optional webcam or video file input
- Overlay FPS and predictions on video frames

Usage:
python run_video_inference.py --model models/resnet50.xml --source 0 --threads 4 --streams 1
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
        model_path (str): Path to the .xml OpenVINO IR model
        num_threads (int): Number of CPU threads
        num_streams (int): Number of CPU throughput streams

    Returns:
        compiled_model: Compiled OpenVINO model
    """
    ie = Core()
    ie.set_property({
        "CPU_THREADS_NUM": num_threads,
        "CPU_THROUGHPUT_STREAMS": num_streams
    })
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_frame(frame: np.ndarray, input_shape):
    """
    Preprocess a video frame for model input.

    Args:
        frame (np.ndarray): Original video frame (HWC)
        input_shape (tuple): Model input shape (batch, channels, height, width)

    Returns:
        np.ndarray: Preprocessed frame (1, C, H, W)
    """
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    frame_transposed = frame_resized.transpose((2, 0, 1))
    return frame_transposed[np.newaxis, :]

def main():
    """
    Main function for real-time video inference.
    Parses command-line arguments, loads model, runs async inference,
    and displays FPS and Top-5 predictions per frame.
    """
    parser = argparse.ArgumentParser(description="Real-Time Video/Webcam Inference with OpenVINO")
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR model (.xml)")
    parser.add_argument("--source", default=0, help="Video file path or webcam index (default 0)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU throughput streams")
    args = parser.parse_args()

    # Load and compile model
    compiled_model = load_model(args.model, args.threads, args.streams)
    input_layer = compiled_model.inputs[0]

    # Open video source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error opening video source {args.source}")
        return

    # Create asynchronous inference request
    infer_request = compiled_model.create_infer_request()

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        input_data = preprocess_frame(frame, input_layer.shape)

        # Start async inference
        infer_request.start_async(inputs={input_layer: input_data})

        # Wait for inference completion
        infer_request.wait()
        output = infer_request.output_tensors[0].data

        # Compute FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Top-5 predictions
        top5_idx = output[0].argsort()[-5:][::-1]
        top5_text = ", ".join([f"{i}:{output[0][i]:.2f}" for i in top5_idx])

        # Overlay FPS and predictions
        cv2.putText(frame, f"FPS: {fps:.2f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Top-5: {top5_text}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Display frame
        cv2.imshow("Inference", frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
