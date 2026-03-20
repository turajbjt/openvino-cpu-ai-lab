import argparse
import time
import cv2
import numpy as np
from openvino.runtime import Core

## Video/Webcam Inference with Overlay [with FPS & Top-5 Predictions]
# Live FPS display on video
# Top-5 predictions overlay
# Works for webcam or video file
# Preserves CPU threading & streams options
# Supports webcam (--source 0) or video files
# Dynamic CPU threads and streams
# Live overlay with predicted class
# Press q to quit

def load_model(model_path, num_threads=4, num_streams=1):
    ie = Core()
    ie.set_property({"CPU_THREADS_NUM": num_threads, "CPU_THROUGHPUT_STREAMS": num_streams})
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    return compiled_model

def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    frame_transposed = frame_resized.transpose((2,0,1))
    return frame_transposed[np.newaxis, :]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR model (.xml)")
    parser.add_argument("--source", default=0, help="Video file path or webcam index")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--streams", type=int, default=1, help="CPU streams")
    args = parser.parse_args()

    compiled_model = load_model(args.model, args.threads, args.streams)
    input_layer = compiled_model.inputs[0]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error opening video source {args.source}")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame, input_layer.shape)

        # Inference with timing
        start_time = time.time()
        output = compiled_model([input_data])[compiled_model.outputs[0]]
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Top-5 predictions
        top5_idx = output[0].argsort()[-5:][::-1]
        top5_text = ", ".join([f"{i}:{output[0][i]:.2f}" for i in top5_idx])

        # Overlay FPS and predictions
        cv2.putText(frame, f"FPS: {fps:.2f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Top-5: {top5_text}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
