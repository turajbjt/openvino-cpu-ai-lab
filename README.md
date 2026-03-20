# OpenVINO CPU AI Lab

This repository demonstrates CPU-based AI inference using OpenVINO, including image classification, object detection, and segmentation. It supports live video, webcam inference, and CPU-optimized benchmarking.

**Features**

* Efficient CPU inference with OpenVINO
* Multi-model support:
  - Classification: ResNet, MobileNet, EfficientNet
  - Object Detection: YOLOv8, SSD, Faster R-CNN
  - Segmentation: DeepLabv3, Mask R-CNN
* Benchmark scripts with configurable CPU threads & streams
* Video and webcam inference with real-time FPS overlay
* Top-5 predictions displayed for better insight
* Model conversion scripts (ONNX → OpenVINO IR)
* Optional Flask web interface for live demos

## Prerequisites

* Intel CPU (Ideally Gen 11+ Core or Xeon Gen 3+).
* Docker & Docker Compose installed.

## 1. Initial Setup

Clone this repository (or your fork) and build the optimized Docker image.

```bash
# Clone the repository
git clone https://github.com/turajbjt/openvino-cpu-ai-lab.git
cd openvino-cpu-ai-lab

# Install dependencies:
pip install -r requirements.txt

# (Optional) If you want to run the web interface:
export FLASK_APP=app.py
flask run
```

---

**Usage**

1. Image Inference:

```bash
python run_inference.py --model models/resnet50.xml --input data/sample.jpg --threads 4 --streams 1
```

Output:

``Inference time: 25.43 ms
Top-5 predictions:
  Class 243: 0.78
  Class 281: 0.12
  Class 340: 0.05
  Class 12: 0.03
  Class 66: 0.02
``

2. Video/Webcam Inference:

```bash
python run_video_inference.py --model models/yolov8.xml --source 0 --threads 4 --streams 2
```

* --source 0 → default webcam
* --source <video_path> → video file
* FPS and Top-5 predictions are displayed live on the video window
* Press q to quit

3. Benchmark CPU Performance:

```bash
python benchmark_model.py --model models/resnet50.xml --threads 4 --streams 2
```

* Measures CPU inference time
* Allows configurable threading and throughput streams

4. Auto Benchmark

```bash
python auto_benchmark.py --models_dir models --sample data/sample.jpg --threads 4 --streams 2
```

Sample Output:

``=== CPU Benchmark Results (ms per inference) ===
resnet50.xml: 23.45 ms
mobilenet.xml: 12.67 ms
yolov8.xml: 45.32 ms
ssd.xml: 50.10 ms
``

**Project Structure**

openvino-cpu-ai-lab/
* /models/                # IR models for OpenVINO
* /data/                  # Sample images or videos
* /scripts/               # Conversion and benchmark scripts
* run_inference.py        # Image inference with top-5 predictions
* run_video_inference.py  # Video/webcam inference with FPS & top-5
* /benchmark_model.py     # CPU benchmarking
* /app.py                 # Optional Flask web interface
* /requirements.txt
* /README.md

