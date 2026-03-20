## OpenVINO CPU AI Lab

Repository for running CPU inference with OpenVINO on image and video data, including batch inference, async video inference, and automated benchmarking.

# Features

* Batch Image Inference with Top-5 predictions
* Video/Webcam Inference with FPS display
* Async Inference for smoother video demos
* CPU Benchmarking (single model and auto multi-model)
* Configurable threads, streams, and input sources
* Self-documenting scripts for easy maintainability

# **Setup**

1. *Clone Repo*

```bash
git clone https://github.com/turajbjt/openvino-cpu-ai-lab.git
cd openvino-cpu-ai-lab
```

2. *Install Dependencies*

```bash
pip install -r requirements.txt
```

Required packages:

* openvino
* opencv-python
* numpy
* argparse (standard library)


3. *Prepare Models*

Place your OpenVINO IR models (.xml and .bin) in the models/ folder.

Example:

``
models/resnet50.xml
models/resnet50.bin
``

4. *Prepare Sample Data*

* Image: ``data/sample.jpg``
* Video: ``data/sample.mp4`` (optional)


# **Scripts Overview**

| Script | Description | CLI Example |
| ------ | ----------- | ----------- |
| ``run_inference.py`` | Batch image inference with Top-5 predictions | ``python run_inference.py --model models/resnet50.xml --inputs data/img1.jpg data/img2.jpg --threads 4 --streams 2`` |
| run_video_inference.py | Real-time video/webcam inference with FPS and Top-5 predictions | ``python run_video_inference.py --model models/resnet50.xml --source 0 --threads 4 --streams 2`` |
| benchmark_model.py | Benchmark a single model with CPU threads and streams | ``python benchmark_model.py --model models/resnet50.xml --sample data/sample.jpg --threads 4 --streams 2`` |
| auto_benchmark.py | Automatically benchmark all models in a folder | ``python auto_benchmark.py --models_dir models --sample data/sample.jpg --threads 4 --streams 2`` |

**Usage Examples**

**Batch Image Inference**

```bash
python run_inference.py --model models/resnet50.xml --inputs data/img1.jpg data/img2.jpg --threads 4 --streams 2
```

Output:

``Inference time for batch of 2 images: 65.32 ms
Image: data/img1.jpg
Top-5 predictions:
  Class 243: 0.78
  Class 281: 0.12
  ...
``

# **Video/Webcam Inference**

```bash
python run_video_inference.py --model models/resnet50.xml --source 0 --threads 4 --streams 2
```

* Displays real-time FPS
* Shows Top-5 predictions per frame
* Press q to quit

# **Single Model Benchmark**

```bash
python benchmark_model.py --model models/resnet50.xml --sample data/sample.jpg --threads 4 --streams 2
```

Output:

``Benchmark result for models/resnet50.xml: 12.67 ms per inference
``

# **Auto Benchmark All Models**

```bash
python auto_benchmark.py --models_dir models --sample data/sample.jpg --threads 4 --streams 2
```

Sample Output:

``=== CPU Benchmark Results (ms per inference) ===
mobilenet.xml: 12.67 ms
resnet50.xml: 23.45 ms
yolov8.xml: 45.32 ms
ssd.xml: 50.10 ms
``

* Results are sorted by fastest inference
* Works with images or video frames

# **Recommended Hardware & Performance Tips**

* Increase ``--threads`` for multi-core CPUs
* Increase ``--streams`` for CPU throughput optimization
* For video demos, use async inference (``run_video_inference.py``) for smoother FPS
* Consider **FP16** or **INT8** models for faster CPU inference

# **Folder Structure**

openvino-cpu-ai-lab/
* ─ models/              # OpenVINO IR models (.xml, .bin)
* ─ data/                # Sample images/videos
* ─ run_inference.py     # Batch image inference
* ─ run_video_inference.py # Video/webcam inference
* ─ benchmark_model.py   # Single-model benchmark
* ─ auto_benchmark.py    # Multi-model automatic benchmark
* ─ requirements.txt
* ─ README.md

# Contributing

* Follow the Style Guide, for docstrings, inline comments, and CLI conventions

