# OpenVINO CPU AI Lab (INT4 Optimized)

A CPU-only AI Laboratory optimized for running local Large Language Models (LLMs) at peak speed using **Intel OpenVINO** and **INT4 Weight Quantization**.

This lab runs entirely within Docker and uses volume mounts so you can code locally on your host machine while the models run inside the optimized container.

## Prerequisites

* Intel CPU (Ideally Gen 11+ Core or Xeon Gen 3+).
* Docker & Docker Compose installed.

## 1. Initial Setup

Clone this repository (or your fork) and build the optimized Docker image.

```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd openvino-cpu-ai-lab

# Build the image (this installs OpenVINO dev tools)
docker compose build
```

2. Running the Lab

The lab uses volume mounts. Your local ``lab/`` and ``models/`` directories are mapped inside the container.

**Step A: Start the Lab Container**

This will start the container and drop you into a bash shell inside /app.

```bash
docker compose up -d  # Start in backend
docker attach openvino-ai-lab # Attach to interactive session
# OR: docker compose run --rm ai-lab # Start and enter directly
```

**Step B: Quantize a Model (INT4)**

Inside the container terminal, run the quantization script. This downloads a model from Hugging Face, converts it to OpenVINO format, and applies 4-bit quantization.
Bash

```bash
# Example: Quantize TinyLlama (fast download)
python3 lab/convert_and_quantize.py \
  --model_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output_dir "tinyllama-int4-ov"
```

Your quantized model is now stored locally on your host machine in ``models/tinyllama-int4-ov`` but is accessible inside the container.

**Step C: Run High-Speed Inference**

Now run the chat script, pointing to your newly created local model folder.

```bash
python3 lab/buffered_inference.py --model_name "tinyllama-int4-ov"
```

Performance Tuning Tips

OpenVINO automatically detects your Intel CPU capabilities (AVX-512, AMX). INT4 quantization maximizes performance by reducing the bottleneck of moving weights from memory to computation cores.

If you desire even more performance:

* Context Window: Reduce ``max_new_tokens`` in ``buffered_inference.py``.
* CPU Pinning: If running multiple containers, use Docker's ``--cpuset-cpus`` to dedicate specific physical cores to the AI Lab.
