## OpenVINO CPU AI Lab – Style Guide

# 1. **Required Packages**

All scripts in the repo rely on the following Python packages:

| Package | Purpose |
| ------- | ------- |
| ``openvino`` | Core OpenVINO runtime for model compilation and inference |
| ``opencv-python`` |	Image/video preprocessing and display |
| ``numpy`` |	Array manipulation and batch processing |
| ``argparse`` |	Command-line argument parsing (standard library) |

**Installation:**

```bash
pip install -r requirements.txt
```

# 2. **Module & File-Level Docstrings**

Every Python file must start with a module docstring.

Include:
* File purpose
* Key features
* Example usage

Example:

```bash
"""
run_inference.py

Batch image inference using OpenVINO CPU.

Features:
- Multiple image batch support
- Top-5 predictions per image
- Configurable CPU threads and streams
- Warm-up for stable timing

Usage:
python run_inference.py --model models/resnet50.xml --inputs data/img1.jpg data/img2.jpg --threads 4 --streams 2
"""
```

# 3. **Function & Class Docstrings**

Use **triple-quoted docstrings** for every function or class.

Include:
* Purpose
* Args
* Returns

Example:

```bash
def preprocess_image(image_path: str, input_shape):
    """
    Read and preprocess a single image for model input.

    Args:
        image_path (str): Path to the image file
        input_shape (tuple): Model input shape (batch, channels, height, width)

    Returns:
        np.ndarray: Preprocessed image array ready for inference
    """
```

# 4. **Inline Comments**

Explain **non-obvious logic**, e.g.:
* HWC → CHW reshaping
* CPU threads/streams configuration
* Warm-up runs
* Async inference

Keep comments concise and one idea per line.

```bash
# Convert HWC image to CHW format
image_transposed = image_resized.transpose((2,0,1))
```

# 5. **CLI Argument Conventions**

Use ``argparse`` with:
* Descriptive names: ``--model``, ``--inputs``, ``--threads``
* ``help`` description
* Default values when appropriate

Example:

```bash
parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
```

# 6. **CPU Optimization**

Document CPU-specific options clearly:
* ``CPU_THREADS_NUM``: Number of CPU threads
* ``CPU_THROUGHPUT_STREAMS``: Number of throughput streams

Include both in docstrings and inline comments.

```bash
ie.set_property({
    "CPU_THREADS_NUM": num_threads,
    "CPU_THROUGHPUT_STREAMS": num_streams
})
```

# 7. Input Preprocessing

Always document input requirements:
* Supported formats (.jpg, .png, .mp4, etc.)
* Resizing, normalization assumptions

Handle errors gracefully:
* Missing files
* Unsupported formats

# 8. **Warm-Up & Timing**

* Perform a warm-up inference before measuring inference time to stabilize CPU performance.
* Use ``time.time()`` for timing.
* Inline comment why warm-up is needed (JIT compilation, caching).

```bash
# Warm-up run to stabilize CPU performance
_ = compiled_model([input_data])[compiled_model.outputs[0]]
```

# 9. **Async Inference (Video)**

* Use ``create_infer_request()``, ``start_async()``, and ``wait()`` for video/webcam scripts.
* Document FPS calculation and overlay logic.

```bash
# Start async inference for next frame
infer_request.start_async(inputs={input_layer: input_data})
infer_request.wait()  # Wait until inference finishes
```

# 10. **Output Formatting**

Print results clearly:
* Top-5 predictions for images
* Sorted benchmark times
* FPS for video

Example:

``Image: data/img1.jpg
Top-5 predictions:
  Class 243: 0.78
  Class 281: 0.12
``

# 11. **General Python Style**

Follow **PEP8**:
* 4-space indentation
* Max line length ~88-100
* Spaces around operators

Meaningful variable names:
* ``compiled_model``, ``input_layer``, ``sample_input``, ``infer_request``

Script structure:
* 1 - Imports
* 2 - Constants/config
* 3 - Functions
* 4 - Main block (``if __name__ == "__main__":``)

# 12. **Contribution Guidelines**

* Follow this style guide for new scripts or updates.
* Include **docstrings**, **inline comments**, and **CLI examples**.
* Ensure all scripts handle errors gracefully and provide clear output.
* Maintain **consistent naming conventions** across all files.


