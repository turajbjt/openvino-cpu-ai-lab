## OpenVINO Repo – Python Script Style Guide

1. **Module & File-Level Docstrings**

Every Python file should start with a module docstring.

Include:
* File purpose
* Key features
* Example usage

Example:

```bash
""
run_inference.py

Batch image inference using OpenVINO CPU.

Features:
- Multiple image batch support
- Top-5 predictions per image
- Configurable CPU threads and streams
- Warm-up for stable timing

Usage:
python run_inference.py --model models/resnet50.xml --inputs data/img1.jpg data/img2.jpg --threads 4 --streams 1
"""
```

2. **Function & Class Docstrings**

Every function or class must have a docstring explaining:

* Purpose
* Arguments (Args)
* Return values (Returns)

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

3. **Inline Comments**

Use inline comments for non-obvious operations, such as:

* Reshaping arrays (HWC → CHW)
* CPU optimization options (threads/streams)
* Warm-up inference
* Async inference steps

Keep comments brief but explanatory, one idea per comment.

```bash
# Convert HWC image to CHW format
image_transposed = image_resized.transpose((2,0,1))
```

4. **CLI Argument Conventions**

Use ``argparse`` with:

* Descriptive names (``--model``, ``--inputs``, ``--threads``)
* ``help`` description for each argument
* Default values when appropriate

Example:

```bash
parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
```

5. **CPU Optimization Documentation**

Clearly document any CPU-specific options:
* CPU_THREADS_NUM → number of CPU threads
* CPU_THROUGHPUT_STREAMS → number of inference streams
Include in function docstrings and inline comments.

```bash
ie.set_property({
    "CPU_THREADS_NUM": num_threads,
    "CPU_THROUGHPUT_STREAMS": num_streams
})
```

6. **Preprocessing & Input Handling**

Always document input requirements:
* Supported image/video formats
* Resizing and normalization assumptions

Functions should handle errors gracefully:
* Missing files
* Unsupported formats

7. **Warm-Up & Timing**

* Always perform a warm-up inference to stabilize timing.
* Use ``time.time()`` for measuring inference.
* Document why warm-up is needed (CPU JIT, cache, etc.)

```bash
# Warm-up run to stabilize CPU performance
_ = compiled_model([input_data])[compiled_model.outputs[0]]
```

8. **Async Inference (Video)**

* Use ``create_infer_request()`` and ``start_async`` + ``wait()`` for video/webcam.
* Document FPS calculation and overlay logic.

```bash
# Start async inference for next frame
infer_request.start_async(inputs={input_layer: input_data})
infer_request.wait()  # Wait until inference finishes
```

9. **Output Formatting**

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

10. **General Python Style**

Follow **PEP8**:
* 4-space indentation
* Max line length ~88-100 chars
* Spaces around operators

Use **meaningful variable names**:

* ``compiled_model``, ``input_layer``, ``sample_input``, ``infer_request``

Consistent ordering:
* 1 - Imports
* 2 - Constants/config
* 3 - Functions
* 4 - Main block (``if __name__ == "__main__":``)

11. **Optional Extras**

* Add **error handling** for file not found or model load failure.
* Include **examples in docstrings** for each script.
* Keep **all scripts visually similar** (function order, docstring style, CLI format) for consistency.

