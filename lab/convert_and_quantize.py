# lab/convert_and_quantize.py
import argparse
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face Model to OpenVINO INT4 IR")
    parser.add_argument("-m", "--model_id", type=str, required=True, help="Hugging Face model ID (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Local directory to save the quantized model")
    
    args = parser.parse_args()
    
    model_id = args.model_id
    save_dir = Path("/app/models") / args.output_dir

    print(f"[*] Downloading and converting {model_id}...")
    print(f"[*] This may take a while depending on model size and internet speed.")

    # 1. Load Model with INT4 quantization config
    # export=True forces conversion to OpenVINO format
    # quantization_config handles the INT4 weight compression
    model = OVModelForCausalLM.from_pretrained(
        model_id, 
        export=True, 
        compile=False, # Compile later for inference
        load_in_4bit=True, # Critical: Enables INT4 weight-only quantization
    )
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3. Save the OpenVINO IR Model locally
    print(f"[*] Saving INT4 model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"[+] Conversion complete. You can now run inference.")

if __name__ == "__main__":
    main()
