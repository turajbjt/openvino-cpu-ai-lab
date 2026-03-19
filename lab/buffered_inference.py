# lab/buffered_inference.py
import argparse
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Local OpenVINO INT4 Inference")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Name of the folder in /app/models/")
    
    args = parser.parse_args()
    model_path = Path("/app/models") / args.model_name

    print(f"[*] Loading model from {model_path} onto CPU...")
    
    # 1. Load the locally saved OpenVINO IR model.
    # OpenVINO automatically detects INT4 weights and optimizes accordingly.
    model = OVModelForCausalLM.from_pretrained(
        model_path, 
        device="CPU", # Explicitly set device
        ov_config={"PERFORMANCE_HINT": "LATENCY"}, # Tune for speed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. Setup streaming output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("[+] Model loaded. Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input.strip():
                continue

            # 3. Simple chat template (Adjust based on your specific model requirements)
            prompt = f"System: You are a helpful assistant.\nUser: {user_input}\nAssistant: "
            inputs = tokenizer(prompt, return_tensors="pt")

            # 4. Generate with streaming
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()

            print("Assistant > ", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)
            print() # Newline after generation completes

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
