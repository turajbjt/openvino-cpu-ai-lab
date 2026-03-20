from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

from engine.config import MODEL_DIR, DEVICE
from engine.rag import RAGEngine
from engine.doc_loader import load_txt


class ModelEngine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        self.model = OVModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device=DEVICE
        )

        self.rag = RAGEngine()

        # preload docs if available
        try:
            chunks = load_txt("docs/knowledge.txt")
            if chunks:
                self.rag.add_documents(chunks)
                print(f"[RAG] Loaded {len(chunks)} chunks")
        except FileNotFoundError:
            print("[RAG] No default docs found")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        context_chunks = []
        if self.rag.index is not None:
            context_chunks = self.rag.search(prompt, k=3)

        context = "\n\n".join(context_chunks)

        full_prompt = f"""### Instruction:
Use the context to answer the question.

### Context:
{context}

### Question:
{prompt}

### Answer:
"""

        inputs = self.tokenizer(full_prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # strip prompt from output (important)
        if "### Answer:" in decoded:
            return decoded.split("### Answer:")[-1].strip()

        return decoded.strip()e
