from fastapi import FastAPI, UploadFile, File
from engine.loader import ModelEngine
from engine.doc_loader import load_txt

app = FastAPI()

engine = ModelEngine()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/generate")
def generate(prompt: str):
    response = engine.generate(prompt)
    return {"response": response}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()

    path = f"docs/{file.filename}"

    with open(path, "wb") as f:
        f.write(content)

    chunks = load_txt(path)
    engine.rag.add_documents(chunks)

    return {"status": "indexed", "chunks": len(chunks)}


@app.post("/reset")
def reset():
    engine.rag.reset()
    return {"status": "rag reset"}
