import gradio as gr
import requests

API_URL = "http://localhost:8000"


def chat(message, history):
    res = requests.post(
        f"{API_URL}/generate",
        params={"prompt": message}
    )
    return res.json()["response"]


def upload_file(file):
    files = {"file": open(file.name, "rb")}
    res = requests.post(f"{API_URL}/upload", files=files)
    return res.json()


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local CPU AI (RAG)")

    chatbot = gr.ChatInterface(fn=chat)

    file_upload = gr.File(label="Upload document")
    upload_output = gr.JSON()

    file_upload.upload(upload_file, file_upload, upload_output)

demo.launch()
