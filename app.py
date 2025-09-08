import os
from typing import List
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama

PDF_PATH = os.environ.get("RAG_PDF_PATH", "C:\\Users\\student\\Desktop\\6610110408\\Miniproject-social-2\\pdf-files\\5-dimentions-happiness.pdf")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "C:\\Users\student\\Desktop\\6610110408\\Miniproject-social-2\\chroma_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "10cc7f532a62b2208f2bdeb03148705d")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "o0rmXIz8Xk1QDlHDkPbgLglKWg+qXjzOPnJt/21VmAXGBYuXkFQKlIyt71CpXQrAndBq5tsDAoj9BL+UUiVqkXHj7X1LeM7kRUfoBAgcbTzfo+3me0MPhMcFyF0Hpo1zdrRhbvhzSb5fsbVRURAeVgdB04t89/1O/w1cDnyilFU=")

def build_chat_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

def build_prompt(context: str, question: str) -> str:
    return f"""
        Context:
        {context}

        Role: You are a helpful assistant.

        Task:
        - Analyze the above Thai context
        - Then answer the user’s question clearly in English
        - Use simple and friendly language
        - If possible, include relevant emojis like 😊📘❤️

        Question: {question}
        Answer:
        """.strip()

def make_rag_answer(vectorstore: Chroma, chat_llm: ChatOllama, question: str, k: int = 5) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "[No document found]"
    prompt = build_prompt(context=context, question=question)
    response = chat_llm.invoke(prompt)
    answer = getattr(response, "content", None) or str(response)
    return answer.strip() if answer else "[ERROR] Empty response from LLM."

app = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/", methods=["POST"]) 
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage) 
def handle_message(event: MessageEvent):
    user_text = (event.message.text or "").strip()
    if not user_text:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="(empty message)"))
        return

    if user_text.lower() in {"/help", "help"}:
        help_msg = (
            "Hi! Send me a question about the PDF and I'll answer using RAG.\n\n"
            "Commands:\n"
            "- /source : show PDF + embedding info\n"
            "- /id : echo message id\n"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=help_msg))
        return

    if user_text.lower() == "/source":
        info = f"Indexed PDF: {os.path.basename(PDF_PATH)}\nEmbeddings: {EMBED_MODEL_NAME}\nTop-k: {RETRIEVAL_K}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=info))
        return

    if user_text.lower() == "/id":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"msg id: {event.message.id}"))
        return

    answer = make_rag_answer(app.config["VECTORSTORE"], app.config["CHAT_LLM"], user_text, k=RETRIEVAL_K)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n… (truncated)"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))

if __name__ == "__main__":
    print("[BOOT] Loading vectorstore…")
    embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    print("[BOOT] Initializing chat LLM…")
    chat_llm = build_chat_llm()

    app.config["VECTORSTORE"] = vectorstore
    app.config["CHAT_LLM"] = chat_llm

    port = int(os.environ.get("PORT", "5000"))
    print(f"[RUN] Flask listening on Localhost:{port}")
    app.run(port=port)
