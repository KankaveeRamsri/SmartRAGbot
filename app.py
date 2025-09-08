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
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "3"))

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "10cc7f532a62b2208f2bdeb03148705d")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "o0rmXIz8Xk1QDlHDkPbgLglKWg+qXjzOPnJt/21VmAXGBYuXkFQKlIyt71CpXQrAndBq5tsDAoj9BL+UUiVqkXHj7X1LeM7kRUfoBAgcbTzfo+3me0MPhMcFyF0Hpo1zdrRhbvhzSb5fsbVRURAeVgdB04t89/1O/w1cDnyilFU=")

def build_chat_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "granite3.3:latest")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

def build_prompt(context: str, question: str) -> str:
    return f"""
        Context:
        {context}

        Role: You are an engineer.
        Task:
        - Use a warm and friendly tone
        - Answer in English language.
        - Summarize the information clearly and concisely
        - Make it easy to understand, even for beginners
        - Include relevant emojis such as 🔋☀️🔌 when appropriate

        Question: {question}
        Answer:
        """.strip()

def make_rag_answer(vectorstore: Chroma, chat_llm: ChatOllama, question: str, k: int = 3) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "[No document found]"
    prompt = build_prompt(context=context, question=question)
    response = chat_llm.invoke(prompt)
    answer = getattr(response, "content", None) or str(response)
    return answer.strip() if answer else "[ERROR] Empty response from LLM."