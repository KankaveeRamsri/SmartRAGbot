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