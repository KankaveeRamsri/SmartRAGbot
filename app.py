import os
import json
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

user_history = list()
history_file = "user_history.json"

# ฟังก์ชันสำหรับอ่านข้อมูลจากไฟล์ JSON
def load_user_history():
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ฟังก์ชันสำหรับบันทึกข้อมูลลงไฟล์ JSON
def save_user_history(user_history):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(user_history, f, ensure_ascii=False, indent=4)

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
        - Then answer the user’s question clearly in **Thai only**
        - Do not switch language even if there are other languages in the context
        - Use simple and friendly language
        - If possible, include relevant emojis like 😊📘❤️

        Question: {question}
        Answer:
        """.strip()

def make_rag_answer(vectorstore: Chroma, chat_llm: ChatOllama, question: str, k: int = 5) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    
     # ตรวจสอบว่าไม่มีข้อมูลที่เกี่ยวข้อง
    if not docs:
        answer = "ขอโทษค่ะ, ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารนี้"
    
    def clean_context(context: str) -> str:
        banned_patterns = ["<im_start>", "<im_end>", "<|im_start|>", "<|im_end|>"]
        for pattern in banned_patterns:
            context = context.replace(pattern, "")
        return context

    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "[No document found]"
    context = clean_context(context)
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
    
    user_history = load_user_history()

    if len(user_history) >= 5:
        user_history = []
    
    user_history.append({"question": user_text,
                         "answer": None})

    if user_text.lower() in {"/help", "help"}:
        help_msg = (
            "สวัสดีครับ/ค่ะ! ฉันคือบอทผู้ช่วยแนะนำ 'คู่มือความสุข 5 มิติสำหรับผู้สูงอายุ' 🧓💬\n\n"
            "มิติแห่งความสุข:\n"
            "- 🛋️ สุขสบาย : อยู่ดี กินดี มีความปลอดภัย\n"
            "- 🎉 สุขสนุก : มีกิจกรรมที่ชอบ สนุกกับชีวิต\n"
            "- 👑 สุขสง่า : มีคุณค่า ภูมิใจในตนเอง\n"
            "- 💡 สุขสว่าง : ไม่หยุดเรียนรู้ เปิดใจรับสิ่งใหม่\n"
            "- 🕊️ สุขสงบ : เย็นใจ เป็นสุขจากภายใน\n\n"
            "พิมพ์ชื่อมิติที่สนใจเพื่อให้ฉันช่วยแนะนำได้เลยครับ/ค่ะ 😊"
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
    
    # ตรวจสอบคำตอบ
    if "ไม่ทราบ" in answer or "ไม่มีข้อมูล" in answer or "ไม่มีความเกี่ยวข้อง" in answer or "ไม่เกี่ยวข้อง" in answer:
        answer = "ขอโทษค่ะ, ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"

    user_history[-1]["answer"] = answer

    print("Current Length User History: {}".format(len(user_history)))
    
    save_user_history(user_history)
    
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
