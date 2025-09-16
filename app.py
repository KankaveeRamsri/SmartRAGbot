import os
import json
import time
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
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

def build_prompt(context: str, question: str, history: List[dict]) -> str:
    history_text = ""
    for h in history[-3:]:  # ใช้แค่ 3 บทสนทนาล่าสุด
        q = h.get("question", "")
        a = h.get("answer", "")
        history_text += f"ผู้ใช้: {q}\nผู้ช่วย: {a}\n"

    return f"""
        Context:
        {context}

        Previous conversation:
        {history_text}

        Current question:
        {question}

        Role: You are a helpful assistant.

        Task:
        - Analyze the above Thai context
        - Then answer the user’s question clearly in **Thai only** using simple and friendly language
        - Do not switch language even if there are other languages in the context
        - If the question is unrelated to the context, say clearly: "ขอโทษค่ะ คำถามนี้อยู่นอกเหนือจากเนื้อหาในเอกสาร"
        - If possible, include relevant emojis like 😊📘❤️

        Question: {question}
        Answer:
        """.strip()

def make_rag_answer(vectorstore: Chroma, chat_llm: ChatOllama, question: str, history: List[dict], k: int = 3) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    
    def clean_context(context: str) -> str:
        banned_patterns = ["<im_start>", "<im_end>", "<|im_start|>", "<|im_end|>"]
        for pattern in banned_patterns:
            context = context.replace(pattern, "")
        return context

    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "[No document found]"
    context = clean_context(context)

    prompt = build_prompt(context=context, question=question,  history=history)
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
    
    if user_text.lower() not in {"ประวัติ", "history"}:
        user_history.append({"question": user_text, "answer": None})

    
    start_time = time.time()
    
    if user_text.lower() in {"ประวัติ", "history"}:
        if not user_history:
            reply_text = "ยังไม่มีประวัติการถามตอบค่ะ 😊"
        else:
            reply_lines = ["📜 ประวัติการถาม-ตอบล่าสุด:"]
            for idx, entry in enumerate(user_history[-5:], start=1):  # เอา 5 อันล่าสุด
                q = entry.get("question", "-")
                a = entry.get("answer", "-")
                reply_lines.append(f"{idx}. ❓ {q}\n   ➡️ {a}")
            reply_text = "\n\n".join(reply_lines)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        return
    
        # ✅ คำสั่งล้างประวัติ
    if user_text.lower() in {"ล้างประวัติ", "เคลียร์", "clear"}:
        user_history = []
        save_user_history(user_history)  # เขียนไฟล์ใหม่เป็น []
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🧹 ล้างประวัติเรียบร้อยแล้วค่ะ")
        )
        return

    # ตรวจจับคำทักทายทั่วไป
    greetings = ["สวัสดี", "hello", "hi", "ดีจ้า", "หวัดดี", "ดีครับ", "ดีค่ะ"]
    if user_text.strip() in greetings:
        greetings_msg = (
            "สวัสดีครับ/ค่ะ 😊 ฉันคือบอทผู้ช่วยที่จะช่วยให้คุณเข้าใจเนื้อหาใน 'คู่มือความสุข 5 มิติสำหรับผู้สูงอายุ' ได้ง่ายขึ้น 🧓📘\n\n"
            "คุณสามารถพิมพ์คำถามเกี่ยวกับคู่มือนี้เข้ามาได้เลย เช่น:\n"
            "- 'คู่มือนี้เกี่ยวกับอะไร?'\n"
            "- 'ช่วยอธิบายเรื่องสุขภาพจิตให้หน่อย'\n"
            "- 'มีข้อแนะนำเรื่องกิจกรรมไหม?'\n\n"
            "หากต้องการดูคำสั่งเพิ่มเติม ให้พิมพ์ /help ได้เลยนะครับ/ค่ะ 💬"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=greetings_msg))
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

    cached_answer = None
    for entry in user_history:
        if entry["question"].strip().lower() == user_text.lower():
            cached_answer = entry.get("answer")
            break

    if cached_answer:
        answer = cached_answer + "\n\n🔁 คำถามนี้เคยถามไว้แล้วค่ะ"
    else:
        answer = make_rag_answer(app.config["VECTORSTORE"], app.config["CHAT_LLM"], user_text, history=user_history, k=RETRIEVAL_K)
    
    # ตรวจสอบคำตอบ
    if "ไม่ทราบ" in answer or "ไม่มีข้อมูล" in answer or "ไม่มีความเกี่ยวข้อง" in answer or "ไม่เกี่ยวข้อง" in answer:
        answer = "ขอโทษค่ะ, ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"

    if any(word in answer for word in ["ฉันคิดว่า", "โดยทั่วไป", "ในความเห็นของฉัน", "อาจเป็นไปได้ว่า"]):
        answer = "ขอโทษค่ะ คำถามนี้ดูเหมือนจะไม่เกี่ยวข้องกับเนื้อหาในเอกสาร"

    elapsed_time = time.time() - start_time
    seconds_used = round(elapsed_time, 2)

    answer += f"\n\n⏱️ ตอบกลับใน {seconds_used} วินาที"

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
