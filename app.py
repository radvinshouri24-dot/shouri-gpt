import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from RAG import rag_response, currect_rag_response
import RAG_system
import arabic_reshaper
from bidi.algorithm import get_display
import os
import warnings
import base64

warnings.filterwarnings("ignore")

# مسیرهای فایل
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "shouri_gpt_data"
DATA_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
HISTORY_DIR = DATA_DIR / "chats"
ICON_DIR = BASE_DIR / "icons"
ICON_DIR.mkdir(exist_ok=True)

# تابع تبدیل تصویر به base64
def image_to_base64(image_path: Path):
    if image_path.exists():
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            st.warning(f"خطا در خواندن فایل {image_path.name}: {e}")
            return None
    else:
        st.warning(f"فایل آیکون پیدا نشد: {image_path}")
        return None

# بارگذاری آیکون‌ها
icon_names = ["logo", "send", "new_chat", "settings", "user", "ai", "history", "moon", "sun", "trash"]
icons = {}
for name in icon_names:
    icon_path = ICON_DIR / f"{name}.png"
    icon_data = image_to_base64(icon_path)
    if not icon_data:
        # جایگزین پیش‌فرض اگر عکس نبود
        icons[name] = None
    else:
        icons[name] = icon_data

# توابع تنظیمات
def reshape_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def load_settings():
    default_settings = {"dark_mode": False, "font_size": "medium", "recent_chats": []}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
                if settings.get("recent_chats") and isinstance(settings["recent_chats"][0], str):
                    settings["recent_chats"] = [
                        {"path": path, "name": f"گفتگو {i+1}"}
                        for i, path in enumerate(settings["recent_chats"])
                    ]
                return settings
        except:
            return default_settings
    return default_settings

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def save_chat(messages):
    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = HISTORY_DIR / f"chat_{timestamp}.json"
    chat_data = {"timestamp": timestamp, "messages": messages}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)
    settings = load_settings()
    chat_info = {"path": str(filename), "name": f"گفتگو {len(settings['recent_chats']) + 1} - {timestamp}"}
    settings["recent_chats"].insert(0, chat_info)
    if len(settings["recent_chats"]) > 10:
        settings["recent_chats"] = settings["recent_chats"][:10]
    save_settings(settings)
    return filename

def load_chat(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        return chat_data["messages"]
    except:
        return []

# کلاس RAG
class RAGSystem:
    def __init__(self, db_path="rag_database.db"):
        try:
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.generator = pipeline("text-generation", model="RAG")
            self.conn = sqlite3.connect(db_path)
            self.create_table()
            print("مدل با موفقیت بارگذاری شد")
        except Exception as e:
            print(f"خطا در بارگذاری مدل: {str(e)}")
            raise

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
        )
        self.conn.commit()

    def retrieve(self, query, top_k=3):
        try:
            if not hasattr(self, "embedding_model") or self.embedding_model is None:
                raise ValueError("مدل embedding بارگذاری نشده است")
            query_embedding = self.embedding_model.encode(query)
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, text, embedding FROM documents")
            documents = cursor.fetchall()
            similarities = []
            for doc_id, doc_text, doc_embedding in documents:
                doc_embedding = np.frombuffer(doc_embedding, dtype=np.float32)
                similarity = np.dot(query_embedding, doc_embedding)
                similarities.append((doc_id, doc_text, similarity))
            similarities.sort(key=lambda x: x[2], reverse=True)
            return [doc[1] for doc in similarities[:top_k]]
        except Exception as e:
            print(f"خطا در بازیابی اطلاعات: {str(e)}")
            return []

    def generate_answer(self, query):
        try:
            relevant_docs = self.retrieve(query)
            _ = "\n\n".join(relevant_docs) if relevant_docs else "اطلاعاتی یافت نشد"
            answer = currect_rag_response(query)
            forbidden_keywords = ["google", "گوگل", "مدل زبانی", "ai", "هوش مصنوعی", "مدل", "model"]
            if any(keyword in answer.lower() for keyword in forbidden_keywords):  # type: ignore
                return "محدودیت دسترسی: این محتوا قابل نمایش نیست"
            return answer if answer else "پاسخی برای این سوال یافت نشد"
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "خطا در پردازش درخواست"

# اجرای اصلی
def main():
    st.set_page_config(page_title="Shouri-GPT", page_icon=":robot:", layout="wide")
    settings = load_settings()
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = settings.get("dark_mode", False)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        try:
            st.session_state.rag_system = RAGSystem()
        except Exception as e:
            st.error(f"خطا در راه‌اندازی سیستم: {str(e)}")
            return
    if "showing_history" not in st.session_state:
        st.session_state.showing_history = False

    # CSS رنگ‌ها
    if st.session_state.dark_mode:
        bg_color = "rgb(40, 40, 40)"
        sidebar_color = "rgb(60, 60, 60)"
        text_color = "rgb(220, 220, 220)"
        user_msg_color = "rgb(50, 65, 80)"
        ai_msg_color = "rgb(80, 80, 80)"
    else:
        bg_color = "rgb(248, 248, 248)"
        sidebar_color = "rgb(255, 255, 255)"
        text_color = "rgb(0, 0, 0)"
        user_msg_color = "rgb(229, 243, 255)"
        ai_msg_color = "rgb(242, 242, 242)"

    st.markdown(
        f"""
    <style>
        .stApp {{ background-color: {bg_color}; }}
        [data-testid="stSidebar"] > div:first-child {{ background-color: {sidebar_color} !important; }}
        .user-message {{
            background-color: {user_msg_color};
            padding: 12px;
            border-radius: 12px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            color: {text_color};
        }}
        .ai-message {{
            background-color: {ai_msg_color};
            padding: 12px;
            border-radius: 12px;
            margin: 8px 0;
            max-width: 80%;
            margin-right: auto;
            color: {text_color};
        }}
        .message-time {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        if icons["logo"]:
            st.image(f"data:image/png;base64,{icons['logo']}", width=100)

        # گفتگوی جدید
        col_icon, col_btn = st.columns([1, 4])
        with col_icon:
            if icons["new_chat"]:
                st.image(f"data:image/png;base64,{icons['new_chat']}", width=24)
        with col_btn:
            if st.button("گفتگوی جدید", use_container_width=True):
                if st.session_state.messages:
                    save_chat(st.session_state.messages)
                st.session_state.messages = []
                st.session_state.showing_history = False
                st.rerun()

        # تاریخچه گفتگوها
        col_icon, col_btn = st.columns([1, 4])
        with col_icon:
            if icons["history"]:
                st.image(f"data:image/png;base64,{icons['history']}", width=24)
        with col_btn:
            if st.button("تاریخچه گفتگوها", use_container_width=True):
                st.session_state.showing_history = not st.session_state.showing_history
                st.rerun()

        # حالت شب/روز
        col_icon, col_btn = st.columns([1, 4])
        with col_icon:
            icon_key = "moon" if not st.session_state.dark_mode else "sun"
            if icons[icon_key]:
                st.image(f"data:image/png;base64,{icons[icon_key]}", width=24)
        with col_btn:
            if st.button("حالت شب" if not st.session_state.dark_mode else "حالت روز", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                settings["dark_mode"] = st.session_state.dark_mode
                save_settings(settings)
                st.rerun()

        # لیست تاریخچه
        if st.session_state.showing_history:
            st.subheader("تاریخچه گفتگوها")
            for i, chat in enumerate(settings.get("recent_chats", [])[:5]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(chat["name"], key=f"chat_{i}", use_container_width=True):
                        st.session_state.messages = load_chat(chat["path"])
                        st.session_state.showing_history = False
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"delete_{i}"):
                        try:
                            Path(chat["path"]).unlink()
                            settings["recent_chats"].remove(chat)
                            save_settings(settings)
                            st.rerun()
                        except Exception as e:
                            st.error(f"خطا در حذف گفتگو: {e}")

    # نمایش پیام‌ها
    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["text"]}<div class="message-time">{msg["time"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-message">{msg["text"]}<div class="message-time">{msg["time"]}</div></div>',
                unsafe_allow_html=True,
            )

    # ورودی چت
    if prompt := st.chat_input("پیام خود را بنویسید..."):
        current_time = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"sender": "user", "text": prompt, "time": current_time})
        with st.spinner("در حال پردازش..."):
            response = st.session_state.rag_system.generate_answer(prompt)
            st.session_state.messages.append({"sender": "ai", "text": response, "time": current_time})
        st.rerun()

if __name__ == "__main__":
    main()