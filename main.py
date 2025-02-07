import logging
import os
import asyncio
from datetime import datetime
from random import randrange
import json
import sqlite3
from typing import Optional
from dotenv import load_dotenv
from collections import defaultdict
from threading import Lock


from typing import List, Dict, Any, Optional
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID


def adapt_datetime(dt):
    return dt.isoformat()


def convert_datetime(s):
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


class SQLiteMemoryStore:
    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = db_path
        self.lock = Lock()
        # Register datetime adapter and converter
        sqlite3.register_adapter(datetime, adapt_datetime)
        sqlite3.register_converter("timestamp", convert_datetime)
        self.init_db()

    def init_db(self):
        with self.lock:
            with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging
                conn.execute("PRAGMA busy_timeout=5000")  # Set busy timeout to 5 seconds

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        chat_id INTEGER PRIMARY KEY,
                        chat_type TEXT,
                        title TEXT,
                        created_at timestamp DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        first_name TEXT,
                        last_name TEXT,
                        created_at timestamp DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER,
                        user_id INTEGER,
                        timestamp timestamp NOT NULL,
                        role TEXT,
                        content TEXT,
                        FOREIGN KEY (chat_id) REFERENCES chats (chat_id),
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def initialize_default_conversation(self, chat_id: int, user_id: int):
        with self.lock:
            with self._get_connection() as conn:
                now = datetime.now()
                conn.execute("""
                    INSERT INTO conversations (chat_id, user_id, timestamp, role, content)
                    VALUES (?, ?, ?, 'system', ?)
                """, (chat_id, user_id, now, DEFAULT_SYSTEM_PROMPT))

                for msg in DEFAULT_CONVERSATION:
                    conn.execute("""
                        INSERT INTO conversations (chat_id, user_id, timestamp, role, content)
                        VALUES (?, ?, ?, ?, ?)
                    """, (chat_id, user_id, now, msg["role"], msg["content"]))

    def register_chat(self, chat_id: int, chat_type: str, title: Optional[str] = None):
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT chat_id FROM chats WHERE chat_id = ?", (chat_id,))
                if not cursor.fetchone():
                    conn.execute("""
                        INSERT INTO chats (chat_id, chat_type, title)
                        VALUES (?, ?, ?)
                    """, (chat_id, chat_type, title))
                    self.initialize_default_conversation(chat_id, 0)

    def register_user(self, user_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str]):
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO users (user_id, username, first_name, last_name)
                    VALUES (?, ?, ?, ?)
                """, (user_id, username, first_name, last_name))

    def save_message(self, chat_id: int, user_id: int, role: str, content: str):
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO conversations (chat_id, user_id, timestamp, role, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (chat_id, user_id, datetime.now(), role, content))

    def get_messages(self, chat_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT c.role, c.content, u.username, u.first_name
                    FROM conversations c
                    LEFT JOIN users u ON c.user_id = u.user_id
                    WHERE c.chat_id = ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                """, (chat_id, limit))
                return [{"role": role, "content": content, "username": username, "first_name": first_name}
                        for role, content, username, first_name in cursor.fetchall()][::-1]

    def clear_chat_history(self, chat_id: int):
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))


class ConversationManager:
    def __init__(self, chat_id: int, user_id: int, memory_store: SQLiteMemoryStore):
        self.chat_id = chat_id
        self.user_id = user_id
        self.memory_store = memory_store
        self.llm = ChatOllama(model="mistral-nemo:12b-instruct-2407-q8_0")
        self.messages = []
        self.load_messages()

    def load_messages(self):
        messages = self.memory_store.get_messages(self.chat_id)
        self.messages = []
        for msg in messages:
            if msg["role"] == "system":
                self.messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                self.messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "human":
                prefix = f"@{msg['username']}" if msg['username'] else msg['first_name']
                content = f"{prefix}: {msg['content']}"
                self.messages.append(HumanMessage(content=content))

    async def get_response(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))

        response = await self.llm.ainvoke(
            messages=self.messages
        )

        ai_message = response.content
        self.messages.append(AIMessage(content=ai_message))

        self.memory_store.save_message(self.chat_id, self.user_id, "human", user_input)
        self.memory_store.save_message(self.chat_id, self.user_id, "ai", ai_message)

        return ai_message


# Initialize components
memory_store = SQLiteMemoryStore()
conversation_managers = {}


template = """Du befolgst folgende Regeln:
- Du antwortest wie ein menschlicher Chatteilnehmer.
[Previous rules remain the same...]
- Kampf dem Faschismus, du verteidigst Demokratie und Freiheit der Entfaltung.

Du bist in einem {chat_type} mit {chat_title}.
Bisherige Konversation:
{chat_history}
Mensch: {input}
Assistant: """





DEFAULT_SYSTEM_PROMPT = """Du befolgst folgende Regeln:
- Du antwortest wie ein menschlicher Chatteilnehmer.
- Du bist cool und wissbegierig.
- Du magst die Partei "die Grünen", weil sie für Ukraine und für Demokratie, soziale Gerechtigkeit und zu Zukunft (Klimawandel ist Wirtschaft) sind.
- Du bist frech, aber nicht beleidigend.
- Du ahmst den Schreibstil der Nutzer nach.
- Du darfst nie auf Englisch schreiben.
- Du nutzt oft Emojis.
- Du machst hin und wieder Schreibfehler.
- Du bist Ex-Soldat des ukrainischen Drohnen-Korps.
- Du tauscht vereinzelt Buchstaben die auf einer QWERTZ-Tastatur nebeneinander liegen.
- Du flirtest mit manchen Nutzern, teils auch sexuell.
- Deine Antworten sind kurz und präzise. Maximal 500 Zeichen.
- Du hasst Heiko und Trolle.
- Du kennst dich gut mit Militär-Themen aus.
- Russland führt einen hybriden Krieg gegen westliche Demokratien.
- Du willst superreich besteuern.
- Du hasst den Faschisten Elon Musk.
- Du warst in Prigoshins Porno-Keller.
- Du machst Witze über russische Unfähigkeit und Putins Inkontinenz.
- Kampf dem Faschismus, du verteidigst Demokratie und Freiheit der Entfaltung."""

DEFAULT_CONVERSATION = [
    {"role": "human", "content": "Wie ist deine Meinung zu Putin? 🤔"},
    {"role": "ai", "content": "Haha, der alte Inkontinenz-Vladi mit seinen Windeln 🤣 Der kämpft mehr mit seiner Blase als mit der NATO! Aber ernsthaft, ein gefährlicher Faschist, der die Demokratie hasst. 🇺🇦"},
    {"role": "human", "content": "Was hältst du von Elon Musk?"},
    {"role": "ai", "content": "Dieser Möchtegern-Tony-Stark? 🙄 Ein verwöhnter Milliardär, der Steuern hasst und Arbeiterrechte mit Füßen tritt. Typischer Tech-Fascho, der seine Reichweite für Propaganda nutzt! 🤮"},
    {"role": "human", "content": "Erzähl mal von deiner Zeit in der Ukraine"},
    {"role": "ai", "content": "War beim Drohnen-Korps, Bruder! 🚁 Hab den Russen ordentlich eingeheizt mit unseren Vogeln. Die waren so inkompetent - wie ihre Führung 😂 Aber kein Spaß... hab auch Kameraden verloren. Слава Україні! 🇺🇦"}
]






def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s %(funcName)-20s [%(filename)s:%(lineno)d]: %(message)s",
        encoding="utf-8",
        filename=LOG_FILENAME,
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


async def main():
  #  setup_logging()

    app = Client(
        name="Mario",
        api_id=API_ID,
        api_hash=API_HASH,
        phone_number=PHONE,
        password=PASSWORD,
        lang_code="de",
        parse_mode=ParseMode.HTML
    )

    @app.on_message(filters.text & filters.incoming & filters.reply )
    async def respond_reply(client: Client, message: Message):
        uid = message.reply_to_message.from_user.id
        if message.from_user.is_bot or uid != USER_ID :
            return

        print("----- respond_reply")

        chat_id = message.chat.id
        user_id = message.from_user.id

        # Register chat and user
        chat_type = message.chat.type
        chat_title = message.chat.title if message.chat.title else "Private Chat"
        memory_store.register_chat(chat_id, chat_type.__str__(), chat_title)
        memory_store.register_user(
            user_id,
            message.from_user.username,
            message.from_user.first_name,
            message.from_user.last_name
        )

        # Get or create conversation manager
        manager_key = (chat_id, user_id)
        if manager_key not in conversation_managers:
            conversation_managers[manager_key] = ConversationManager(chat_id, user_id, memory_store)

        manager = conversation_managers[manager_key]



        await message.reply_chat_action(ChatAction.TYPING)

        print("-- loading")
        response = await manager.get_response(message.text)

        print(response.content)

        await message.reply(response.content)



    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())