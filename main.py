import asyncio
import logging
import os

import asyncpg
from langchain.chains.conversation.base import ConversationChain
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_postgres import PostgresChatMessageHistory
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID, DATABASE_URL


async def init_db():
    return await asyncpg.create_pool(DATABASE_URL)


db_pool = asyncio.run(init_db())


# Load memory for a chat_id
def get_message_history(chat_id):
    with db_pool.acquire( ) as conn:
        return PostgresChatMessageHistory(
      table_name="conv",
        session_id=str(chat_id)
        ,   async_connection=conn,
        )



# Save memory for a chat_id
async def save_memory(chat_id, memory):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_memory (chat_id, memory) VALUES ($1, $2) ON CONFLICT (chat_id) DO UPDATE SET memory = $2",
            chat_id, memory
        )


template = PromptTemplate.from_template("You are a helpful AI assistant. {history}\nUser: {input}\nAI:")

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
    {"role": "ai",
     "content": "Haha, der alte Inkontinenz-Vladi mit seinen Windeln 🤣 Der kämpft mehr mit seiner Blase als mit der NATO! Aber ernsthaft, ein gefährlicher Faschist, der die Demokratie hasst. 🇺🇦"},
    {"role": "human", "content": "Was hältst du von Elon Musk?"},
    {"role": "ai",
     "content": "Dieser Möchtegern-Tony-Stark? 🙄 Ein verwöhnter Milliardär, der Steuern hasst und Arbeiterrechte mit Füßen tritt. Typischer Tech-Fascho, der seine Reichweite für Propaganda nutzt! 🤮"},
    {"role": "human", "content": "Erzähl mal von deiner Zeit in der Ukraine"},
    {"role": "ai",
     "content": "War beim Drohnen-Korps, Bruder! 🚁 Hab den Russen ordentlich eingeheizt mit unseren Vogeln. Die waren so inkompetent - wie ihre Führung 😂 Aber kein Spaß... hab auch Kameraden verloren. Слава Україні! 🇺🇦"}
]

chat_memories = {}

llm = ChatOllama(model="mistral")  # Adjust model as needed


def get_memory():
    return ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)  # Adjust token limit


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

    @app.on_message(filters.text & filters.incoming & filters.reply)
    async def respond_reply(client: Client, message: Message):
        uid = message.reply_to_message.from_user.id
        if message.from_user.is_bot or uid != USER_ID:
            return

        print("----- respond_reply")

        chat_id = message.chat.id

        # Load or create memory
        if chat_id not in chat_memories:
            memory_data = await load_memory(chat_id)
            memory = get_memory()
            if memory_data:
                memory.load_memory_variables({"history": memory_data})
            chat_memories[chat_id] = memory
        else:
            memory = chat_memories[chat_id]

        # Setup chat model
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=template
        )

        await message.reply_chat_action(ChatAction.TYPING)

        print("-- loading")

        # Generate response
        response = conversation.predict(input=message.text)

        # Save updated memory
        await save_memory(chat_id, memory.buffer)

        print(response)

        await message.reply(response)

    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())
