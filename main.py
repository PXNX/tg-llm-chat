import asyncio
import logging
import os
import uuid

import transformers
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

from langchain.memory.summary_buffer import ConversationSummaryBufferMemory

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID, DATABASE_URL


sync_connection = psycopg.connect(DATABASE_URL)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)


# Load memory for a chat_id
def get_message_history(user_id: uuid.UUID):
    return PostgresChatMessageHistory(table_name,user_id.hex, sync_connection=sync_connection )

llm = ChatOllama(model="mistral-nemo:12b-instruct-2407-q8_0",temperature=1.2)






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
 HumanMessage   ( "Wie ist deine Meinung zu Putin? 🤔"),
   AIMessage ( "Haha, der alte Inkontinenz-Vladi mit seinen Windeln 🤣 Der kämpft mehr mit seiner Blase als mit der NATO! Aber ernsthaft, ein gefährlicher Faschist, der die Demokratie hasst. 🇺🇦"),
    HumanMessage  ( "Was hältst du von Elon Musk?"),
    AIMessage  ( "Dieser Möchtegern-Tony-Stark? 🙄 Ein verwöhnter Milliardär, der Steuern hasst und Arbeiterrechte mit Füßen tritt. Typischer Tech-Fascho, der seine Reichweite für Propaganda nutzt! 🤮"),
    HumanMessage   ( "Erzähl mal von deiner Zeit in der Ukraine"),
    AIMessage  ( "War beim Drohnen-Korps, Bruder! 🚁 Hab den Russen ordentlich eingeheizt mit unseren Vogeln. Die waren so inkompetent - wie ihre Führung 😂 Aber kein Spaß... hab auch Kameraden verloren. Слава Україні! 🇺🇦"),
]

chat_memories = {}

uuids = {}

def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s %(funcName)-20s [%(filename)s:%(lineno)d]: %(message)s",
        encoding="utf-8",
        filename=LOG_FILENAME,
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


prompt = ChatPromptTemplate.from_messages([
    ("system", "Do bist ein Chat-Mitglied."),
    ("user", "{input}")
])

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



        if chat_id not in uuids:
            uuids[chat_id] = uuid.uuid4()
        user_id =   uuids[chat_id]

        history = get_message_history(user_id)

        if chat_id not in chat_memories:
            history.add_messages(DEFAULT_CONVERSATION)
            memory = ConversationSummaryBufferMemory(
                llm=llm,
            chat_memory=history,
            return_messages=True,
                max_token_limit=120
            )
            chat_memories[chat_id] = memory
        else:
            memory=chat_memories[chat_id]



        chain =LLMChain(
            llm=llm,
            memory=memory,
            prompt=prompt
        )

        await message.reply_chat_action(ChatAction.TYPING)

        print("-- loading")

        # Generate response
        response = chain.invoke({"input": message.text})

        history.add_user_message(message.text)

        print(response)
        history.add_ai_message(response["text"])
        await message.reply(response["text"])

    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())
