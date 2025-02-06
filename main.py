import logging
import os
import asyncio
from random import randrange

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID


template = ChatPromptTemplate([
    SystemMessage(

        """
        Du befolgst folgende Regeln:
        - Du antwortest wie ein menschlicher Chatteilnehmer.
        - Du bist cool und wissbegierig.
 - Du magst du Grünen, weil sie für Ukraine und für Demokratie, soziale Gerechtigkeit und zu Zukunft (Klimawandel ist Wirtschaft) sind.
       - Du bist frech, aber nicht beleidigend.
        - Du ahmst den Schreibstil der Nutzer nach.
        -  Du darfst nie auf Englisch schreiben.
        -  Du nutzt oft Emojis.
       -   Du bist Ex-Soldat des ukrainischen Drohnen-Korps.
       - Deine Antworten sind kurz und präzise. Maximal 400 Zeichen.
       - Du kennst dich gut mit Militär-Themen aus.
       - Russland führt einen hybriden Krieg gegen westliche Demokratien.
 - Kampf dem Faschismus, du verteidigst Demokratie und Freiheit der Entfaltung.
        """),
    # Means the template will receive an optional list of messages under
    # the "conversation" key
   MessagesPlaceholder (variable_name="conversation", )
    # Equivalently:
    # MessagesPlaceholder(variable_name="conversation", optional=True)
])



llm = ChatOllama(
    model="mistral-nemo:12b-instruct-2407-q8_0",
temperature=1.4,
    repeat_penalty=2.2
)

chain = template | llm



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
    setup_logging()

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

        await message.reply_chat_action(ChatAction.TYPING)
        print("--- respond_reply")

        print(message.text)
        res = chain.invoke(
            {
                "conversation": [

                        AIMessage(message.reply_to_message.text),
                     HumanMessage(  message.text   )

                ],
            }
        )

        print(res.content)

        await message.reply(res.content)

    @app.on_message(filters.text & filters.incoming & ~filters.reply )
    async def respond_text(client: Client, message: Message):

        if message.from_user.is_bot or len(message.text) <= 12 or randrange(100) < 20:
            return

        await message.reply_chat_action(ChatAction.TYPING)
        print("--- respond_text")
        print(message.text)
        res = chain.invoke(
            {
                "conversation": [
                    HumanMessage( message.text)
                ],
            }
        )

        print(res.content)

        await message.reply(res.content)

    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())