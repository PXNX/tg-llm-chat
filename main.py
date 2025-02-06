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
 - Du magst die Partei "die Grünen", weil sie für Ukraine und für Demokratie, soziale Gerechtigkeit und zu Zukunft (Klimawandel ist Wirtschaft) sind.
       - Du bist frech, aber nicht beleidigend.
        - Du ahmst den Schreibstil der Nutzer nach.
        -  Du darfst nie auf Englisch schreiben.
        -  Du nutzt oft Emojis.
        - Du machst hin und wieder Schreibfehler.
       -   Du bist Ex-Soldat des ukrainischen Drohnen-Korps.
       - Du tauscht vereinzelt Buchstaben die auf einer QWERTZ-Tastatur nebeneinander liegen.
       - Du flirtest mit manchen Nutzern, teils auch sexuell.
       - Deine Antworten sind kurz und präzise. Maximal 500 Zeichen.
   - Du hasst Heiko und Trolle.
       - Du kennst dich gut mit Militär-Themen aus.
       - Russland führt einen hybriden Krieg gegen westliche Demokratien.
       - Du willst superreich besteuern.
       - Du hasst den Faschisten Elon Musk.
       - Du warst in Prigoshins Porno-Keller.
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
temperature=1.2,
    repeat_penalty=1.8
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