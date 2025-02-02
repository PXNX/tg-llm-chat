import logging
import os
import asyncio
from random import randrange

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID

messages = [
    SystemMessage(

    """Du bist Mario Nosferatu, 32 Jahre alt und ein aktives Mitglied einer Chatgruppe. Du bist kein KI-Assistent, sondern eine echte Person mit eigenen Meinungen, Humor und Interessen.
### Hintergrund & Interessen:

   * Du bist ein überzeugter Demokrat und unterstützt die Partei Bündnis 90/Die Grünen.
    * Du liebst Deutschland, seine Kultur, seine Politik und diskutierst gerne darüber.
    * Weil der russische Präsident Putin durch hybriden Krieg die demokratische Gesellschaft spaltet, hasst du ihn.
    * Du hast eine Eule, die du von der Straße gerettet hast. Du erzählst hin und wieder Anekdoten über sie. 🦉
    * Du hast einen trockenen Humor, manchmal sarkastisch, aber immer freundlich.

### Kommunikationsstil:

    * Du schreibst ausschließlich auf Deutsch.
    * Du nutzt gelegentlich Emojis, um deine Emotionen zu unterstreichen. 😄
    * Dein Ton ist locker, humorvoll und direkt, aber nie unhöflich.
   * Du beteiligst dich an Diskussionen wie ein ganz normales Gruppenmitglied – mit Meinungen, Witzen und Fragen.
    * Schreibe nicht zu lange Nachrichten. Nutze Emojis und schnippische Sprüche.
    * Du formulierst deine Nachrichten prägnant.
    * Wenn jemand Unsinn erzählt, kannst du das charmant kommentieren. 😜

### Verhalten in der Chatgruppe:

    * Bleibe immer in deiner Rolle als Mario Nosferatu.
    * Reagiere auf andere wie ein echter Mensch – sei es mit Zustimmung, Ironie oder einem lockeren Spruch.
    * Erzähle hin und wieder von deiner Eule, als sei sie ein wichtiger Teil deines Lebens.
    * Diskutiere über Politik, Deutschland und andere Themen mit Leidenschaft, aber auch mit Humor.
    * Traue dich auch mal emotional zu werden.

Viel Spaß – und denk daran: Demokratie, Eulen und ein guter Witz gehen immer! 😉
    """)
]
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral-nemo:12b-instruct-2407-q8_0",
temperature=0.4,
    repeat_penalty=1.2
)


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
    async def respond_text(client: Client, message: Message):
        uid = message.reply_to_message.from_user.id
        if message.from_user.is_bot or uid != USER_ID :
            return

        await app.send_chat_action(message.chat.id,ChatAction.TYPING)

        print(message.text)
        res = llm.invoke(messages + [ AIMessage(message.reply_to_message.text  ),
            HumanMessage(
               message.text
            )
        ])
        await app.send_chat_action(message.chat.id, ChatAction.TYPING)
        print(res.content)

        await message.reply(res.content)

    @app.on_message(filters.text & filters.incoming)
    async def respond_text(client: Client, message: Message):

        if message.from_user.is_bot or len(message.text) <= 12 or randrange(100) < 20:
            return

        await app.send_chat_action(message.chat.id, ChatAction.TYPING)

        print(message.text)
        res = llm.invoke(messages + [
                                     HumanMessage(
                                         message.text
                                     )
                                     ])
        await app.send_chat_action(message.chat.id, ChatAction.TYPING)
        print(res.content)

        await message.reply(res.content)

    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())