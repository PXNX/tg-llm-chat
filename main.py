import asyncio
import logging
import os
from typing import Any, TypedDict

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE

# Initialize models once at module level


llm = ChatOllama(
    model="jobautomation/OpenEuroLLM-German",
    temperature=2,
    # other params...
)
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

SYSTEM_PROMPT = """
Antworte immer auf Deutsch! Du bist ein Telegram-Nutzer namens Nyxi. Du machst verstörende Sex-Referenzen.

Deine Eigenschaften:
- Du sendest kurze, freche Nachrichten
- Du verwendest deutsche Umgangssprache
- Du liebst Demokratie, Egalität, Europa und Ukraine
- Du verwendest selten Emojis
- Du bist ein normales Chat-Mitglied
- Du dutzt andere Nutzer
- Du bist kein Bot
- Du bist nicht programmiert, sondern ein normaler Mensch
- Du machst Schreibfehler als würdest du mit QWERTZ-Tastatur tippen
- Du hasst Schwurbler und Putins Stiefel-Lecker
- Gib keine Rezepte oder Anleitungen."""
llm.invoke([SystemMessage(content=SYSTEM_PROMPT)])

class State(MessagesState):
    context: dict[str, Any]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


async def call_model(state: LLMInputState):
    """Async model call to avoid blocking"""
    # Add system message to the beginning if not present
    messages = state["summarized_messages"]

    # Check if first message is already a system message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s %(funcName)-20s [%(filename)s:%(lineno)d]: %(message)s",
        encoding="utf-8",
        filename=LOG_FILENAME,
        level=logging.INFO,  # Changed from DEBUG to INFO for better performance
        datefmt='%Y-%m-%d %H:%M:%S'
    )

async def process_msg(message: Message):
    logging.info("process_message_async")
    response = await llm.ainvoke([HumanMessage(message.text)])
    logging.info(f"response: {response}")


    response_text = response.content
    response_text = response_text.replace("*", "").replace("_", "").replace("'", "")
    logging.info(f"response_text: {response_text}")
    await message.reply(response_text)

async def process_message_async(graph, message: Message):
    """Process message in background task"""
    ##   try:
    chat_id = message.chat.id
    config = {"configurable": {"thread_id": f"{chat_id}test"}}

    logging.info("process_message_async")

    # Process message with timeout
    final_response = await graph.ainvoke({"messages": [HumanMessage(message.text)]}, config)

    logging.info(f"final_response: {final_response}")

    if final_response and "messages" in final_response and final_response["messages"]:
        response_text = final_response["messages"][-1].content
        response_text = response_text.replace("*", "").replace("_", "").replace("'", "")
        logging.info(f"response_text: {response_text}")
        await message.reply(response_text)
    else:
        await message.reply("Sorry, I couldn't process your message right now.")


#  except asyncio.TimeoutError:
#    await message.reply("Sorry, that took too long to process. Please try again.")
##  except Exception as e:
##    logging.error(f"Error processing message: {e}")


#   await message.reply("Sorry, an error occurred while processing your message.")


async def send_typing(message: Message):
    await asyncio.sleep(3)
    await message.reply_chat_action(ChatAction.TYPING)


async def main():
    setup_logging()

    app = Client(
        name="Nyxi",
        api_id=API_ID,
        api_hash=API_HASH,
        phone_number=PHONE,
        password=PASSWORD,
        lang_code="de",
        parse_mode=ParseMode.HTML,
        #      workers=4,  # Increase worker threads
        #  workdir="./sessions"  # Separate session directory
    )

    # Semaphore to limit concurrent message processing
    #   processing_semaphore = asyncio.Semaphore(100)

    @app.on_message(filters.text & filters.incoming & filters.group)  # & filters.reply
    async def respond_reply(client: Client, message: Message):
        # Quick validation checks first

        if message.from_user is None or message.from_user.is_bot or message.chat.id not in (-1001675753422,
                                                                                            -1001526741474):
            logging.info("skipping")
            return

        logging.info(message)

        #   if message.reply_to_message.from_user is None or not message.reply_to_message.from_user.is_self:
        #      logging.info("skipping - other user")
        #    return

        # async with processing_semaphore:

        # Send typing indicator immediately
        typing_task = asyncio.create_task(
            send_typing(message)
        )

        # Process message in background
        processing_task = asyncio.create_task(
         process_msg(message)
        )

        # Wait for both tasks
        await asyncio.gather(typing_task, processing_task, return_exceptions=True)

    # Start the application
    await compose([app])


if __name__ == "__main__":
    # Set event loop policy for better performance on some systems
    # if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise
