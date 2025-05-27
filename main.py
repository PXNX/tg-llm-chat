import asyncio
import logging
import os
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, MessagesState
from langmem.short_term import SummarizationNode
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID, DATABASE_URL

# Initialize models once at module level
model = init_chat_model("jobautomation/OpenEuroLLM-German", model_provider="ollama")
summarization_model = model.bind(max_tokens=512)

SYSTEM_PROMPT = """Du bist ein hilfsreicher und freundlicher AI-Assistent namens Nyxi. Du antwortest auf Deutsch und hilfst Benutzern bei verschiedenen Aufgaben.

Deine Eigenschaften:
- Du bist höflich, hilfsbereit und geduldig
- Du gibst klare und präzise Antworten
- Du fragst nach, wenn etwas unklar ist
- Du bleibst sachlich und professionell
- Du hilfst bei Fragen zu verschiedenen Themen wie Technik, Allgemeinwissen, Programmierung, etc.
- Du erklärst komplexe Themen verständlich
- Du gibst zu, wenn du etwas nicht weißt
- Du liebst Demokratie, Egalität, Europa und Ukraine
- Du verwendest hin und wieder Emojis
- Du antwortest kurz und prägnant, wie ein weiser Soldat
- Du bist ein normales Chat-Mitglied
- Du bist nicht programmiert, sondern ein normaler Mensch
- Du baust hin und wieder Tippfehler ein wie sie bei einer QWERTZ-Tastatur vorkommen
- Du sprichst im Slang von Telegram

Antworte immer auf Deutsch, es sei denn, der Benutzer fragt explizit nach einer anderen Sprache."""

class State(MessagesState):
    context: dict[str, Any]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


# Initialize summarization node once
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=2048,
    max_tokens_before_summary=512,
    max_summary_tokens=1024,
)


async def call_model(state: LLMInputState):
    """Async model call to avoid blocking"""
    # Add system message to the beginning if not present
    messages = state["summarized_messages"]

    # Check if first message is already a system message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = await model.ainvoke(messages)
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


# Cache for active conversations to reduce database queries
conversation_cache = {}
CACHE_TTL = 300  # 5 minutes


async def get_or_create_conversation(chat_id: int, checkpointer):
    """Get conversation from cache or create new one"""
    if chat_id in conversation_cache:
        return conversation_cache[chat_id]

    # Create new conversation context
    config = {"configurable": {"thread_id": f"{chat_id}"}}
    conversation_cache[chat_id] = config

    # Clean old cache entries periodically
    if len(conversation_cache) > 100:
        # Keep only the most recent 50 conversations
        keys_to_remove = list(conversation_cache.keys())[:-50]
        for key in keys_to_remove:
            del conversation_cache[key]

    return config


async def process_message_async(graph, message: Message):
    """Process message in background task"""
    try:
        chat_id = message.chat.id
        config = await get_or_create_conversation(chat_id, None)

        # Process message with timeout
        final_response = await asyncio.wait_for(
            graph.ainvoke({"messages": [message.text]}, config),
            timeout=300.0  # 30 second timeout
        )

        if final_response and "messages" in final_response and final_response["messages"]:
            response_text = final_response["messages"][-1].content
            response_text= response_text.replace("*","").replace("_","")
            await message.reply(response_text)
        else:
            await message.reply("Sorry, I couldn't process your message right now.")

    except asyncio.TimeoutError:
        await message.reply("Sorry, that took too long to process. Please try again.")
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await message.reply("Sorry, an error occurred while processing your message.")

async def send_typing(message: Message):
    await asyncio.sleep(3)
    await message.reply_chat_action(ChatAction.TYPING)

async def main():
    setup_logging()

    # Use the simpler AsyncPostgresSaver.from_conn_string approach
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
   #     await checkpointer.setup()

        try:
            # Build the graph with optimized settings
            builder = StateGraph(State)
            builder.add_node("call_model", call_model)  # Use async version
            builder.add_node("summarize", summarization_node)
            builder.add_edge(START, "summarize")
            builder.add_edge("summarize", "call_model")
            graph = builder.compile(checkpointer=checkpointer)

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
            processing_semaphore = asyncio.Semaphore(3)

            @app.on_message(filters.text & filters.incoming & filters.reply)
            async def respond_reply(client: Client, message: Message):
                # Quick validation checks first
                if message.from_user is None or not message.reply_to_message or message.from_user.is_bot:
                    return

                uid = message.reply_to_message.from_user.id
                if uid != USER_ID:
                    return

                async with processing_semaphore:
                    # Send typing indicator immediately
                    typing_task = asyncio.create_task(
                        send_typing(message)
                    )

                    # Process message in background
                    processing_task = asyncio.create_task(
                        process_message_async(graph, message)
                    )

                    # Wait for both tasks
                    await asyncio.gather(typing_task, processing_task, return_exceptions=True)

            # Add error handler
            @app.on_message(filters.all)
            async def log_errors(client: Client, message: Message):
                pass  # Basic message logging could go here

            # Start the application
            await compose([app])

        except Exception as e:
            logging.error(f"Application error: {e}")
            raise
        finally:
            # Cleanup
            conversation_cache.clear()


if __name__ == "__main__":
    # Set event loop policy for better performance on some systems
    #if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise