import asyncio
import logging
import os
from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langmem.short_term import SummarizationNode
from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, USER_ID

model = init_chat_model("jobautomation/OpenEuroLLM-German",model_provider="ollama")
summarization_model = model.bind(max_tokens=128)

class State(MessagesState):
    context: dict[str, Any]



class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

def call_model(state: LLMInputState):


    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)





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
    #setup_logging()

    app = Client(
        name="Nyxi",
        api_id=API_ID,
        api_hash=API_HASH,
        phone_number=PHONE,
        password=PASSWORD,
        lang_code="de",
        parse_mode=ParseMode.HTML
    )

    @app.on_message(filters.text & filters.incoming & filters.reply)
    async def respond_reply(client: Client, message: Message):

        print("----- respond_reply ---1")

        uid = message.reply_to_message.from_user.id
        if message.from_user.is_bot or uid != USER_ID:
            return

        print("----- respond_reply")

        chat_id = message.chat.id

        config = {"configurable": {"thread_id": chat_id}}

        await message.reply_chat_action(ChatAction.TYPING)

        final_response = graph.invoke({"messages": message.text}, config)

        print("----- final_response", final_response)

        res = final_response["messages"][-1].content
        print("----- ai msg", res)

        await message.reply(res)







    await compose([app])


if __name__ == "__main__":
    asyncio.run(main())
