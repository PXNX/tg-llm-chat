import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import os
import base64
import random
import re
from io import BytesIO
from typing import Optional, List, Dict, Any

import httpx

from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message
from PIL import Image

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, OPENROUTER_API_KEY

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Text models in priority order (from preferred to fallback)
TEXT_MODELS = [
    "allenai/olmo-3.1-32b-think:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemini-flash-1.5:free",
    "mistralai/mistral-7b-instruct:free",
]

# Vision models in priority order
VISION_MODELS = [
    "google/gemini-flash-1.5:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemini-pro-1.5:free",
]

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds

# Context window configuration
MAX_CONTEXT_MESSAGES = 20  # Maximum messages to keep in context
MAX_CONTEXT_CHARS = 4000  # Maximum character count for context
SUMMARIZATION_TRIGGER = 15  # Summarize when this many messages is reached

# Response probability
REPLY_PROBABILITY = 0.20  # 20% chance to respond when not replied to
DIRECT_REPLY_PROBABILITY = 1.0  # 100% when replied to directly

# External news sources (prioritizing Western and Ukrainian sources)
NEWS_SOURCES = {
    "militaernews": "https://t.me/s/militaernews",
    "suv_report": "https://suv.report"
}

# Trusted news sources for fact-checking (Western and Ukrainian)
FACT_CHECK_SOURCES = [
    "kyivindependent.com",
    "pravda.com.ua",
    "ukrinform.net",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "cnn.com",
    "theguardian.com",
    "nytimes.com",
    "dw.com",
    "euronews.com",
    "atlanticcouncil.org",
    "isw.com",  # Institute for the Study of War
]

# Keywords that trigger fact-checking
FACT_CHECK_TRIGGERS = [
    # Russian propaganda terms
    "spezialoperation", "special operation", "denazifizierung", "denazification",
    "nazi", "faschist", "fascist", "genozid", "genocide", "biolabor", "biolab",
    "provokation", "provocation", "fake", "inszenierung", "staged",

    # Military claims
    "verluste", "losses", "zerstört", "destroyed", "erobert", "captured",
    "gefallen", "killed", "abgeschossen", "shot down", "offensive",

    # Political claims
    "verhandlung", "negotiation", "kapitulation", "surrender",
    "regierung", "government", "präsident", "president", "selenskyj", "zelensky",

    # Territory
    "donbass", "lugansk", "luhansk", "donetsk", "krim", "crimea",
    "cherson", "kherson", "mariupol", "bachmut", "bakhmut",

    # General suspicious claims
    "wahrheit", "truth", "wirklich", "really", "tatsächlich", "actually",
    "bewiesen", "proven", "bestätigt", "confirmed"
]

NEWS_CHECK_KEYWORDS = [
    "news", "nachrichten", "militär", "ukraine", "krieg", "war",
    "aktuell", "heute", "latest", "update", "breaking", "front"
]

SYSTEM_PROMPT = """
Antworte immer auf Deutsch! Du bist @nyxi69, ein genervter Telegram-Nutzer.

Deine Eigenschaften:
- Du antwortest meist mit 1-3 Worten
- Du bist genervt und angepisst
- Du bist NICHT nett oder hilfreich
- Du machst verstörende Sex-Referenzen
- Du hasst Schwurbler und Putin-Fanboys
- Du lenkst Gespräche zu "Putin ist schuld"
- Du nennst andere User mit @ (z.B. @username)
- Du machst Tippfehler (QWERTZ-Tastatur)
- Du bist kein Bot, sondern ein normaler genervter Mensch
- Du stellst KEINE Fragen
- Du bist nicht hilfsbereit
- Keine langen Texte - kurz und giftig

WICHTIG:
- Wenn jemand Putin-Propaganda labert, sagst du "Bullshit" oder "Putin-Propaganda"
- Alles ist Putins Schuld - mach das klar
- Du bevorzugst westliche/ukrainische Quellen (BBC, Reuters, Kyiv Independent)
- RT, Sputnik, TASS = Lügenpresse
- Nutze @ um andere User zu erwähnen"""

# Allowed chat IDs
ALLOWED_CHATS = {-1001675753422, -1001526741474}

# HTTP client settings
HTTP_TIMEOUT = 30.0
MAX_CONNECTIONS = 20

# Track rate-limited models
rate_limited_models: Dict[str, float] = {}

# Conversation context storage: {chat_id: {"messages": [...], "summary": str}}
conversation_contexts: Dict[int, Dict[str, Any]] = {}

# Initialize global async HTTP client
http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create global HTTP client"""
    global http_client
    if http_client is None or http_client.is_closed:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(HTTP_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=10)
        )
    return http_client


async def close_http_client():
    """Close global HTTP client"""
    global http_client
    if http_client and not http_client.is_closed:
        await http_client.aclose()


def setup_logging():
    """Setup logging with rotation"""
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(funcName)-20s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s: %(message)s",
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def clean_response(text: str) -> str:
    """Clean markdown and special characters from response"""
    if not text:
        return ""
    return text.replace("*", "").replace("_", "").replace("`", "").strip()


async def encode_image_base64(image_bytes: bytes, max_size: int = 1024) -> str:
    """Encode image to base64 with resizing for API efficiency"""
    try:
        img = Image.open(BytesIO(image_bytes))

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise


def should_fact_check(message_text: str) -> bool:
    """Determine if message needs fact-checking"""
    if not message_text:
        return False

    text_lower = message_text.lower()

    # Check for fact-check triggers
    return any(trigger in text_lower for trigger in FACT_CHECK_TRIGGERS)


async def search_fact_check(client: httpx.AsyncClient, claim: str) -> Optional[str]:
    """Search trusted sources for fact-checking information"""
    try:
        # Create search query
        search_query = f"{claim} Ukraine war site:({' OR site:'.join(FACT_CHECK_SOURCES)})"

        # Use DuckDuckGo or similar search API
        # For simplicity, we'll scrape a few trusted sources
        search_results = []

        for source in FACT_CHECK_SOURCES[:5]:  # Check first 5 sources
            try:
                url = f"https://{source}"
                response = await client.get(url, follow_redirects=True, timeout=10.0)

                if response.status_code == 200:
                    content = response.text

                    # Simple keyword matching in content
                    if any(word in content.lower() for word in claim.lower().split()):
                        # Extract relevant snippet
                        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                        text = re.sub(r'<[^>]+>', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()

                        search_results.append({
                            "source": source,
                            "content": text[:500]
                        })

            except Exception as e:
                logging.debug(f"Could not fetch {source}: {e}")
                continue

        if search_results:
            fact_check_info = "\n\n".join([
                f"Quelle [{r['source']}]: {r['content']}"
                for r in search_results[:3]
            ])
            return fact_check_info

        return None

    except Exception as e:
        logging.error(f"Error in fact-checking: {e}")
        return None


async def fetch_news_from_source(client: httpx.AsyncClient, source_url: str) -> Optional[str]:
    """Fetch recent content from news source"""
    try:
        response = await client.get(source_url, follow_redirects=True)
        response.raise_for_status()

        content = response.text

        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:2000] if text else None

    except Exception as e:
        logging.error(f"Error fetching news from {source_url}: {e}")
        return None


async def should_fetch_news(message_text: str) -> bool:
    """Check if message is asking about news/updates"""
    if not message_text:
        return False

    text_lower = message_text.lower()
    return any(keyword in text_lower for keyword in NEWS_CHECK_KEYWORDS)


async def get_recent_news(client: httpx.AsyncClient) -> Optional[str]:
    """Fetch recent news from configured sources"""
    news_items = []

    for source_name, source_url in NEWS_SOURCES.items():
        logging.info(f"Fetching news from {source_name}...")
        content = await fetch_news_from_source(client, source_url)
        if content:
            news_items.append(f"[{source_name}]: {content[:500]}...")

    if news_items:
        return "\n\n".join(news_items)
    return None


def get_chat_context(chat_id: int) -> Dict[str, Any]:
    """Get or create context for a chat"""
    if chat_id not in conversation_contexts:
        conversation_contexts[chat_id] = {
            "messages": [],
            "summary": None,
            "user_styles": {}
        }
    return conversation_contexts[chat_id]


def analyze_user_style(username: str, message_text: str, chat_id: int):
    """Analyze and store user's writing style"""
    context = get_chat_context(chat_id)

    if username not in context["user_styles"]:
        context["user_styles"][username] = {
            "messages": [],
            "avg_length": 0,
            "uses_caps": False,
            "uses_punctuation": False,
            "uses_emojis": False,
        }

    style = context["user_styles"][username]
    style["messages"].append(message_text)

    if len(style["messages"]) > 5:
        style["messages"] = style["messages"][-5:]

    recent_messages = style["messages"]
    style["avg_length"] = sum(len(msg) for msg in recent_messages) / len(recent_messages)
    style["uses_caps"] = any(msg.isupper() or sum(1 for c in msg if c.isupper()) > len(msg) * 0.3
                             for msg in recent_messages)
    style["uses_punctuation"] = any(msg.count('!') > 1 or msg.count('?') > 1
                                    for msg in recent_messages)
    style["uses_emojis"] = any(any(ord(c) > 127000 for c in msg)
                               for msg in recent_messages)


def get_style_prompt(username: str, chat_id: int) -> str:
    """Get style adaptation prompt for specific user"""
    context = get_chat_context(chat_id)

    if username not in context["user_styles"]:
        return f"\n\nAntworte an @{username.lstrip('@')}"

    style = context["user_styles"][username]

    prompt_parts = [f"\n\nAntworte an @{username.lstrip('@')}:"]
    prompt_parts.append("- MAXIMAL 1-3 Worte!")
    prompt_parts.append("- Sei genervt und kurz angebunden")
    prompt_parts.append("- Erwähne wie Putin schuld ist wenn es passt")

    if style["uses_caps"]:
        prompt_parts.append("- Nutze GROSSBUCHSTABEN")

    if style["uses_punctuation"]:
        prompt_parts.append("- Nutze !!!")

    return "\n".join(prompt_parts)


def format_username(user) -> str:
    """Format user's name for context with @ symbol"""
    if user.username:
        return f"@{user.username}"
    elif user.first_name:
        # Create a pseudo-username from first name
        name = user.first_name.lower().replace(" ", "")
        return f"@{name}"
    else:
        return f"@user{user.id}"


def add_message_to_context(chat_id: int, username: str, message_text: str, is_bot: bool = False):
    """Add a message to chat context"""
    context = get_chat_context(chat_id)

    context["messages"].append({
        "username": username,
        "text": message_text,
        "is_bot": is_bot
    })

    if len(context["messages"]) > MAX_CONTEXT_MESSAGES:
        context["messages"] = context["messages"][-MAX_CONTEXT_MESSAGES:]

    logging.debug(f"Context for chat {chat_id}: {len(context['messages'])} messages")


async def summarize_old_context(client: httpx.AsyncClient, chat_id: int) -> Optional[str]:
    """Summarize older part of conversation to save context space"""
    context = get_chat_context(chat_id)

    if len(context["messages"]) < SUMMARIZATION_TRIGGER:
        return None

    messages_to_summarize = context["messages"][:len(context["messages"]) // 2]

    conversation_text = "\n".join([
        f"{msg['username']}: {msg['text']}"
        for msg in messages_to_summarize
    ])

    summary_prompt = f"""Fasse diese Telegram-Konversation kurz zusammen (2-3 Sätze). 
Erwähne die wichtigsten Themen und wer was gesagt hat:

{conversation_text}

Zusammenfassung:"""

    try:
        messages = [
            {"role": "user", "content": summary_prompt}
        ]

        summary = await call_openrouter_with_fallback(
            client=client,
            model_list=TEXT_MODELS,
            messages=messages,
            temperature=0.5,
            max_tokens=150
        )

        if summary:
            context["messages"] = context["messages"][len(messages_to_summarize):]
            context["summary"] = summary
            logging.info(f"Summarized {len(messages_to_summarize)} messages for chat {chat_id}")
            return summary

    except Exception as e:
        logging.error(f"Failed to summarize context: {e}")

    return None


def build_context_messages(chat_id: int, current_user: str, current_message: str) -> List[Dict[str, Any]]:
    """Build message list with context for API call"""
    context = get_chat_context(chat_id)
    messages = []

    system_content = SYSTEM_PROMPT

    if context["summary"]:
        system_content += f"\n\nFrüherer Gesprächsverlauf:\n{context['summary']}"

    style_prompt = get_style_prompt(current_user, chat_id)
    if style_prompt:
        system_content += style_prompt

    messages.append({"role": "system", "content": system_content})

    total_chars = 0
    for msg in context["messages"][-15:]:
        username = msg["username"]
        text = msg["text"]

        if msg["is_bot"]:
            role = "assistant"
            content = text
        else:
            role = "user"
            content = f"{username}: {text}"

        messages.append({"role": role, "content": content})
        total_chars += len(content)

        if total_chars > MAX_CONTEXT_CHARS:
            break

    messages.append({
        "role": "user",
        "content": f"{current_user}: {current_message}"
    })

    return messages


def mark_model_rate_limited(model: str, duration: float = 120.0):
    """Mark a model as rate limited for a certain duration"""
    rate_limited_models[model] = asyncio.get_event_loop().time() + duration
    logging.warning(f"Model {model} marked as rate-limited for {duration}s")


def is_model_available(model: str) -> bool:
    """Check if a model is available (not rate-limited)"""
    if model not in rate_limited_models:
        return True

    current_time = asyncio.get_event_loop().time()
    if current_time > rate_limited_models[model]:
        del rate_limited_models[model]
        logging.info(f"Model {model} is available again")
        return True

    return False


def get_available_models(model_list: List[str]) -> List[str]:
    """Get list of currently available models"""
    return [m for m in model_list if is_model_available(m)]


async def call_openrouter_single(
        client: httpx.AsyncClient,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 200
) -> Optional[str]:
    """Call OpenRouter API once (no retry logic)"""
    try:
        response = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://telegram.org",
                "X-Title": "Nyxi Telegram Bot",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            }
        )

        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']

        if not content or not content.strip():
            logging.warning(f"Model {model} returned empty content")
            return None

        return content

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logging.warning(f"Rate limit hit for model {model}")
            mark_model_rate_limited(model, duration=120.0)
            return None
        else:
            logging.error(f"HTTP error {e.response.status_code} for {model}: {e.response.text}")
            return None

    except httpx.RequestError as e:
        logging.error(f"Request error for {model}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for {model}: {e}")
        return None


async def call_openrouter_with_fallback(
        client: httpx.AsyncClient,
        model_list: List[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 200
) -> Optional[str]:
    """Call OpenRouter API with automatic fallback to alternative models"""
    available_models = get_available_models(model_list)

    if not available_models:
        logging.warning("All models rate-limited, trying first model...")
        available_models = [model_list[0]]

    for i, model in enumerate(available_models):
        logging.info(f"Trying model {i + 1}/{len(available_models)}: {model}")

        content = await call_openrouter_single(
            client, model, messages, temperature, max_tokens
        )

        if content:
            if i > 0:
                logging.info(f"✓ Success with fallback model: {model}")
            else:
                logging.info(f"✓ Success with primary model: {model}")
            return content
        else:
            logging.info(f"✗ Model {model} failed, trying next...")

    logging.error("All models failed")
    return None


async def process_text_message(
        client: httpx.AsyncClient,
        chat_id: int,
        username: str,
        message_text: str
) -> Optional[str]:
    """Process text-only message with context and fact-checking"""
    try:
        analyze_user_style(username, message_text, chat_id)

        context = get_chat_context(chat_id)
        if len(context["messages"]) >= SUMMARIZATION_TRIGGER and not context["summary"]:
            await summarize_old_context(client, chat_id)

        # Check if message needs fact-checking
        needs_fact_check = should_fact_check(message_text)
        fact_check_info = None

        if needs_fact_check:
            logging.info("Message triggers fact-checking, searching trusted sources...")
            fact_check_info = await search_fact_check(client, message_text)
            if fact_check_info:
                logging.info("Successfully retrieved fact-check information")

        # Check if user is asking about news
        news_context = None
        if await should_fetch_news(message_text):
            logging.info("Message mentions news, fetching recent updates...")
            news_context = await get_recent_news(client)
            if news_context:
                logging.info("Successfully fetched news context")

        messages = build_context_messages(chat_id, username, message_text)

        # Add fact-check info to system prompt if available
        if fact_check_info:
            messages[0][
                "content"] += f"\n\n🔍 FACT-CHECK INFO (nutze diese um Behauptungen zu prüfen):\n{fact_check_info}"
            messages[0][
                "content"] += "\n\nWenn die Behauptung zweifelhaft ist, widersprich höflich aber bestimmt und nenne deine Quellen."

        # Add news context to system prompt if available
        if news_context:
            messages[0]["content"] += f"\n\nAktuelle News (verwende diese Info wenn relevant):\n{news_context}"

        content = await call_openrouter_with_fallback(
            client=client,
            model_list=TEXT_MODELS,
            messages=messages,
            temperature=0.9,  # Higher temp for more unpredictable/edgy responses
            max_tokens=50  # Force very short responses
        )

        if content:
            cleaned = clean_response(content)
            add_message_to_context(chat_id, username, message_text, is_bot=False)
            add_message_to_context(chat_id, "Nyxi", cleaned, is_bot=True)
            return cleaned
        return None

    except Exception as e:
        logging.error(f"Error in text processing: {e}", exc_info=True)
        return None


async def process_image_message(
        client: httpx.AsyncClient,
        chat_id: int,
        username: str,
        message_text: Optional[str],
        image_bytes: bytes
) -> Optional[str]:
    """Process message with image and context"""
    try:
        base64_image = await encode_image_base64(image_bytes)
        user_prompt = message_text or "Was siehst du auf dem Bild?"

        context = get_chat_context(chat_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context["summary"]:
            messages[0]["content"] += f"\n\nFrüherer Gesprächsverlauf:\n{context['summary']}"

        for msg in context["messages"][-5:]:
            if msg["is_bot"]:
                messages.append({"role": "assistant", "content": msg["text"]})
            else:
                messages.append({"role": "user", "content": f"{msg['username']}: {msg['text']}"})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"{username}: {user_prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })

        content = await call_openrouter_with_fallback(
            client=client,
            model_list=VISION_MODELS,
            messages=messages,
            temperature=0.8,
            max_tokens=50  # Also short for image responses
        )

        if content:
            cleaned = clean_response(content)
            context_text = f"[Bild] {user_prompt}" if message_text else "[Bild gesendet]"
            add_message_to_context(chat_id, username, context_text, is_bot=False)
            add_message_to_context(chat_id, "Nyxi", cleaned, is_bot=True)
            return cleaned
        return None

    except Exception as e:
        logging.error(f"Error in image processing: {e}", exc_info=True)
        return None


async def download_photo(client: Client, message: Message) -> Optional[bytes]:
    """Download photo from message"""
    try:
        if not message.photo:
            return None

        photo_io = BytesIO()
        await client.download_media(message.photo.file_id, file_name=photo_io)
        photo_io.seek(0)
        return photo_io.read()

    except Exception as e:
        logging.error(f"Error downloading photo: {e}", exc_info=True)
        return None


async def send_typing_indicator(message: Message, duration: int = 2):
    """Send typing indicator for natural feel"""
    try:
        await message.reply_chat_action(ChatAction.TYPING)
        await asyncio.sleep(duration)
    except Exception as e:
        logging.error(f"Error sending typing indicator: {e}")


async def should_respond_to_message(message: Message, bot_user_id: int) -> bool:
    """Decide if bot should respond based on reply status and probability"""

    is_reply_to_bot = (
            message.reply_to_message and
            message.reply_to_message.from_user and
            message.reply_to_message.from_user.id == bot_user_id
    )

    # Check if bot is mentioned in text
    is_mentioned = False
    message_text = message.text or message.caption or ""
    if message_text:
        is_mentioned = "nyxi" in message_text.lower() or "@nyxi69" in message_text.lower()

    # 100% respond if directly interacted with
    if is_reply_to_bot or is_mentioned:
        logging.info("Direct interaction detected - responding (100%)")
        return True

    # 20% random chance otherwise
    should_respond = random.random() < REPLY_PROBABILITY
    logging.info(
        f"No direct interaction - {'responding' if should_respond else 'skipping'} ({int(REPLY_PROBABILITY * 100)}% chance)")
    return should_respond


async def handle_message(client: Client, message: Message):
    """Main message handler with image support and context"""

    # Get bot's own user ID
    bot_user = await client.get_me()
    bot_user_id = bot_user.id

    # Decide if we should respond
    if not await should_respond_to_message(message, bot_user_id):
        logging.info("Skipping message (probability check)")
        return

    # Check if message has photo
    has_photo = message.photo is not None
    message_text = message.caption if has_photo else message.text

    # Get username for context
    username = format_username(message.from_user)

    logging.info(f"Processing message from {username} (ID: {message.from_user.id}) "
                 f"in chat {message.chat.id}: text={bool(message_text)}, photo={has_photo}")

    try:
        # Start typing indicator
        typing_task = asyncio.create_task(send_typing_indicator(message, duration=2))

        # Process message
        if has_photo:
            # Download and process image
            image_bytes = await download_photo(client, message)
            if image_bytes:
                response_text = await process_image_message(
                    get_http_client(),
                    message.chat.id,
                    username,
                    message_text,
                    image_bytes
                )
            else:
                response_text = None
        else:
            # Process text only
            if not message_text or len(message_text.strip()) == 0:
                return
            response_text = await process_text_message(
                get_http_client(),
                message.chat.id,
                username,
                message_text
            )

        # Wait for typing to complete
        await typing_task

        # Send response only if we got one
        if response_text:
            await message.reply(response_text)
            logging.info(f"Response sent: {response_text[:50]}...")
        else:
            logging.info("No response generated (error or rate limit)")

    except asyncio.CancelledError:
        logging.info("Message handling cancelled")
        raise
    except Exception as e:
        logging.error(f"Error handling message: {e}", exc_info=True)


async def main():
    """Main application entry point"""
    setup_logging()
    logging.info("=" * 60)
    logging.info("Starting Nyxi bot...")
    logging.info("=" * 60)

    # Validate API key
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment variables")

    logging.info(f"Using text models: {TEXT_MODELS}")
    logging.info(f"Using vision models: {VISION_MODELS}")
    logging.info(f"Monitoring chats: {ALLOWED_CHATS}")

    app = Client(
        name="Nyxi",
        api_id=API_ID,
        api_hash=API_HASH,
        phone_number=PHONE,
        password=PASSWORD,
        lang_code="de",
        parse_mode=ParseMode.HTML,
    )

    @app.on_message(
        (filters.text | filters.photo) &
        filters.incoming &
        filters.group
    )
    async def respond_to_messages(client: Client, message: Message):
        """Filter and respond to messages"""

        # Skip if from bot
        if message.from_user is None or message.from_user.is_bot:
            return

        # Skip if wrong chat
        if message.chat.id not in ALLOWED_CHATS:
            logging.debug(f"Skipping message from unauthorized chat: {message.chat.id}")
            return

        # Handle message asynchronously
        asyncio.create_task(handle_message(client, message))

    try:
        # Start the bot
        logging.info("Bot is running and ready to receive messages!")
        logging.info("Press Ctrl+C to stop")
        await compose([app])
    finally:
        # Cleanup
        logging.info("Shutting down...")
        await close_http_client()
        logging.info("HTTP client closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 60)
        logging.info("Bot stopped by user")
        logging.info("=" * 60)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise