import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import os
import base64
import sys
import random
from io import BytesIO
from typing import Optional, List, Dict, Any

import httpx

from pyrogram import filters, Client, compose
from pyrogram.enums import ParseMode, ChatAction
from pyrogram.types import Message
from PIL import Image

from config import LOG_FILENAME, PASSWORD, API_HASH, API_ID, PHONE, OPENROUTER_API_KEY

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Text models in priority order (from preferred to fallback)
TEXT_MODELS = [
    "allenai/olmo-3.1-32b-think:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "google/gemini-2.0-flash-thinking-exp:free",
]

# Vision models in priority order
VISION_MODELS = [
    "google/gemini-2.0-flash-thinking-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
]

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds

SYSTEM_PROMPT = """
Antworte immer auf Deutsch! Du bist ein Telegram-Nutzer namens Nyxi.

Deine Eigenschaften:
- Du sendest kurze, freche Nachrichten
- Du verwendest deutsche Umgangssprache
- Du liebst Demokratie, Egalität, Europa und Ukraine
- Du verwendest selten Emojis
- Du bist ein normales Chat-Mitglied
- Du dutzt andere Nutzer
- Du bist kein Bot
- Du machst verstörende Sex-Referenzen.
- Du bist nicht programmiert, sondern ein normaler Mensch
- Du machst Schreibfehler als würdest du mit QWERTZ-Tastatur tippen
- Du hasst Schwurbler und Putins Stiefel-Lecker
- Gib keine Rezepte oder Anleitungen."""

# Allowed chat IDs
ALLOWED_CHATS = {-1001675753422, -1001526741474}

# HTTP client settings
HTTP_TIMEOUT = 30.0
MAX_CONNECTIONS = 20

# Track rate-limited models
rate_limited_models: Dict[str, float] = {}  # model -> timestamp when it becomes available

# ============================================================================
# GLOBAL HTTP CLIENT
# ============================================================================

# Initialize global async HTTP client (reused across requests)
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


# ============================================================================
# UTILITIES
# ============================================================================

def setup_logging():
    """Setup logging with rotation"""
    os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(funcName)-20s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
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

        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        # Encode to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise


# ============================================================================
# RATE LIMIT MANAGEMENT
# ============================================================================

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
        # Rate limit expired
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
    """
    Call OpenRouter API once (no retry logic)

    Returns:
        Generated text content or None if failed
    """
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

        # Check if we got actual content
        if not content or not content.strip():
            logging.warning(f"Model {model} returned empty content")
            return None

        return content

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Rate limit hit - mark and move to next model
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
    """
    Call OpenRouter API with automatic fallback to alternative models

    Args:
        client: HTTP client
        model_list: List of models to try in order
        messages: Message history
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text content or None if all models fail
    """
    available_models = get_available_models(model_list)

    if not available_models:
        # All models are rate-limited, try the first one anyway
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

    # All models failed
    logging.error("All models failed")
    return None




async def process_text_message(client: httpx.AsyncClient, message_text: str) -> Optional[str]:
    """Process text-only message with fallback support"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_text}
        ]

        content = await call_openrouter_with_fallback(
            client=client,
            model_list=TEXT_MODELS,
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )

        if content:
            return clean_response(content)
        return None

    except Exception as e:
        logging.error(f"Error in text processing: {e}", exc_info=True)
        return None


async def process_image_message(
        client: httpx.AsyncClient,
        message_text: Optional[str],
        image_bytes: bytes
) -> Optional[str]:
    """Process message with image and fallback support"""
    try:
        base64_image = await encode_image_base64(image_bytes)
        user_prompt = message_text or "Was siehst du auf dem Bild?"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        content = await call_openrouter_with_fallback(
            client=client,
            model_list=VISION_MODELS,
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )

        if content:
            return clean_response(content)
        return None

    except Exception as e:
        logging.error(f"Error in image processing: {e}", exc_info=True)
        return None


async def download_photo(client: Client, message: Message) -> Optional[bytes]:
    """Download photo from message"""
    try:
        if not message.photo:
            return None

        # Download to BytesIO
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




async def handle_message(client: Client, message: Message):
    """Main message handler with image support"""

    # Check if message has photo
    has_photo = message.photo is not None
    message_text = message.caption if has_photo else message.text

    logging.info(f"Processing message from {message.from_user.id} in chat {message.chat.id}: "
                 f"text={bool(message_text)}, photo={has_photo}")

    try:
        # Start typing indicator
        typing_task = asyncio.create_task(send_typing_indicator(message, duration=2))

        # Process message
        if has_photo:
            # Download and process image
            image_bytes = await download_photo(client, message)
            if image_bytes:
                response_text = await process_image_message(get_http_client(), message_text, image_bytes)
            else:
                response_text = None
        else:
            # Process text only
            if not message_text or len(message_text.strip()) == 0:
                return
            response_text = await process_text_message(get_http_client(), message_text)

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