import os
from datetime import datetime
from typing import Final

from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
PASSWORD = os.getenv("PASSWORD")
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")
USER_ID: Final[int] = int(os.getenv("USER_ID"))

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Database Configuration (if needed later)
DATABASE_URL = os.getenv("DATABASE_URL")

# Logging Configuration
LOG_FILENAME = rf"./logs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}.log"