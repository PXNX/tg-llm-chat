import os
from datetime import datetime
from typing import Final

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PASSWORD = os.getenv("PASSWORD")
API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")
USER_ID:Final[int] = int(os.getenv("USER_ID"))

LOG_FILENAME = rf"./logs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}.log"
