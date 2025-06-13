"""Configuration for project env vars and secrets"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv()
load_dotenv(find_dotenv(".secrets"), override=True)

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PROPERTIES_DB_URL = os.getenv("PROPERTIES_DB_URL")

OUTPUT_DIRECTORY = Path("outputs")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
