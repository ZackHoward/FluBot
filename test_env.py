import os
from dotenv import load_dotenv
from pathlib import Path

print("📂 Current directory:", os.getcwd())

# Try loading with explicit path
load_dotenv(dotenv_path=Path('.') / '.env')

# Try getting the key
key = os.getenv("OPENAI_API_KEY")
print("🔑 Loaded key:", key)

print("HELLO WORLD")

