# database.py
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["FinsightDB"]

async def init_db():
    await db.sessions.create_index("session_id", unique=True)
    await db.messages.create_index([("session_id", 1), ("timestamp", 1)])
    await db.files.create_index("session_id")
    await db.file_binaries.create_index([("session_id", 1), ("filename", 1)])
    print(" Database indexes created")

async def close_db():
    client.close()