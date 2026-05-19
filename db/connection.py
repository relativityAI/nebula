import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from db.models import Profile, AnalysisRun
from dotenv import load_dotenv

load_dotenv()

async def init_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/nebula?authSource=admin")
    client = AsyncIOMotorClient(mongo_uri)
    await init_beanie(
        database=client.get_default_database(),
        document_models=[Profile, AnalysisRun]
    )
