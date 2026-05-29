import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from src.db.models import Profile, AnalysisRun
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

async def init_db():
    # mongo_url = os.getenv("MONGODB_URL", "mongodb://root:example@localhost:27017/")
    # db_name = os.getenv("MONGODB_DB_NAME", "nebula")
    
    # client = AsyncIOMotorClient(mongo_url)
    # await init_beanie(
    #     database=client[db_name],
    #     document_models=[Profile, AnalysisRun]
    # )


    #####################

    mongodb_url = os.getenv("MONGODB_URL")
    db_name = os.getenv("MONGODB_DB_NAME")
    
    if not mongodb_url:
        logger.error("MONGODB_URL not found in environment variables")
        return

    logger.info(f"Connecting to MongoDB at {mongodb_url}...")
    client = AsyncIOMotorClient(mongodb_url)
    
    # Motor 3.x attribute access returns a MotorDatabase, which Beanie tries to call.
    # We explicitly set append_metadata to something non-callable to skip Beanie's check.
    client.append_metadata = None # type: ignore 
    
    await init_beanie(
        database=client[db_name],
        document_models=[
            Profile, AnalysisRun
        ]
    )
    logger.info("Beanie initialization complete.")
