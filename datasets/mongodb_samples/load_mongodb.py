"""Script to load sample data into MongoDB."""

import json
import sys
from pathlib import Path

from pymongo import MongoClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_sample_data():
    """Load sample data into MongoDB."""
    settings = get_settings()

    # Connect to MongoDB
    logger.info("Connecting to MongoDB", uri=settings.mongodb_uri)
    client = MongoClient(settings.mongodb_uri)

    try:
        # Select database
        db = client[settings.mongodb_database]

        # Load users collection
        users_file = Path(__file__).parent / "mongodb_samples" / "sample_users.json"
        if users_file.exists():
            with open(users_file, "r") as f:
                users_data = json.load(f)

            # Drop existing collection
            db.users.drop()
            logger.info("Dropped existing users collection")

            # Insert sample data
            result = db.users.insert_many(users_data)
            logger.info(
                "Loaded users data",
                count=len(result.inserted_ids),
                collection="users"
            )

            # Create indexes
            db.users.create_index("email", unique=True)
            db.users.create_index("city")
            db.users.create_index("age")
            logger.info("Created indexes on users collection")

        else:
            logger.warning("Sample users file not found", path=str(users_file))

        # Verify data
        count = db.users.count_documents({})
        logger.info("Verification complete", total_users=count)

        # Print sample query
        sample_user = db.users.find_one()
        if sample_user:
            logger.info("Sample document", user=sample_user.get("name"))

        logger.info("Data loading complete!")

    except Exception as e:
        logger.error("Error loading data", error=str(e))
        raise
    finally:
        client.close()


if __name__ == "__main__":
    load_sample_data()
