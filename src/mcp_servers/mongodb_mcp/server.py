"""MongoDB MCP Server implementation using FastMCP."""

import json
import logging
import sys
from typing import Any, Dict, List, Optional

# CRITICAL: Configure logging BEFORE importing FastMCP
# FastMCP must not write to stdout as it interferes with MCP JSON-RPC protocol
logging.basicConfig(
    level=logging.CRITICAL,  # Only show critical errors
    stream=sys.stderr,        # Write to stderr, not stdout
    format='%(levelname)s - %(name)s - %(message)s'
)

# Suppress specific loggers that FastMCP uses
logging.getLogger('fastmcp').setLevel(logging.CRITICAL)
logging.getLogger('mcp').setLevel(logging.CRITICAL)
logging.getLogger('docket').setLevel(logging.CRITICAL)

# Suppress Rich console output by redirecting to stderr
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Patch rich.console to use stderr instead of stdout
try:
    from rich.console import Console
    # Monkey patch the default console to use stderr
    import rich.console
    rich.console._console = Console(file=sys.stderr, force_terminal=True)
except Exception:
    pass  # If patching fails, continue anyway

from fastmcp import FastMCP
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError

from ...utils.config import get_settings
from . import tools

# Note: We use basic logging instead of structlog to avoid any stdout interference
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("mongodb-mcp")

# Global MongoDB client (initialized on first use)
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """
    Get or create MongoDB client connection.

    Returns:
        MongoClient: MongoDB client instance

    Raises:
        ConnectionFailure: If connection to MongoDB fails
    """
    global _mongo_client

    if _mongo_client is None:
        settings = get_settings()
        mongodb_uri = settings.get_mongodb_uri()
        logger.info(f"Connecting to MongoDB: {mongodb_uri}")

        try:
            _mongo_client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
            )
            # Verify connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB connection successful")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    return _mongo_client


@mcp.tool()
def ping() -> Dict[str, Any]:
    """
    Test MongoDB connection.

    Returns:
        Dict with connection status
    """
    return tools.ping()


@mcp.tool()
def list_databases() -> List[str]:
    """
    List all databases in MongoDB.

    Returns:
        List[str]: List of database names
    """
    try:
        client = get_mongo_client()
        databases = client.list_database_names()
        logger.info(f"Listed {len(databases)} databases")
        return databases
    except PyMongoError as e:
        logger.error(f"Error listing databases: {e}")
        raise


@mcp.tool()
def list_collections(database: str) -> List[str]:
    """
    List all collections in a specific database.

    Args:
        database: Name of the database

    Returns:
        List[str]: List of collection names

    Raises:
        ValueError: If database doesn't exist
    """
    try:
        client = get_mongo_client()

        # Check if database exists
        if database not in client.list_database_names():
            raise ValueError(f"Database '{database}' does not exist")

        db = client[database]
        collections = db.list_collection_names()
        logger.info(f"Listed {len(collections)} collections in database '{database}'")
        return collections
    except PyMongoError as e:
        logger.error(f"Error listing collections in '{database}': {e}")
        raise


@mcp.tool()
def get_collection_schema(database: str, collection: str, sample_size: int = 100, use_cache: bool = True) -> Dict[str, Any]:
    """
    Infer comprehensive schema from sample documents in a collection.

    This enhanced version includes:
    - Deep nested document schema inference (up to 5 levels)
    - Array element type detection
    - Collection statistics (size, average document size, etc.)
    - Index information
    - Sample documents (up to 10 examples)
    - Field frequency analysis
    - Caching with 5-minute TTL

    Args:
        database: Name of the database
        collection: Name of the collection
        sample_size: Number of documents to sample for schema inference (default: 100)
        use_cache: Whether to use cached schema if available (default: True)

    Returns:
        Dict containing comprehensive schema information

    Raises:
        ValueError: If database or collection doesn't exist
    """
    try:
        return tools.get_collection_schema(database, collection, sample_size, use_cache)
    except Exception as e:
        logger.error(f"Error getting collection schema for '{database}.{collection}': {e}")
        raise


@mcp.tool()
def execute_query(
    database: str,
    collection: str,
    query: str,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Execute a MongoDB query on a collection.

    Args:
        database: Name of the database
        collection: Name of the collection
        query: MongoDB query as JSON string (e.g., '{"age": {"$gt": 25}}')
        limit: Maximum number of results to return (default: 100)

    Returns:
        Dict containing results and metadata

    Raises:
        ValueError: If database/collection doesn't exist or query is invalid JSON
        OperationFailure: If query execution fails
    """
    try:
        client = get_mongo_client()

        # Validate database and collection
        if database not in client.list_database_names():
            raise ValueError(f"Database '{database}' does not exist")

        db = client[database]
        if collection not in db.list_collection_names():
            raise ValueError(f"Collection '{collection}' does not exist in database '{database}'")

        # Parse query JSON
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON query: {str(e)}")

        # Execute query
        coll = db[collection]
        cursor = coll.find(query_dict).limit(limit)
        results = list(cursor)

        # Convert ObjectId to string for JSON serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])

        response = {
            "database": database,
            "collection": collection,
            "query": query_dict,
            "result_count": len(results),
            "limit": limit,
            "results": results
        }

        logger.info(
            f"Executed query on '{database}.{collection}': {len(results)} results"
        )
        return response

    except OperationFailure as e:
        logger.error(
            f"Query execution failed on '{database}.{collection}': {e}"
        )
        raise
    except PyMongoError as e:
        logger.error(
            f"Error executing query on '{database}.{collection}': {e}"
        )
        raise


@mcp.tool()
def validate_query(database: str, collection: str, query: str) -> Dict[str, Any]:
    """
    Validate a MongoDB query without executing it.

    Args:
        database: Name of the database
        collection: Name of the collection
        query: MongoDB query as JSON string

    Returns:
        Dict containing validation results
    """
    try:
        client = get_mongo_client()

        # Validate database and collection
        if database not in client.list_database_names():
            return {
                "valid": False,
                "error": f"Database '{database}' does not exist",
                "error_type": "DatabaseNotFound"
            }

        db = client[database]
        if collection not in db.list_collection_names():
            return {
                "valid": False,
                "error": f"Collection '{collection}' does not exist in database '{database}'",
                "error_type": "CollectionNotFound"
            }

        # Parse query JSON
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": f"Invalid JSON: {str(e)}",
                "error_type": "InvalidJSON"
            }

        # Try to explain the query (validates syntax without executing)
        try:
            coll = db[collection]
            coll.find(query_dict).limit(1).explain()

            logger.info(
                f"Query validated successfully for '{database}.{collection}'"
            )
            return {
                "valid": True,
                "query": query_dict,
                "message": "Query is valid"
            }
        except OperationFailure as e:
            return {
                "valid": False,
                "error": str(e),
                "error_type": "InvalidQuery"
            }

    except PyMongoError as e:
        logger.error(
            f"Error validating query for '{database}.{collection}': {e}"
        )
        return {
            "valid": False,
            "error": str(e),
            "error_type": "MongoDBError"
        }


@mcp.tool()
def get_indexes(database: str, collection: str) -> List[Dict[str, Any]]:
    """
    Get all indexes for a collection.

    Args:
        database: Name of the database
        collection: Name of the collection

    Returns:
        List of index information dictionaries

    Raises:
        ValueError: If database or collection doesn't exist
    """
    try:
        client = get_mongo_client()

        # Validate database and collection
        if database not in client.list_database_names():
            raise ValueError(f"Database '{database}' does not exist")

        db = client[database]
        if collection not in db.list_collection_names():
            raise ValueError(f"Collection '{collection}' does not exist in database '{database}'")

        coll = db[collection]
        indexes = list(coll.list_indexes())

        logger.info(
            f"Retrieved {len(indexes)} indexes for '{database}.{collection}'"
        )
        return indexes

    except PyMongoError as e:
        logger.error(
            f"Error getting indexes for '{database}.{collection}': {e}"
        )
        raise


@mcp.tool()
def aggregate(
    database: str,
    collection: str,
    pipeline: str
) -> Dict[str, Any]:
    """
    Execute an aggregation pipeline on a collection.

    Args:
        database: Name of the database
        collection: Name of the collection
        pipeline: Aggregation pipeline as JSON string (array of stages)

    Returns:
        Dict containing aggregation results

    Raises:
        ValueError: If database/collection doesn't exist or pipeline is invalid
        OperationFailure: If aggregation execution fails
    """
    try:
        client = get_mongo_client()

        # Validate database and collection
        if database not in client.list_database_names():
            raise ValueError(f"Database '{database}' does not exist")

        db = client[database]
        if collection not in db.list_collection_names():
            raise ValueError(f"Collection '{collection}' does not exist in database '{database}'")

        # Parse pipeline JSON
        try:
            pipeline_list = json.loads(pipeline)
            if not isinstance(pipeline_list, list):
                raise ValueError("Pipeline must be a JSON array of stages")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON pipeline: {str(e)}")

        # Execute aggregation
        coll = db[collection]
        cursor = coll.aggregate(pipeline_list)
        results = list(cursor)

        # Convert ObjectId to string for JSON serialization
        for doc in results:
            if '_id' in doc and hasattr(doc['_id'], '__str__'):
                doc['_id'] = str(doc['_id'])

        response = {
            "database": database,
            "collection": collection,
            "pipeline": pipeline_list,
            "result_count": len(results),
            "results": results
        }

        logger.info(
            f"Executed aggregation on '{database}.{collection}': {len(results)} results"
        )
        return response

    except OperationFailure as e:
        logger.error(
            f"Aggregation execution failed on '{database}.{collection}': {e}"
        )
        raise
    except PyMongoError as e:
        logger.error(
            f"Error executing aggregation on '{database}.{collection}': {e}"
        )
        raise


# MCP Resources (with caching via enhanced tools)
@mcp.resource("mongodb://{database}/{collection}/schema")
def get_schema_resource(database: str, collection: str) -> str:
    """
    Get comprehensive collection schema as a resource.

    Includes:
    - Deep nested document schema
    - Array element types
    - Collection statistics
    - Index information
    - Field frequency analysis
    - Cached with 5-minute TTL

    Args:
        database: Name of the database
        collection: Name of the collection

    Returns:
        JSON string of comprehensive schema information
    """
    schema = tools.get_collection_schema(database, collection, use_cache=True)
    return json.dumps(schema, indent=2)


@mcp.resource("mongodb://{database}/{collection}/samples")
def get_samples_resource(database: str, collection: str) -> str:
    """
    Get sample documents from a collection as a resource.

    Returns up to 10 sample documents from the cached schema.

    Args:
        database: Name of the database
        collection: Name of the collection

    Returns:
        JSON string of sample documents
    """
    # Use cached schema which includes sample documents
    schema = tools.get_collection_schema(database, collection, use_cache=True)
    return json.dumps(schema.get("sample_documents", []), indent=2)


@mcp.resource("mongodb://{database}/{collection}/stats")
def get_stats_resource(database: str, collection: str) -> str:
    """
    Get collection statistics as a resource.

    Includes size, document count, average document size, index information.

    Args:
        database: Name of the database
        collection: Name of the collection

    Returns:
        JSON string of collection statistics
    """
    schema = tools.get_collection_schema(database, collection, use_cache=True)
    stats = {
        "database": schema["database"],
        "collection": schema["collection"],
        "document_count": schema["document_count"],
        "statistics": schema["statistics"],
        "indexes": schema["indexes"]
    }
    return json.dumps(stats, indent=2)


@mcp.resource("mongodb://{database}/collections")
def get_collections_resource(database: str) -> str:
    """
    Get list of all collections in a database as a resource.

    Args:
        database: Name of the database

    Returns:
        JSON string of collection names
    """
    collections = tools.list_collections(database)
    return json.dumps(collections, indent=2)


@mcp.tool()
def clear_schema_cache(database: Optional[str] = None, collection: Optional[str] = None) -> Dict[str, str]:
    """
    Clear schema cache entries.

    Args:
        database: Optional database name to clear cache for
        collection: Optional collection name to clear cache for (requires database)

    Returns:
        Dict with status message

    If both database and collection are provided, clears only that specific cache entry.
    If only database is provided, clears all cache entries for that database.
    If neither is provided, clears the entire cache.
    """
    try:
        tools.clear_schema_cache(database, collection)

        if database and collection:
            message = f"Cleared schema cache for {database}.{collection}"
        elif database:
            message = f"Cleared schema cache for all collections in database {database}"
        else:
            message = "Cleared entire schema cache"

        logger.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"Error clearing schema cache: {e}")
        return {"status": "error", "message": str(e)}


def cleanup():
    """Clean up MongoDB connection."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Run the MCP server
    try:
        # Disable banner to keep stdout clean for MCP JSON-RPC protocol
        mcp.run(show_banner=False)
    finally:
        cleanup()
