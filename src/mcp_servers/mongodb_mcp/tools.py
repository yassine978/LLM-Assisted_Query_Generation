"""MongoDB tools - Core functionality without MCP framework dependency.

This module provides MongoDB operations that can be used standalone or wrapped by MCP servers.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global MongoDB client (initialized on first use)
_mongo_client: Optional[MongoClient] = None

# Schema cache with TTL (5 minutes)
_schema_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 300  # 5 minutes in seconds


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
        logger.info("Connecting to MongoDB", uri=mongodb_uri)

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
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise

    return _mongo_client


def ping() -> Dict[str, Any]:
    """
    Test MongoDB connection.

    Returns:
        Dict with connection status
    """
    try:
        client = get_mongo_client()
        # Ping the server
        result = client.admin.command('ping')

        # Get server info
        server_info = client.server_info()

        return {
            "success": True,
            "message": "MongoDB connection successful",
            "server_version": server_info.get("version", "unknown"),
            "ping_response": result
        }
    except ConnectionFailure as e:
        logger.error("MongoDB ping failed", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }
    except PyMongoError as e:
        logger.error("Error pinging MongoDB", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def list_databases() -> List[str]:
    """
    List all databases in MongoDB.

    Returns:
        List[str]: List of database names
    """
    try:
        client = get_mongo_client()
        databases = client.list_database_names()
        logger.info("Listed databases", count=len(databases))
        return databases
    except PyMongoError as e:
        logger.error("Error listing databases", error=str(e))
        raise


def _serialize_value(value: Any) -> Any:
    """
    Serialize MongoDB values for JSON compatibility.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable value
    """
    if isinstance(value, ObjectId):
        return str(value)
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_serialize_value(v) for v in value]
    else:
        return value


def _infer_nested_schema(value: Any, max_depth: int = 5, current_depth: int = 0) -> Dict[str, Any]:
    """
    Recursively infer schema for nested documents.

    Args:
        value: Value to analyze
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        Schema information for the value
    """
    if current_depth >= max_depth:
        return {"type": "object", "max_depth_reached": True}

    if isinstance(value, dict):
        fields = {}
        for key, val in value.items():
            fields[key] = _infer_nested_schema(val, max_depth, current_depth + 1)
        return {"type": "object", "fields": fields}

    elif isinstance(value, list):
        if not value:
            return {"type": "array", "element_type": "unknown", "length": 0}

        # Sample first 10 elements to infer type
        element_types = set()
        sample_elements = value[:10]

        for item in sample_elements:
            if isinstance(item, dict):
                element_types.add("object")
            elif isinstance(item, list):
                element_types.add("array")
            else:
                element_types.add(type(item).__name__)

        # If all elements are objects, try to infer nested schema
        if element_types == {"object"} and sample_elements:
            nested_schema = _infer_nested_schema(sample_elements[0], max_depth, current_depth + 1)
            return {
                "type": "array",
                "element_type": "object",
                "element_schema": nested_schema.get("fields", {}),
                "length": len(value)
            }

        return {
            "type": "array",
            "element_types": sorted(list(element_types)),
            "length": len(value)
        }

    else:
        return {"type": type(value).__name__}


def _infer_array_type(array_values: List[Any]) -> Dict[str, Any]:
    """
    Infer the type of array elements.

    Args:
        array_values: List of array values from different documents

    Returns:
        Type information for array elements
    """
    if not array_values:
        return {"type": "array", "element_type": "unknown"}

    all_element_types = set()
    max_length = 0

    for arr in array_values:
        if isinstance(arr, list):
            max_length = max(max_length, len(arr))
            for element in arr[:10]:  # Sample first 10 elements
                if isinstance(element, dict):
                    all_element_types.add("object")
                elif isinstance(element, list):
                    all_element_types.add("array")
                else:
                    all_element_types.add(type(element).__name__)

    return {
        "type": "array",
        "element_types": sorted(list(all_element_types)),
        "max_length": max_length
    }


def _get_collection_stats(db: Any, collection: str) -> Dict[str, Any]:
    """
    Get collection statistics.

    Args:
        db: MongoDB database object
        collection: Collection name

    Returns:
        Collection statistics
    """
    try:
        coll = db[collection]
        stats = db.command("collStats", collection)

        return {
            "size": stats.get("size", 0),
            "storageSize": stats.get("storageSize", 0),
            "avgObjSize": stats.get("avgObjSize", 0),
            "count": stats.get("count", 0),
            "nindexes": stats.get("nindexes", 0),
            "totalIndexSize": stats.get("totalIndexSize", 0),
            "capped": stats.get("capped", False)
        }
    except Exception as e:
        logger.warning(f"Could not retrieve collection stats: {str(e)}")
        return {
            "size": 0,
            "storageSize": 0,
            "avgObjSize": 0,
            "count": 0,
            "nindexes": 0,
            "totalIndexSize": 0,
            "capped": False
        }


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
        logger.info("Listed collections", database=database, count=len(collections))
        return collections
    except PyMongoError as e:
        logger.error("Error listing collections", database=database, error=str(e))
        raise


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
        Dict containing comprehensive schema information with the following structure:
        {
            "collection": str,
            "database": str,
            "document_count": int,
            "sample_size": int,
            "statistics": {...},
            "indexes": [...],
            "fields": {...},
            "sample_documents": [...]
        }

    Raises:
        ValueError: If database or collection doesn't exist
    """
    try:
        # Check cache
        cache_key = f"{database}_{collection}_schema"
        if use_cache and cache_key in _schema_cache:
            timestamp, cached_schema = _schema_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                logger.info(
                    "Returning cached schema",
                    database=database,
                    collection=collection,
                    age_seconds=int(time.time() - timestamp)
                )
                return cached_schema

        client = get_mongo_client()

        # Validate database and collection
        if database not in client.list_database_names():
            raise ValueError(f"Database '{database}' does not exist")

        db = client[database]
        if collection not in db.list_collection_names():
            raise ValueError(f"Collection '{collection}' does not exist in database '{database}'")

        coll = db[collection]

        # Get collection statistics
        stats = _get_collection_stats(db, collection)

        # Get indexes
        try:
            indexes = list(coll.list_indexes())
        except Exception as e:
            logger.warning(f"Could not retrieve indexes: {str(e)}")
            indexes = []

        # Sample documents for schema inference
        documents = list(coll.find().limit(sample_size))

        if not documents:
            empty_schema = {
                "collection": collection,
                "database": database,
                "document_count": 0,
                "sample_size": 0,
                "statistics": stats,
                "indexes": indexes,
                "fields": {},
                "sample_documents": []
            }
            # Cache empty schema too
            _schema_cache[cache_key] = (time.time(), empty_schema)
            return empty_schema

        # Track field occurrences for frequency analysis
        field_occurrences = {}
        field_values = {}

        # First pass: collect all field information
        for doc in documents:
            for key, value in doc.items():
                if key not in field_occurrences:
                    field_occurrences[key] = 0
                    field_values[key] = []

                field_occurrences[key] += 1
                field_values[key].append(value)

        # Second pass: infer detailed schema for each field
        fields_schema = {}
        for field, values in field_values.items():
            # Basic type inference
            types = set()
            for value in values:
                types.add(type(value).__name__)

            # Get first non-None example
            example = next((v for v in values if v is not None), None)

            # Frequency (percentage of documents with this field)
            frequency = (field_occurrences[field] / len(documents)) * 100

            # Build field schema
            field_info = {
                "types": sorted(list(types)),
                "frequency": round(frequency, 2),
                "example": _serialize_value(example)
            }

            # Deep schema inference for nested documents
            if "dict" in types:
                # Get first dict value for nested schema
                dict_value = next((v for v in values if isinstance(v, dict)), None)
                if dict_value:
                    nested_schema = _infer_nested_schema(dict_value, max_depth=5)
                    field_info["nested_schema"] = nested_schema

            # Array type inference
            if "list" in types:
                array_values = [v for v in values if isinstance(v, list)]
                if array_values:
                    array_info = _infer_array_type(array_values)
                    field_info["array_info"] = array_info

                    # If array contains objects, infer nested schema
                    first_array = array_values[0]
                    if first_array and isinstance(first_array[0], dict):
                        nested_schema = _infer_nested_schema(first_array[0], max_depth=5)
                        field_info["array_element_schema"] = nested_schema

            fields_schema[field] = field_info

        # Get sample documents (up to 10)
        sample_documents = []
        for doc in documents[:10]:
            sample_documents.append(_serialize_value(doc))

        # Build complete schema
        schema = {
            "collection": collection,
            "database": database,
            "document_count": stats["count"],
            "sample_size": len(documents),
            "statistics": {
                "size_bytes": stats["size"],
                "storage_size_bytes": stats["storageSize"],
                "avg_document_size_bytes": stats["avgObjSize"],
                "index_count": stats["nindexes"],
                "total_index_size_bytes": stats["totalIndexSize"],
                "capped": stats["capped"]
            },
            "indexes": [
                {
                    "name": idx.get("name"),
                    "keys": idx.get("key"),
                    "unique": idx.get("unique", False),
                    "sparse": idx.get("sparse", False)
                }
                for idx in indexes
            ],
            "fields": fields_schema,
            "sample_documents": sample_documents
        }

        # Cache the schema
        _schema_cache[cache_key] = (time.time(), schema)

        logger.info(
            "Generated comprehensive schema",
            database=database,
            collection=collection,
            field_count=len(fields_schema),
            sample_count=len(sample_documents)
        )
        return schema

    except PyMongoError as e:
        logger.error(
            "Error getting collection schema",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise


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
            "Executed query",
            database=database,
            collection=collection,
            result_count=len(results)
        )
        return response

    except OperationFailure as e:
        logger.error(
            "Query execution failed",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise
    except PyMongoError as e:
        logger.error(
            "Error executing query",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise


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
                "Query validated successfully",
                database=database,
                collection=collection
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
            "Error validating query",
            database=database,
            collection=collection,
            error=str(e)
        )
        return {
            "valid": False,
            "error": str(e),
            "error_type": "MongoDBError"
        }


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
            "Retrieved indexes",
            database=database,
            collection=collection,
            count=len(indexes)
        )
        return indexes

    except PyMongoError as e:
        logger.error(
            "Error getting indexes",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise


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
            "Executed aggregation",
            database=database,
            collection=collection,
            result_count=len(results)
        )
        return response

    except OperationFailure as e:
        logger.error(
            "Aggregation execution failed",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise
    except PyMongoError as e:
        logger.error(
            "Error executing aggregation",
            database=database,
            collection=collection,
            error=str(e)
        )
        raise


def clear_schema_cache(database: Optional[str] = None, collection: Optional[str] = None):
    """
    Clear schema cache entries.

    Args:
        database: Optional database name to clear cache for
        collection: Optional collection name to clear cache for (requires database)

    If both database and collection are provided, clears only that specific cache entry.
    If only database is provided, clears all cache entries for that database.
    If neither is provided, clears the entire cache.
    """
    global _schema_cache

    if database and collection:
        cache_key = f"{database}_{collection}_schema"
        if cache_key in _schema_cache:
            del _schema_cache[cache_key]
            logger.info(f"Cleared schema cache for {database}.{collection}")
    elif database:
        keys_to_delete = [k for k in _schema_cache.keys() if k.startswith(f"{database}_")]
        for key in keys_to_delete:
            del _schema_cache[key]
        logger.info(f"Cleared schema cache for database {database} ({len(keys_to_delete)} entries)")
    else:
        count = len(_schema_cache)
        _schema_cache.clear()
        logger.info(f"Cleared entire schema cache ({count} entries)")


def cleanup():
    """Clean up MongoDB connection and clear cache."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed")

    # Clear schema cache on cleanup
    clear_schema_cache()
