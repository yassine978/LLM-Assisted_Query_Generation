"""Redis Tools - Core Redis operations and schema inference.

This module provides tools for interacting with Redis, including:
- Connection management with URI support
- Key-value operations (GET, SET, DEL, etc.)
- Data structure operations (Hash, List, Set, Sorted Set)
- Schema inference and pattern analysis
- Key scanning and pattern matching
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import redis
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global connection pool
_redis_client: Optional[redis.Redis] = None

# Schema cache with TTL
_schema_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 300  # 5 minutes


def get_redis_client() -> redis.Redis:
    """Get or create Redis client connection.

    Returns:
        redis.Redis: Connected Redis client

    Raises:
        redis.ConnectionError: If connection fails
    """
    global _redis_client

    if _redis_client is None:
        settings = get_settings()
        redis_uri = settings.get_redis_uri()

        logger.info("Connecting to Redis", uri=redis_uri.replace(settings.redis_password or '', '***'))

        try:
            _redis_client = redis.from_url(
                redis_uri,
                decode_responses=True,
                socket_connect_timeout=5
            )

            # Test connection
            _redis_client.ping()
            logger.info("Redis connection successful")

        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            _redis_client = None
            raise

    return _redis_client


def close_redis_connection():
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def ping() -> Dict[str, Any]:
    """Test Redis connection.

    Returns:
        Dict with connection status
    """
    try:
        client = get_redis_client()
        client.ping()
        return {
            "success": True,
            "message": "Redis connection successful"
        }
    except Exception as e:
        logger.error("Redis ping failed", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_key(key: str) -> Dict[str, Any]:
    """Get value for a key.

    Args:
        key: Redis key

    Returns:
        Dict containing key value and type
    """
    try:
        client = get_redis_client()
        key_type = client.type(key)

        if key_type == "none":
            return {
                "success": False,
                "error": f"Key '{key}' does not exist"
            }

        # Get value based on type
        if key_type == "string":
            value = client.get(key)
        elif key_type == "hash":
            value = client.hgetall(key)
        elif key_type == "list":
            value = client.lrange(key, 0, -1)
        elif key_type == "set":
            value = list(client.smembers(key))
        elif key_type == "zset":
            value = client.zrange(key, 0, -1, withscores=True)
        else:
            value = f"Unsupported type: {key_type}"

        # Get TTL if exists
        ttl = client.ttl(key)
        ttl_info = None
        if ttl > 0:
            ttl_info = ttl
        elif ttl == -1:
            ttl_info = "no expiration"

        return {
            "success": True,
            "key": key,
            "type": key_type,
            "value": value,
            "ttl": ttl_info
        }

    except Exception as e:
        logger.error("Error getting key", key=key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def set_key(key: str, value: str, ex: Optional[int] = None) -> Dict[str, Any]:
    """Set a string key.

    Args:
        key: Redis key
        value: Value to set
        ex: Optional expiration in seconds

    Returns:
        Dict with operation result
    """
    try:
        client = get_redis_client()
        client.set(key, value, ex=ex)

        return {
            "success": True,
            "key": key,
            "message": f"Key '{key}' set successfully" + (f" with expiration {ex}s" if ex else "")
        }

    except Exception as e:
        logger.error("Error setting key", key=key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def delete_key(key: str) -> Dict[str, Any]:
    """Delete a key.

    Args:
        key: Redis key to delete

    Returns:
        Dict with operation result
    """
    try:
        client = get_redis_client()
        deleted = client.delete(key)

        return {
            "success": True,
            "deleted": bool(deleted),
            "message": f"Deleted {deleted} key(s)"
        }

    except Exception as e:
        logger.error("Error deleting key", key=key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_keys(pattern: str = "*", limit: int = 100) -> Dict[str, Any]:
    """Get keys matching a pattern.

    Args:
        pattern: Redis key pattern (default: "*")
        limit: Maximum number of keys to return

    Returns:
        Dict with matching keys
    """
    try:
        client = get_redis_client()

        # Use SCAN for better performance on large datasets
        keys = []
        cursor = 0
        while len(keys) < limit:
            cursor, batch = client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)

            if cursor == 0:  # Full iteration complete
                break

        # Trim to limit
        keys = keys[:limit]

        return {
            "success": True,
            "pattern": pattern,
            "count": len(keys),
            "keys": keys
        }

    except Exception as e:
        logger.error("Error scanning keys", pattern=pattern, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_db_info() -> Dict[str, Any]:
    """Get Redis database information.

    Returns:
        Dict with database statistics
    """
    try:
        client = get_redis_client()
        info = client.info()
        keyspace = client.info("keyspace")

        # Current database
        current_db = client.connection_pool.connection_kwargs.get("db", 0)
        db_info = keyspace.get(f"db{current_db}", {})

        return {
            "success": True,
            "redis_version": info.get("redis_version"),
            "redis_mode": info.get("redis_mode", "standalone"),
            "current_db": current_db,
            "keys_count": db_info.get("keys", 0),
            "expires_count": db_info.get("expires", 0),
            "avg_ttl": db_info.get("avg_ttl", 0),
            "used_memory": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "uptime_days": info.get("uptime_in_days")
        }

    except Exception as e:
        logger.error("Error getting database info", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_schema(use_cache: bool = True, max_keys: int = 1000) -> Dict[str, Any]:
    """Infer Redis schema by analyzing key patterns and types.

    Args:
        use_cache: Whether to use cached schema
        max_keys: Maximum number of keys to analyze

    Returns:
        Dict containing schema information
    """
    cache_key = "redis_schema"

    # Check cache
    if use_cache and cache_key in _schema_cache:
        timestamp, cached_schema = _schema_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug("Using cached schema", age_seconds=int(time.time() - timestamp))
            return cached_schema

    try:
        client = get_redis_client()

        # Get all keys (with limit)
        all_keys = []
        cursor = 0
        while len(all_keys) < max_keys:
            cursor, batch = client.scan(cursor, count=100)
            all_keys.extend(batch)
            if cursor == 0:
                break

        all_keys = all_keys[:max_keys]

        # Analyze key patterns
        patterns = _analyze_key_patterns(all_keys)

        # Get type distribution
        type_distribution = {}
        pattern_details = {}

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            matching_keys = [k for k in all_keys if _matches_pattern(k, pattern)]

            # Sample keys from this pattern
            sample_keys = matching_keys[:10]  # Sample first 10
            types = {}
            examples = []

            for key in sample_keys:
                key_type = client.type(key)
                types[key_type] = types.get(key_type, 0) + 1

                # Get example value
                if key_type == "string":
                    value = client.get(key)
                    examples.append({"key": key, "type": key_type, "value": value[:100] if len(str(value)) > 100 else value})
                elif key_type == "hash":
                    value = client.hgetall(key)
                    examples.append({"key": key, "type": key_type, "fields": list(value.keys())})
                elif key_type == "list":
                    length = client.llen(key)
                    examples.append({"key": key, "type": key_type, "length": length})
                elif key_type == "set":
                    card = client.scard(key)
                    examples.append({"key": key, "type": key_type, "cardinality": card})
                elif key_type == "zset":
                    card = client.zcard(key)
                    examples.append({"key": key, "type": key_type, "cardinality": card})

            pattern_details[pattern] = {
                "count": len(matching_keys),
                "types": types,
                "examples": examples[:5]  # Limit examples
            }

            # Update global type distribution
            for t, count in types.items():
                type_distribution[t] = type_distribution.get(t, 0) + count

        # Get database info
        db_info = get_db_info()

        schema = {
            "total_keys": len(all_keys),
            "analyzed_keys": len(all_keys),
            "key_patterns": patterns,
            "pattern_details": pattern_details,
            "type_distribution": type_distribution,
            "database_info": db_info
        }

        # Cache the schema
        _schema_cache[cache_key] = (time.time(), schema)

        logger.info(
            "Generated Redis schema",
            total_keys=len(all_keys),
            patterns=len(patterns),
            types=len(type_distribution)
        )

        return schema

    except Exception as e:
        logger.error("Error getting schema", error=str(e))
        raise


def _analyze_key_patterns(keys: List[str]) -> List[Dict[str, Any]]:
    """Analyze keys to detect common patterns.

    Args:
        keys: List of Redis keys

    Returns:
        List of detected patterns with counts
    """
    # Count patterns
    pattern_counts = {}

    for key in keys:
        # Extract pattern by replacing numbers and UUIDs with placeholders
        pattern = re.sub(r'\d+', '{id}', key)
        pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '{uuid}', pattern)
        pattern = re.sub(r'[a-z0-9]{32}', '{hash}', pattern)

        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Sort by count and convert to list
    patterns = [
        {"pattern": pattern, "count": count, "percentage": round(count / len(keys) * 100, 2)}
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return patterns


def _matches_pattern(key: str, pattern: str) -> bool:
    """Check if a key matches a pattern.

    Args:
        key: Redis key
        pattern: Pattern with {id}, {uuid}, {hash} placeholders

    Returns:
        bool: True if key matches pattern
    """
    # Convert pattern to regex
    regex_pattern = pattern.replace('{id}', r'\d+')
    regex_pattern = regex_pattern.replace('{uuid}', r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}')
    regex_pattern = regex_pattern.replace('{hash}', r'[a-z0-9]{32}')
    regex_pattern = f"^{regex_pattern}$"

    return bool(re.match(regex_pattern, key))


def clear_schema_cache() -> Dict[str, Any]:
    """Clear the schema cache.

    Returns:
        Dict with operation result
    """
    global _schema_cache
    _schema_cache.clear()
    logger.info("Schema cache cleared")

    return {
        "success": True,
        "message": "Schema cache cleared"
    }


def execute_command(command: str, *args) -> Dict[str, Any]:
    """Execute a raw Redis command.

    Args:
        command: Redis command (e.g., "GET", "SET", "HGETALL")
        *args: Command arguments

    Returns:
        Dict with command result
    """
    try:
        client = get_redis_client()
        result = client.execute_command(command, *args)

        return {
            "success": True,
            "command": command,
            "result": result
        }

    except Exception as e:
        logger.error("Error executing command", command=command, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }
