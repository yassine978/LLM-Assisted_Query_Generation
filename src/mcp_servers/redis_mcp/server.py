"""Redis MCP Server - FastMCP server for Redis operations.

This module exposes Redis operations through the MCP (Model Context Protocol),
allowing LLMs to interact with Redis key-value store.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP
from src.mcp_servers.redis_mcp import tools

# Create FastMCP server
mcp = FastMCP("redis-mcp")


# ============================================================================
# Redis Connection Tools
# ============================================================================

@mcp.tool()
def ping() -> Dict[str, Any]:
    """Test Redis connection.

    Returns:
        Connection status
    """
    return tools.ping()


@mcp.tool()
def get_db_info() -> Dict[str, Any]:
    """Get Redis database information and statistics.

    Returns:
        Database information including:
        - Redis version
        - Number of keys
        - Memory usage
        - Connected clients
        - Uptime
    """
    return tools.get_db_info()


# ============================================================================
# Key Operations
# ============================================================================

@mcp.tool()
def get_key(key: str) -> Dict[str, Any]:
    """Get value for a Redis key.

    Args:
        key: Redis key to retrieve

    Returns:
        Key value, type, and TTL information
    """
    return tools.get_key(key)


@mcp.tool()
def set_key(key: str, value: str, ex: int = None) -> Dict[str, Any]:
    """Set a string key in Redis.

    Args:
        key: Redis key
        value: Value to set
        ex: Optional expiration time in seconds

    Returns:
        Operation result
    """
    return tools.set_key(key, value, ex)


@mcp.tool()
def delete_key(key: str) -> Dict[str, Any]:
    """Delete a Redis key.

    Args:
        key: Redis key to delete

    Returns:
        Deletion result
    """
    return tools.delete_key(key)


@mcp.tool()
def get_keys(pattern: str = "*", limit: int = 100) -> Dict[str, Any]:
    """Get Redis keys matching a pattern.

    Args:
        pattern: Key pattern (supports * and ? wildcards, default: "*")
        limit: Maximum number of keys to return (default: 100)

    Returns:
        List of matching keys

    Examples:
        - get_keys("user:*") - Get all user keys
        - get_keys("session:*") - Get all session keys
        - get_keys("*:config") - Get all config keys
    """
    return tools.get_keys(pattern, limit)


# ============================================================================
# Schema Tools
# ============================================================================

@mcp.tool()
def get_schema(use_cache: bool = True, max_keys: int = 1000) -> Dict[str, Any]:
    """Get comprehensive Redis schema information.

    Analyzes keys in Redis to infer:
    - Key patterns and naming conventions
    - Data type distribution
    - Example keys and values
    - Statistics per pattern

    Args:
        use_cache: Whether to use cached schema (default: True, 5-minute TTL)
        max_keys: Maximum number of keys to analyze (default: 1000)

    Returns:
        Comprehensive schema information including:
        - Total keys
        - Key patterns
        - Type distribution
        - Pattern details with examples
        - Database information
    """
    return tools.get_schema(use_cache, max_keys)


@mcp.tool()
def clear_schema_cache() -> Dict[str, Any]:
    """Clear the Redis schema cache.

    Use this to force a fresh schema analysis on the next get_schema call.

    Returns:
        Operation result
    """
    return tools.clear_schema_cache()


# ============================================================================
# Raw Command Execution
# ============================================================================

@mcp.tool()
def execute_command(command: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute a raw Redis command.

    Args:
        command: Redis command (e.g., "GET", "SET", "HGETALL")
        args: Command arguments as a list (optional)

    Returns:
        Command execution result

    Examples:
        - execute_command("GET", ["mykey"])
        - execute_command("HGETALL", ["user:1001"])
        - execute_command("LRANGE", ["logs:user:1001", "0", "-1"])
    """
    if args is None:
        args = []
    return tools.execute_command(command, *args)


# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("redis://schema")
def get_schema_resource() -> str:
    """Get Redis schema as a resource.

    Returns:
        JSON string with schema information
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema, indent=2, default=str)


@mcp.resource("redis://info")
def get_info_resource() -> str:
    """Get Redis database info as a resource.

    Returns:
        JSON string with database information
    """
    info = tools.get_db_info()
    return json.dumps(info, indent=2, default=str)


@mcp.resource("redis://keys/{pattern}")
def get_keys_resource(pattern: str) -> str:
    """Get keys matching a pattern as a resource.

    Args:
        pattern: Key pattern (e.g., "user:*", "session:*")

    Returns:
        JSON string with matching keys
    """
    keys = tools.get_keys(pattern, limit=100)
    return json.dumps(keys, indent=2)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    mcp.run()
