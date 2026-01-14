"""HBase MCP Server - FastMCP server for HBase operations.

This module exposes HBase operations through the MCP (Model Context Protocol),
allowing LLMs to interact with HBase wide-column store.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP
from src.mcp_servers.hbase_mcp import tools

# Create FastMCP server
mcp = FastMCP("hbase-mcp")


# ============================================================================
# HBase Connection Tools
# ============================================================================

@mcp.tool()
def ping() -> Dict[str, Any]:
    """Test HBase connection.

    Returns:
        Connection status and HBase version
    """
    return tools.ping()


@mcp.tool()
def list_tables() -> Dict[str, Any]:
    """List all tables in HBase.

    Returns:
        List of table names
    """
    return tools.list_tables()


@mcp.tool()
def get_table_info(table_name: str) -> Dict[str, Any]:
    """Get information about a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Table information including column families
    """
    return tools.get_table_info(table_name)


# ============================================================================
# Row Operations
# ============================================================================

@mcp.tool()
def get_row(table_name: str, row_key: str) -> Dict[str, Any]:
    """Get a specific row from a table.

    Args:
        table_name: Name of the table
        row_key: Row key to retrieve

    Returns:
        Row data with all columns
    """
    return tools.get_row(table_name, row_key)


@mcp.tool()
def scan_table(
    table_name: str,
    row_start: Optional[str] = None,
    row_stop: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Scan rows from a table.

    Args:
        table_name: Name of the table
        row_start: Optional start row key
        row_stop: Optional stop row key
        limit: Maximum number of rows to return (default: 100)

    Returns:
        Scanned rows

    Examples:
        - scan_table("users") - Get first 100 rows
        - scan_table("users", limit=10) - Get first 10 rows
        - scan_table("users", row_start="user_1000", row_stop="user_2000") - Scan range
    """
    return tools.scan_table(table_name, row_start, row_stop, limit)


@mcp.tool()
def put_row(table_name: str, row_key: str, columns: Dict[str, str]) -> Dict[str, Any]:
    """Put (insert/update) a row in a table.

    Args:
        table_name: Name of the table
        row_key: Row key
        columns: Dictionary of column:value pairs (e.g., {"profile:name": "Alice"})

    Returns:
        Operation result

    Example:
        put_row("users", "user_1001", {"profile:name": "Alice", "contact:email": "alice@example.com"})
    """
    return tools.put_row(table_name, row_key, columns)


@mcp.tool()
def delete_row(table_name: str, row_key: str) -> Dict[str, Any]:
    """Delete a row from a table.

    Args:
        table_name: Name of the table
        row_key: Row key to delete

    Returns:
        Deletion result
    """
    return tools.delete_row(table_name, row_key)


@mcp.tool()
def count_rows(
    table_name: str,
    row_start: Optional[str] = None,
    row_stop: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_operator: Optional[str] = None
) -> Dict[str, Any]:
    """Count rows in a table with optional filtering.

    Args:
        table_name: Name of the table
        row_start: Optional start row key
        row_stop: Optional stop row key
        filter_column: Optional column name to filter on
        filter_value: Optional value to match for filtering
        filter_operator: Optional comparison operator ("=", ">", "<", ">=", "<=", "!=")

    Returns:
        Count result

    Examples:
        - count_rows("users") - Count all users
        - count_rows("users", filter_column="profile:age", filter_value="30", filter_operator=">")
    """
    return tools.count_rows(
        table_name=table_name,
        row_start=row_start,
        row_stop=row_stop,
        filter_column=filter_column,
        filter_value=filter_value,
        filter_operator=filter_operator
    )


# ============================================================================
# Schema Tools
# ============================================================================

@mcp.tool()
def get_schema(use_cache: bool = True, sample_limit: int = 100) -> Dict[str, Any]:
    """Get comprehensive HBase schema information.

    Analyzes all tables in HBase to infer:
    - Table names
    - Column families
    - Column qualifiers
    - Sample rows
    - Estimated row counts

    Args:
        use_cache: Whether to use cached schema (default: True, 5-minute TTL)
        sample_limit: Maximum number of rows to sample per table (default: 100)

    Returns:
        Comprehensive schema information including:
        - Total tables
        - Tables with their column families and qualifiers
        - Sample rows per table
    """
    return tools.get_schema(use_cache, sample_limit)


@mcp.tool()
def clear_schema_cache() -> Dict[str, Any]:
    """Clear the HBase schema cache.

    Use this to force a fresh schema analysis on the next get_schema call.

    Returns:
        Operation result
    """
    return tools.clear_schema_cache()


# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("hbase://schema")
def get_schema_resource() -> str:
    """Get HBase schema as a resource.

    Returns:
        JSON string with schema information
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema, indent=2, default=str)


@mcp.resource("hbase://tables")
def get_tables_resource() -> str:
    """Get list of all tables as a resource.

    Returns:
        JSON string with table list
    """
    tables = tools.list_tables()
    return json.dumps(tables, indent=2)


@mcp.resource("hbase://table/{table_name}")
def get_table_resource(table_name: str) -> str:
    """Get information about a specific table as a resource.

    Args:
        table_name: Name of the table

    Returns:
        JSON string with table information
    """
    table_info = tools.get_table_info(table_name)
    return json.dumps(table_info, indent=2, default=str)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    mcp.run()
