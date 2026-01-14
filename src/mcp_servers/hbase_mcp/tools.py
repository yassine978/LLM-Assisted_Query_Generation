"""HBase Tools - Core HBase operations and schema inference with enhanced filtering.

This module provides tools for interacting with HBase, including:
- Connection management with URI support
- Table operations (create, delete, scan, etc.)
- Row operations (get, put, delete) with enhanced filtering
- Schema inference and column family analysis
- Scan and filter operations with value-based filtering
"""

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import happybase
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global connection pool
_hbase_connection: Optional[happybase.Connection] = None

# Schema cache with TTL
_schema_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 300  # 5 minutes


def get_hbase_connection() -> happybase.Connection:
    """Get or create HBase connection with automatic retry on stale connections.

    Returns:
        happybase.Connection: Connected HBase client

    Raises:
        Exception: If connection fails
    """
    global _hbase_connection

    # Test existing connection if it exists
    if _hbase_connection is not None:
        try:
            # Quick test to see if connection is alive
            _hbase_connection.tables()
            return _hbase_connection
        except Exception as e:
            logger.warning("Existing HBase connection is stale, reconnecting", error=str(e))
            try:
                _hbase_connection.close()
            except:
                pass
            _hbase_connection = None

    # Create new connection
    if _hbase_connection is None:
        settings = get_settings()
        hbase_uri = settings.get_hbase_uri()

        logger.info("Connecting to HBase", uri=hbase_uri)

        try:
            # Parse URI to extract host, port, and protocol
            # Format: hbase+thrift://host:port?protocol=compact
            parsed = urlparse(hbase_uri)

            host = parsed.hostname or settings.hbase_host
            port = parsed.port or settings.hbase_port

            # Extract protocol from query string
            query_params = parse_qs(parsed.query)
            protocol = query_params.get('protocol', [settings.hbase_thrift_protocol])[0]

            _hbase_connection = happybase.Connection(
                host=host,
                port=port,
                protocol=protocol,
                timeout=10000  # 10 seconds
            )

            # Test connection
            _hbase_connection.tables()
            logger.info("HBase connection successful")

        except Exception as e:
            logger.error("HBase connection failed", error=str(e))
            _hbase_connection = None
            raise

    return _hbase_connection


def close_hbase_connection():
    """Close the HBase connection."""
    global _hbase_connection
    if _hbase_connection:
        _hbase_connection.close()
        _hbase_connection = None
        logger.info("HBase connection closed")


def ping() -> Dict[str, Any]:
    """Test HBase connection.

    Returns:
        Dict with connection status
    """
    try:
        connection = get_hbase_connection()
        # Test connection by listing tables
        connection.tables()

        # Try to get version if available
        version = "unknown"
        try:
            version = connection.version
        except AttributeError:
            pass

        return {
            "success": True,
            "message": "HBase connection successful",
            "version": version
        }
    except Exception as e:
        logger.error("HBase ping failed", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def list_tables() -> Dict[str, Any]:
    """List all tables in HBase.

    Returns:
        Dict with list of table names
    """
    try:
        connection = get_hbase_connection()
        tables = connection.tables()

        # Convert bytes to strings
        table_names = [
            t.decode('utf-8') if isinstance(t, bytes) else t
            for t in tables
        ]

        return {
            "success": True,
            "tables": table_names,
            "count": len(table_names)
        }
    except Exception as e:
        logger.error("Error listing tables", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_table_info(table_name: str) -> Dict[str, Any]:
    """Get information about a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Dict with table information including column families
    """
    try:
        connection = get_hbase_connection()
        table = connection.table(table_name)

        # Get table regions/families
        families = table.families()

        # Convert families dict
        families_info = {}
        for family_name, family_props in families.items():
            family_key = family_name.decode('utf-8') if isinstance(family_name, bytes) else family_name
            families_info[family_key] = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in family_props.items()
            }

        return {
            "success": True,
            "table": table_name,
            "column_families": families_info,
            "family_count": len(families_info)
        }
    except Exception as e:
        logger.error("Error getting table info", table=table_name, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def scan_table(
    table_name: str,
    row_start: Optional[str] = None,
    row_stop: Optional[str] = None,
    limit: int = 100,
    columns: Optional[Dict[str, Any]] = None,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_operator: Optional[str] = None
) -> Dict[str, Any]:
    """Scan rows from a table with optional filtering.

    Args:
        table_name: Name of the table
        row_start: Optional start row key
        row_stop: Optional stop row key
        limit: Maximum number of rows to return
        columns: Dict specifying which columns to retrieve
                Example: {"profile": ["name", "age"]} or {"profile:name": ""}
        filter_column: Optional column name to filter on (e.g., "preferences:theme")
        filter_value: Optional value to match for filtering (e.g., "dark")
        filter_operator: Optional comparison operator ("=", ">", "<", ">=", "<=", "!=")
                        Defaults to "=" if not specified

    Returns:
        Dict with scanned rows

    Examples:
        - scan_table("users") - Get first 100 rows with all columns
        - scan_table("users", limit=10) - Get first 10 rows
        - scan_table("users", row_start="user_1000", row_stop="user_2000")
        - scan_table("users", filter_column="preferences:theme", filter_value="dark")
        - scan_table("users", filter_column="profile:age", filter_value="30", filter_operator=">")
        - scan_table("users", columns={"profile": ["name", "age"]})
    """
    try:
        connection = get_hbase_connection()
        table = connection.table(table_name)

        # Parse columns parameter for HBase column filtering
        column_list = None
        if columns and isinstance(columns, dict) and columns:
            column_list = []
            for key, val in columns.items():
                if isinstance(val, list):
                    # Format: {"family": ["qual1", "qual2"]}
                    for qual in val:
                        column_spec = f"{key}:{qual}".encode('utf-8')
                        column_list.append(column_spec)
                elif isinstance(val, str) and val:
                    # Format: {"family:qualifier": "specific_value"}
                    column_spec = key.encode('utf-8')
                    column_list.append(column_spec)
                else:
                    # Format: {"family": ""} - get all qualifiers in family
                    # HBase doesn't support family-only selection,
                    # so we skip this level
                    pass

        # CRITICAL FIX: If filtering by a column, ensure that column is retrieved
        if filter_column:
            if column_list is None:
                column_list = []
            filter_col_bytes = filter_column.encode('utf-8')
            if filter_col_bytes not in column_list:
                column_list.append(filter_col_bytes)
                logger.debug(
                    "Added filter column to retrieval list",
                    filter_column=filter_column
                )

        # Scan table
        rows = []
        scanned_count = 0
        
        for row_key, row_data in table.scan(
            row_start=row_start.encode('utf-8') if row_start else None,
            row_stop=row_stop.encode('utf-8') if row_stop else None,
            limit=limit,
            columns=column_list if column_list else None
        ):
            scanned_count += 1
            
            # Decode row key and data
            decoded_row = {
                "row_key": row_key.decode('utf-8') if isinstance(row_key, bytes) else row_key,
                "columns": {}
            }

            for col_key, col_value in row_data.items():
                col_name = col_key.decode('utf-8') if isinstance(col_key, bytes) else col_key
                col_val = col_value.decode('utf-8') if isinstance(col_value, bytes) else col_value
                decoded_row["columns"][col_name] = col_val

            # Apply value-based filtering (client-side)
            if filter_column and filter_value is not None:
                if filter_column in decoded_row["columns"]:
                    actual_value = decoded_row["columns"][filter_column]

                    # Determine if we should apply filtering
                    should_include = False

                    # Default operator to "=" if not specified
                    operator = filter_operator or "="

                    # Try numeric comparison first
                    try:
                        actual_num = float(actual_value)
                        filter_num = float(filter_value)

                        if operator == "=":
                            should_include = actual_num == filter_num
                        elif operator == ">":
                            should_include = actual_num > filter_num
                        elif operator == "<":
                            should_include = actual_num < filter_num
                        elif operator == ">=":
                            should_include = actual_num >= filter_num
                        elif operator == "<=":
                            should_include = actual_num <= filter_num
                        elif operator == "!=":
                            should_include = actual_num != filter_num
                        else:
                            # Unknown operator, default to equality
                            should_include = actual_num == filter_num

                    except (ValueError, TypeError):
                        # Not numeric, do string comparison
                        if operator == "=":
                            should_include = actual_value == filter_value
                        elif operator == "!=":
                            should_include = actual_value != filter_value
                        elif operator in [">", "<", ">=", "<="]:
                            # Lexicographic string comparison
                            if operator == ">":
                                should_include = actual_value > filter_value
                            elif operator == "<":
                                should_include = actual_value < filter_value
                            elif operator == ">=":
                                should_include = actual_value >= filter_value
                            elif operator == "<=":
                                should_include = actual_value <= filter_value
                        else:
                            # Default to equality for unknown operators
                            should_include = actual_value == filter_value

                    if should_include:
                        rows.append(decoded_row)
                # Skip this row if filter column doesn't exist or filter doesn't match
            else:
                rows.append(decoded_row)

        logger.info(
            "Table scan completed",
            table=table_name,
            scanned_count=scanned_count,
            returned_count=len(rows),
            filter_applied=bool(filter_column and filter_value)
        )

        return {
            "success": True,
            "table": table_name,
            "rows": rows,
            "count": len(rows),
            "scanned_count": scanned_count
        }
    except Exception as e:
        logger.error("Error scanning table", table=table_name, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_row(table_name: str, row_key: str) -> Dict[str, Any]:
    """Get a specific row from a table.

    Args:
        table_name: Name of the table
        row_key: Row key to retrieve

    Returns:
        Dict with row data, or error if row not found
    """
    try:
        connection = get_hbase_connection()
        table = connection.table(table_name)

        # Get row
        row_data = table.row(row_key.encode('utf-8'))

        if not row_data:
            logger.info(
                "Row not found",
                table=table_name,
                row_key=row_key
            )
            return {
                "success": False,
                "error": f"Row '{row_key}' not found in table '{table_name}'"
            }

        # Decode data
        decoded_data = {}
        for col_key, col_value in row_data.items():
            col_name = col_key.decode('utf-8') if isinstance(col_key, bytes) else col_key
            col_val = col_value.decode('utf-8') if isinstance(col_value, bytes) else col_value
            decoded_data[col_name] = col_val

        logger.info(
            "Row retrieved successfully",
            table=table_name,
            row_key=row_key,
            column_count=len(decoded_data)
        )

        return {
            "success": True,
            "table": table_name,
            "row_key": row_key,
            "columns": decoded_data
        }
    except Exception as e:
        logger.error("Error getting row", table=table_name, row_key=row_key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def put_row(table_name: str, row_key: str, columns: Dict[str, str]) -> Dict[str, Any]:
    """Put (insert/update) a row in a table.

    Args:
        table_name: Name of the table
        row_key: Row key
        columns: Dictionary of column:value pairs (e.g., {"profile:name": "Alice"})

    Returns:
        Dict with operation result
    """
    try:
        connection = get_hbase_connection()
        table = connection.table(table_name)

        # Encode columns
        encoded_columns = {
            k.encode('utf-8'): v.encode('utf-8')
            for k, v in columns.items()
        }

        # Put row
        table.put(row_key.encode('utf-8'), encoded_columns)

        logger.info(
            "Row inserted/updated",
            table=table_name,
            row_key=row_key,
            column_count=len(encoded_columns)
        )

        return {
            "success": True,
            "table": table_name,
            "row_key": row_key,
            "message": f"Row '{row_key}' inserted/updated successfully"
        }
    except Exception as e:
        logger.error("Error putting row", table=table_name, row_key=row_key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def delete_row(table_name: str, row_key: str) -> Dict[str, Any]:
    """Delete a row from a table.

    Args:
        table_name: Name of the table
        row_key: Row key to delete

    Returns:
        Dict with operation result
    """
    try:
        connection = get_hbase_connection()
        table = connection.table(table_name)

        # Delete row
        table.delete(row_key.encode('utf-8'))

        logger.info(
            "Row deleted",
            table=table_name,
            row_key=row_key
        )

        return {
            "success": True,
            "table": table_name,
            "row_key": row_key,
            "message": f"Row '{row_key}' deleted successfully"
        }
    except Exception as e:
        logger.error("Error deleting row", table=table_name, row_key=row_key, error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_schema(use_cache: bool = True, sample_limit: int = 100) -> Dict[str, Any]:
    """Infer HBase schema by analyzing tables and their structure.

    Args:
        use_cache: Whether to use cached schema
        sample_limit: Maximum number of rows to sample per table

    Returns:
        Dict containing schema information with improved row key pattern detection
    """
    cache_key = "hbase_schema"

    # Check cache
    if use_cache and cache_key in _schema_cache:
        timestamp, cached_schema = _schema_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug("Using cached schema", age_seconds=int(time.time() - timestamp))
            return cached_schema

    try:
        connection = get_hbase_connection()

        # Get all tables
        tables = connection.tables()
        table_names = [
            t.decode('utf-8') if isinstance(t, bytes) else t
            for t in tables
        ]

        # Analyze each table
        tables_schema = []
        for table_name in table_names:
            table_info = _analyze_table(connection, table_name, sample_limit)
            tables_schema.append(table_info)

        schema = {
            "total_tables": len(tables_schema),
            "tables": tables_schema
        }

        # Cache the schema
        _schema_cache[cache_key] = (time.time(), schema)

        logger.info(
            "Generated HBase schema",
            total_tables=len(tables_schema)
        )

        return schema

    except Exception as e:
        logger.error("Error getting schema", error=str(e))
        raise


def _analyze_table(connection: happybase.Connection, table_name: str, sample_limit: int) -> Dict[str, Any]:
    """Analyze a single table's structure with improved row key pattern detection.

    Args:
        connection: HBase connection
        table_name: Name of the table to analyze
        sample_limit: Maximum number of rows to sample

    Returns:
        Dict with table schema information including row key patterns
    """
    table = connection.table(table_name)

    # Get column families
    families = table.families()
    families_info = {}
    for family_name, family_props in families.items():
        family_key = family_name.decode('utf-8') if isinstance(family_name, bytes) else family_name
        families_info[family_key] = {
            k.decode('utf-8') if isinstance(k, bytes) else k:
            v.decode('utf-8') if isinstance(v, bytes) else v
            for k, v in family_props.items()
        }

    # Sample rows to find column qualifiers
    column_qualifiers = {}  # family -> set of qualifiers
    row_count = 0
    sample_rows = []
    row_key_pattern = None

    for row_key, row_data in table.scan(limit=sample_limit):
        row_count += 1

        decoded_row_key = row_key.decode('utf-8') if isinstance(row_key, bytes) else row_key
        decoded_row = {
            "row_key": decoded_row_key,
            "columns": {}
        }

        # Detect row key pattern (first row)
        if row_count == 1 and row_key_pattern is None:
            # Try to identify the pattern (e.g., "user_", "product_", "order_", etc.)
            if '_' in decoded_row_key:
                parts = decoded_row_key.split('_')
                if parts[0]:  # If prefix exists
                    row_key_pattern = f"{parts[0]}_*"
            elif any(c.isalpha() for c in decoded_row_key):
                # Alphanumeric pattern
                row_key_pattern = "Similar to: " + decoded_row_key[:20] + "*"

        for col_key, col_value in row_data.items():
            col_name = col_key.decode('utf-8') if isinstance(col_key, bytes) else col_key
            col_val = col_value.decode('utf-8') if isinstance(col_value, bytes) else col_value

            # Extract family and qualifier
            if ':' in col_name:
                family, qualifier = col_name.split(':', 1)
                if family not in column_qualifiers:
                    column_qualifiers[family] = set()
                column_qualifiers[family].add(qualifier)

            decoded_row["columns"][col_name] = col_val

        # Keep first 5 rows as samples
        if len(sample_rows) < 5:
            sample_rows.append(decoded_row)

    # Convert sets to lists for JSON serialization
    qualifiers_by_family = {
        family: sorted(list(qualifiers))
        for family, qualifiers in column_qualifiers.items()
    }

    return {
        "table_name": table_name,
        "column_families": list(families_info.keys()),
        "column_qualifiers": qualifiers_by_family,
        "estimated_row_count": row_count,
        "row_key_pattern": row_key_pattern,  # NEW: helps LLM understand row key format
        "sample_rows": sample_rows
    }


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
        Dict with count result
    """
    # Use scan_table with limit set high enough to get all rows
    result = scan_table(
        table_name=table_name,
        row_start=row_start,
        row_stop=row_stop,
        limit=10000,  # High limit for counting
        filter_column=filter_column,
        filter_value=filter_value,
        filter_operator=filter_operator
    )

    if result.get("success"):
        return {
            "success": True,
            "table": table_name,
            "count": result.get("count", 0),
            "scanned_count": result.get("scanned_count", 0),
            "message": f"Found {result.get('count', 0)} rows matching criteria"
        }
    else:
        return result


def clear_schema_cache() -> Dict[str, Any]:
    """Clear the HBase schema cache.

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
