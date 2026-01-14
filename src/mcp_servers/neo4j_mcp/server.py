"""Neo4j MCP Server implementation using FastMCP."""

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

from . import tools

# Note: We use basic logging instead of structlog to avoid any stdout interference
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("neo4j-mcp")


@mcp.tool()
def get_node_labels() -> List[str]:
    """
    List all node labels in the Neo4j database.

    Returns:
        List[str]: List of node label names
    """
    try:
        return tools.get_node_labels()
    except Exception as e:
        logger.error(f"Error listing node labels: {e}")
        raise


@mcp.tool()
def get_relationship_types() -> List[str]:
    """
    List all relationship types in the Neo4j database.

    Returns:
        List[str]: List of relationship type names
    """
    try:
        return tools.get_relationship_types()
    except Exception as e:
        logger.error(f"Error listing relationship types: {e}")
        raise


@mcp.tool()
def get_schema(use_cache: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive schema information from Neo4j.

    Enhanced version includes:
    - Per-label node counts
    - Per-type relationship counts
    - Graph visualization data (label-to-label connections)
    - Caching with 5-minute TTL

    Args:
        use_cache: Whether to use cached schema if available (default: True)

    Returns:
        Dict containing comprehensive schema information
    """
    try:
        return tools.get_schema(use_cache)
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise


@mcp.tool()
def get_node_properties(label: str) -> Dict[str, Any]:
    """
    Get properties for a specific node label.

    Args:
        label: Node label name

    Returns:
        Dict containing property information
    """
    try:
        return tools.get_node_properties(label)
    except Exception as e:
        logger.error(f"Error getting node properties for {label}: {e}")
        raise


@mcp.tool()
def get_relationship_properties(rel_type: str) -> Dict[str, Any]:
    """
    Get properties for a specific relationship type.

    Args:
        rel_type: Relationship type name

    Returns:
        Dict containing property information
    """
    try:
        return tools.get_relationship_properties(rel_type)
    except Exception as e:
        logger.error(f"Error getting relationship properties for {rel_type}: {e}")
        raise


@mcp.tool()
def get_sample_nodes(label: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get sample nodes for a specific label.

    Args:
        label: Node label name
        limit: Number of sample nodes to return (default: 10)

    Returns:
        Dict containing sample nodes with full properties
    """
    try:
        return tools.get_sample_nodes(label, limit)
    except Exception as e:
        logger.error(f"Error getting sample nodes for {label}: {e}")
        raise


@mcp.tool()
def get_sample_relationships(rel_type: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get sample relationships for a specific type.

    Args:
        rel_type: Relationship type name
        limit: Number of sample relationships to return (default: 10)

    Returns:
        Dict containing sample relationships with source and target nodes
    """
    try:
        return tools.get_sample_relationships(rel_type, limit)
    except Exception as e:
        logger.error(f"Error getting sample relationships for {rel_type}: {e}")
        raise


@mcp.tool()
def run_cypher(query: str, limit: int = 100) -> Dict[str, Any]:
    """
    Execute a Cypher query on Neo4j.

    Args:
        query: Cypher query string
        limit: Maximum number of results (default: 100)

    Returns:
        Dict containing query results and metadata
    """
    try:
        return tools.run_cypher(query, limit)
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}")
        raise


@mcp.tool()
def explain_cypher(query: str) -> Dict[str, Any]:
    """
    Get execution plan for a Cypher query without executing it.

    Args:
        query: Cypher query string

    Returns:
        Dict containing query execution plan
    """
    try:
        return tools.explain_cypher(query)
    except Exception as e:
        logger.error(f"Error explaining Cypher query: {e}")
        raise


@mcp.tool()
def validate_cypher(query: str) -> Dict[str, Any]:
    """
    Validate a Cypher query without executing it.

    Args:
        query: Cypher query string

    Returns:
        Dict containing validation results
    """
    try:
        return tools.validate_cypher(query)
    except Exception as e:
        logger.error(f"Error validating Cypher query: {e}")
        raise


@mcp.tool()
def clear_schema_cache() -> Dict[str, str]:
    """
    Clear the Neo4j schema cache.

    Returns:
        Dict with status message
    """
    try:
        tools.clear_schema_cache()
        logger.info("Cleared Neo4j schema cache")
        return {"status": "success", "message": "Cleared Neo4j schema cache"}
    except Exception as e:
        logger.error(f"Error clearing schema cache: {e}")
        return {"status": "error", "message": str(e)}


# MCP Resources (with caching via enhanced tools)
@mcp.resource("neo4j://schema")
def get_schema_resource() -> str:
    """
    Get comprehensive graph schema as a resource.

    Includes:
    - Node labels and relationship types
    - Per-label node counts
    - Per-type relationship counts
    - Property keys
    - Constraints and indexes
    - Graph visualization data
    - Cached with 5-minute TTL

    Returns:
        JSON string of comprehensive schema information
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema, indent=2, default=str)


@mcp.resource("neo4j://schema/nodes")
def get_nodes_schema_resource() -> str:
    """
    Get node labels and counts as a resource.

    Returns:
        JSON string of node label information
    """
    schema = tools.get_schema(use_cache=True)
    nodes_info = {
        "labels": schema["labels"],
        "labelCounts": schema["labelCounts"],
        "totalNodeCount": schema["nodeCount"]
    }
    return json.dumps(nodes_info, indent=2)


@mcp.resource("neo4j://schema/relationships")
def get_relationships_schema_resource() -> str:
    """
    Get relationship types and counts as a resource.

    Returns:
        JSON string of relationship type information
    """
    schema = tools.get_schema(use_cache=True)
    rels_info = {
        "relationshipTypes": schema["relationshipTypes"],
        "relationshipTypeCounts": schema["relationshipTypeCounts"],
        "totalRelationshipCount": schema["relationshipCount"]
    }
    return json.dumps(rels_info, indent=2)


@mcp.resource("neo4j://schema/constraints")
def get_constraints_resource() -> str:
    """
    Get graph constraints as a resource.

    Returns:
        JSON string of constraint information
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema["constraints"], indent=2, default=str)


@mcp.resource("neo4j://schema/indexes")
def get_indexes_resource() -> str:
    """
    Get graph indexes as a resource.

    Returns:
        JSON string of index information
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema["indexes"], indent=2, default=str)


@mcp.resource("neo4j://schema/visualization")
def get_visualization_resource() -> str:
    """
    Get graph visualization data as a resource.

    Returns label-to-label connections for rendering diagrams.

    Returns:
        JSON string of visualization data
    """
    schema = tools.get_schema(use_cache=True)
    return json.dumps(schema["graphVisualization"], indent=2)


@mcp.resource("neo4j://stats")
def get_stats_resource() -> str:
    """
    Get graph statistics as a resource.

    Includes node counts, relationship counts, and counts per label/type.

    Returns:
        JSON string of graph statistics
    """
    schema = tools.get_schema(use_cache=True)
    stats = {
        "nodeCount": schema["nodeCount"],
        "relationshipCount": schema["relationshipCount"],
        "labelCounts": schema["labelCounts"],
        "relationshipTypeCounts": schema["relationshipTypeCounts"]
    }
    return json.dumps(stats, indent=2)


def cleanup():
    """Clean up Neo4j connection."""
    tools.cleanup()


if __name__ == "__main__":
    # Run the MCP server
    try:
        # Disable banner to keep stdout clean for MCP JSON-RPC protocol
        mcp.run(show_banner=False)
    finally:
        cleanup()
