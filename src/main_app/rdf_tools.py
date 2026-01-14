"""RDF Tools - High-level RDF/SPARQL operations (No MCP).

This module provides RDF/SPARQL tools as regular Python functions
for direct integration with the query engine.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main_app.rdf_connector import RDFConnector, execute_sparql
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ping() -> Dict[str, Any]:
    """Test RDF/SPARQL endpoint connection.

    Returns:
        Dict with connection status
    """
    try:
        connector = RDFConnector()
        success = connector.test_connection()

        if success:
            triple_count = connector.count_triples()
            return {
                "success": True,
                "message": "SPARQL endpoint connection successful",
                "triple_count": triple_count
            }
        else:
            return {
                "success": False,
                "error": "Connection test failed"
            }
    except Exception as e:
        logger.error("RDF ping failed", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def run_sparql(query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a SPARQL query.

    Args:
        query: SPARQL query string
        limit: Maximum results (not enforced if query has its own LIMIT)

    Returns:
        Dict with query results
    """
    try:
        results = execute_sparql(query, timeout=30)

        # Extract bindings
        bindings = results.get("results", {}).get("bindings", [])

        return {
            "success": True,
            "results": bindings,
            "result_count": len(bindings),
            "variables": results.get("head", {}).get("vars", [])
        }
    except Exception as e:
        logger.error("SPARQL query failed", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def validate_sparql(query: str) -> Dict[str, Any]:
    """Validate SPARQL query syntax.

    Args:
        query: SPARQL query string

    Returns:
        Dict with validation results
    """
    try:
        # Basic syntax validation
        # Check for required keywords
        query_upper = query.upper()

        if "SELECT" not in query_upper and "ASK" not in query_upper and \
           "CONSTRUCT" not in query_upper and "DESCRIBE" not in query_upper:
            return {
                "valid": False,
                "error": "Query must start with SELECT, ASK, CONSTRUCT, or DESCRIBE"
            }

        if "WHERE" not in query_upper and "SELECT" in query_upper:
            return {
                "valid": False,
                "error": "SELECT queries must have a WHERE clause"
            }

        # Try to execute with LIMIT 0 to check syntax
        test_query = query
        if "LIMIT" not in query_upper:
            test_query += " LIMIT 0"

        try:
            execute_sparql(test_query, timeout=5)
            return {
                "valid": True,
                "message": "Query syntax is valid"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Syntax error: {str(e)}"
            }

    except Exception as e:
        logger.error("SPARQL validation failed", error=str(e))
        return {
            "valid": False,
            "error": str(e)
        }


def get_ontology() -> Dict[str, Any]:
    """Get RDF ontology (classes and properties).

    Returns:
        Dict with classes and properties
    """
    try:
        connector = RDFConnector()

        classes = connector.get_classes(limit=50)
        properties = connector.get_properties(limit=100)
        namespaces = connector.get_namespaces()

        return {
            "success": True,
            "classes": classes,
            "properties": properties,
            "namespaces": namespaces,
            "class_count": len(classes),
            "property_count": len(properties)
        }
    except Exception as e:
        logger.error("Failed to get ontology", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_schema() -> Dict[str, Any]:
    """Get complete RDF schema information.

    Returns:
        Dict with schema details
    """
    try:
        connector = RDFConnector()

        # Get statistics
        triple_count = connector.count_triples()

        # Get classes and properties
        classes = connector.get_classes(limit=50)
        properties = connector.get_properties(limit=100)

        # Get namespaces
        namespaces = connector.get_namespaces()

        # Get sample data
        samples = connector.sample_triples(limit=5)

        return {
            "success": True,
            "total_triples": triple_count,
            "classes": classes,
            "properties": properties,
            "namespaces": namespaces,
            "samples": samples,
            "class_count": len(classes),
            "property_count": len(properties)
        }
    except Exception as e:
        logger.error("Failed to get schema", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_class_instances(class_uri: str, limit: int = 10) -> Dict[str, Any]:
    """Get instances of a specific RDF class.

    Args:
        class_uri: URI of the class
        limit: Maximum number of instances

    Returns:
        Dict with instances
    """
    try:
        connector = RDFConnector()
        instances = connector.get_class_instances(class_uri, limit)

        return {
            "success": True,
            "class_uri": class_uri,
            "instances": instances,
            "count": len(instances)
        }
    except Exception as e:
        logger.error("Failed to get class instances", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_property_usage() -> Dict[str, Any]:
    """Get statistics on property usage.

    Returns:
        Dict with property usage statistics
    """
    try:
        connector = RDFConnector()
        properties = connector.get_properties(limit=100)

        # Calculate statistics
        total_usage = sum(p["count"] for p in properties)

        return {
            "success": True,
            "properties": properties,
            "total_properties": len(properties),
            "total_usage": total_usage
        }
    except Exception as e:
        logger.error("Failed to get property usage", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_namespaces() -> Dict[str, Any]:
    """Get all namespaces used in the dataset.

    Returns:
        Dict with namespaces
    """
    try:
        connector = RDFConnector()
        namespaces = connector.get_namespaces()

        return {
            "success": True,
            "namespaces": namespaces,
            "count": len(namespaces)
        }
    except Exception as e:
        logger.error("Failed to get namespaces", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def sample_triples(limit: int = 10) -> Dict[str, Any]:
    """Get sample triples from the dataset.

    Args:
        limit: Number of triples to return

    Returns:
        Dict with sample triples
    """
    try:
        connector = RDFConnector()
        triples = connector.sample_triples(limit)

        return {
            "success": True,
            "triples": triples,
            "count": len(triples)
        }
    except Exception as e:
        logger.error("Failed to get sample triples", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


def get_graph_statistics() -> Dict[str, Any]:
    """Get overall graph statistics.

    Returns:
        Dict with graph statistics
    """
    try:
        connector = RDFConnector()

        triple_count = connector.count_triples()
        classes = connector.get_classes(limit=100)
        properties = connector.get_properties(limit=100)

        # Count entities (subjects that are instances of classes)
        entity_query = """
        SELECT (COUNT(DISTINCT ?entity) as ?count) WHERE {
            ?entity a ?class
        }
        """
        entity_result = execute_sparql(entity_query)
        entity_count = 0
        if entity_result.get("results", {}).get("bindings"):
            entity_count = int(
                entity_result["results"]["bindings"][0]["count"]["value"]
            )

        return {
            "success": True,
            "total_triples": triple_count,
            "total_entities": entity_count,
            "total_classes": len(classes),
            "total_properties": len(properties)
        }
    except Exception as e:
        logger.error("Failed to get graph statistics", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }
