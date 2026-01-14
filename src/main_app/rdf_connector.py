"""RDF/SPARQL Connector - Direct connection to Fuseki (No MCP).

This module provides direct SPARQL query execution against Apache Jena Fuseki
without using the MCP protocol.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global connection
_sparql_wrapper: Optional[SPARQLWrapper] = None


def get_sparql_connection() -> SPARQLWrapper:
    """Get or create SPARQL connection to Fuseki.

    Returns:
        SPARQLWrapper: Configured SPARQL client

    Raises:
        Exception: If connection fails
    """
    global _sparql_wrapper

    if _sparql_wrapper is None:
        settings = get_settings()
        endpoint = settings.get_sparql_endpoint()

        logger.info("Connecting to SPARQL endpoint", endpoint=endpoint)

        try:
            _sparql_wrapper = SPARQLWrapper(endpoint)
            _sparql_wrapper.setReturnFormat(JSON)
            _sparql_wrapper.setMethod(POST)

            # Add authentication if configured
            if settings.fuseki_username and settings.fuseki_password:
                _sparql_wrapper.setCredentials(
                    settings.fuseki_username,
                    settings.fuseki_password
                )

            logger.info("SPARQL connection successful")

        except Exception as e:
            logger.error("SPARQL connection failed", error=str(e))
            _sparql_wrapper = None
            raise

    return _sparql_wrapper


def execute_sparql(query: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute a SPARQL query.

    Args:
        query: SPARQL query string
        timeout: Query timeout in seconds

    Returns:
        Dict with query results in SPARQL JSON format

    Raises:
        Exception: If query execution fails
    """
    try:
        sparql = get_sparql_connection()
        sparql.setTimeout(timeout)
        sparql.setQuery(query)

        logger.info("Executing SPARQL query", query_preview=query[:100])

        results = sparql.query().convert()

        logger.info("SPARQL query executed successfully")

        return results

    except Exception as e:
        logger.error("SPARQL query failed", error=str(e), query=query[:200])
        raise


class RDFConnector:
    """RDF/SPARQL Connector for direct Fuseki access."""

    def __init__(self):
        """Initialize RDF connector."""
        self.settings = get_settings()
        self.endpoint = self.settings.get_sparql_endpoint()

    def test_connection(self) -> bool:
        """Test SPARQL endpoint connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple ASK query to test connection
            query = "ASK { ?s ?p ?o }"
            execute_sparql(query, timeout=5)
            return True
        except Exception as e:
            logger.error("Connection test failed", error=str(e))
            return False

    def count_triples(self) -> int:
        """Count total triples in the dataset.

        Returns:
            Number of triples
        """
        try:
            query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                return int(bindings[0]["count"]["value"])
            return 0
        except Exception as e:
            logger.error("Failed to count triples", error=str(e))
            return 0

    def get_namespaces(self) -> List[str]:
        """Get all unique namespaces used in the dataset.

        Returns:
            List of namespace URIs
        """
        try:
            query = """
            SELECT DISTINCT ?ns WHERE {
                {
                    SELECT DISTINCT (REPLACE(STR(?s), "([^/#]*[/#])[^/#]*$", "$1") as ?ns)
                    WHERE { ?s ?p ?o }
                }
                UNION
                {
                    SELECT DISTINCT (REPLACE(STR(?p), "([^/#]*[/#])[^/#]*$", "$1") as ?ns)
                    WHERE { ?s ?p ?o }
                }
            }
            LIMIT 50
            """
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])
            return [b["ns"]["value"] for b in bindings if "ns" in b]
        except Exception as e:
            logger.error("Failed to get namespaces", error=str(e))
            return []

    def get_classes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all RDF classes with counts.

        Args:
            limit: Maximum number of classes to return

        Returns:
            List of dicts with class URI and instance count
        """
        try:
            query = f"""
            SELECT ?class (COUNT(DISTINCT ?instance) as ?count) WHERE {{
                ?instance a ?class
            }}
            GROUP BY ?class
            ORDER BY DESC(?count)
            LIMIT {limit}
            """
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            classes = []
            for b in bindings:
                classes.append({
                    "class": b.get("class", {}).get("value", ""),
                    "count": int(b.get("count", {}).get("value", 0))
                })
            return classes
        except Exception as e:
            logger.error("Failed to get classes", error=str(e))
            return []

    def get_properties(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all RDF properties with usage counts.

        Args:
            limit: Maximum number of properties to return

        Returns:
            List of dicts with property URI and usage count
        """
        try:
            query = f"""
            SELECT ?property (COUNT(*) as ?count) WHERE {{
                ?s ?property ?o
            }}
            GROUP BY ?property
            ORDER BY DESC(?count)
            LIMIT {limit}
            """
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            properties = []
            for b in bindings:
                properties.append({
                    "property": b.get("property", {}).get("value", ""),
                    "count": int(b.get("count", {}).get("value", 0))
                })
            return properties
        except Exception as e:
            logger.error("Failed to get properties", error=str(e))
            return []

    def get_class_instances(self, class_uri: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get instances of a specific RDF class.

        Args:
            class_uri: URI of the class
            limit: Maximum number of instances

        Returns:
            List of instance URIs with labels
        """
        try:
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?instance ?label WHERE {{
                ?instance a <{class_uri}> .
                OPTIONAL {{ ?instance rdfs:label ?label }}
            }}
            LIMIT {limit}
            """
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            instances = []
            for b in bindings:
                instances.append({
                    "instance": b.get("instance", {}).get("value", ""),
                    "label": b.get("label", {}).get("value", "")
                })
            return instances
        except Exception as e:
            logger.error("Failed to get class instances", error=str(e))
            return []

    def sample_triples(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get sample triples from the dataset.

        Args:
            limit: Number of triples to return

        Returns:
            List of dicts with subject, predicate, object
        """
        try:
            query = f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o
            }}
            LIMIT {limit}
            """
            results = execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            triples = []
            for b in bindings:
                # Handle both string and dict formats
                subject = b.get("s", {})
                predicate = b.get("p", {})
                obj = b.get("o", {})

                triples.append({
                    "subject": subject if isinstance(subject, dict) else {"value": subject},
                    "predicate": predicate if isinstance(predicate, dict) else {"value": predicate},
                    "object": obj if isinstance(obj, dict) else {"value": obj}
                })
            return triples
        except Exception as e:
            logger.error("Failed to get sample triples", error=str(e))
            return []
