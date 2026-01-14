"""Neo4j tools - Core functionality without MCP framework dependency.

This module provides Neo4j operations that can be used standalone or wrapped by MCP servers.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, exceptions as neo4j_exceptions

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global Neo4j driver (initialized on first use)
_neo4j_driver: Optional[GraphDatabase.driver] = None

# Neo4j connection settings (can be overridden by config)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# Schema cache with TTL (5 minutes)
_schema_cache: Dict[str, tuple[float, Any]] = {}
CACHE_TTL = 300  # 5 minutes in seconds


def get_neo4j_driver() -> GraphDatabase.driver:
    """
    Get or create Neo4j driver connection.

    Returns:
        GraphDatabase.driver: Neo4j driver instance

    Raises:
        Exception: If connection to Neo4j fails
    """
    global _neo4j_driver

    if _neo4j_driver is None:
        logger.info("Connecting to Neo4j", uri=NEO4J_URI)

        try:
            _neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Verify connection
            _neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise

    return _neo4j_driver


def get_node_labels() -> List[str]:
    """
    List all node labels in the Neo4j database.

    Returns:
        List[str]: List of node label names
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            logger.info("Listed node labels", count=len(labels))
            return labels
    except Exception as e:
        logger.error("Error listing node labels", error=str(e))
        raise


def get_relationship_types() -> List[str]:
    """
    List all relationship types in the Neo4j database.

    Returns:
        List[str]: List of relationship type names
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("CALL db.relationshipTypes()")
            types = [record["relationshipType"] for record in result]
            logger.info("Listed relationship types", count=len(types))
            return types
    except Exception as e:
        logger.error("Error listing relationship types", error=str(e))
        raise


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
        Dict containing comprehensive schema information including:
        - labels: List of node labels
        - relationshipTypes: List of relationship types
        - propertyKeys: List of all property keys
        - constraints: List of constraints
        - indexes: List of indexes
        - nodeCount: Total node count
        - relationshipCount: Total relationship count
        - labelCounts: Dict of per-label node counts
        - relationshipTypeCounts: Dict of per-type relationship counts
        - graphVisualization: Graph structure for visualization
    """
    try:
        # Check cache
        cache_key = "neo4j_schema"
        if use_cache and cache_key in _schema_cache:
            timestamp, cached_schema = _schema_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                logger.info(
                    "Returning cached schema",
                    age_seconds=int(time.time() - timestamp)
                )
                return cached_schema

        driver = get_neo4j_driver()
        with driver.session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            # Get relationship types
            rel_result = session.run("CALL db.relationshipTypes()")
            relationship_types = [record["relationshipType"] for record in rel_result]

            # Get property keys
            prop_result = session.run("CALL db.propertyKeys()")
            property_keys = [record["propertyKey"] for record in prop_result]

            # Get constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = []
            for record in constraints_result:
                constraint_dict = {}
                for key, value in dict(record).items():
                    # Convert DateTime objects to ISO format strings
                    if hasattr(value, 'iso_format'):
                        constraint_dict[key] = value.iso_format()
                    else:
                        constraint_dict[key] = value
                constraints.append(constraint_dict)

            # Get indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = []
            for record in indexes_result:
                index_dict = {}
                for key, value in dict(record).items():
                    # Convert DateTime objects to ISO format strings
                    if hasattr(value, 'iso_format'):
                        index_dict[key] = value.iso_format()
                    else:
                        index_dict[key] = value
                indexes.append(index_dict)

            # Get total counts
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            # Get per-label node counts
            label_counts = {}
            for label in labels:
                count_query = f"MATCH (n:`{label}`) RETURN count(n) as count"
                label_counts[label] = session.run(count_query).single()["count"]

            # Get per-type relationship counts
            rel_type_counts = {}
            for rel_type in relationship_types:
                count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                rel_type_counts[rel_type] = session.run(count_query).single()["count"]

            # Get graph visualization data (label-to-label connections)
            viz_query = """
            MATCH (source)-[r]->(target)
            WITH DISTINCT labels(source)[0] as sourceLabel, type(r) as relType, labels(target)[0] as targetLabel
            RETURN sourceLabel, relType, targetLabel
            LIMIT 1000
            """
            viz_result = session.run(viz_query)
            edges = []
            node_labels_set = set()

            for record in viz_result:
                source_label = record["sourceLabel"]
                rel_type = record["relType"]
                target_label = record["targetLabel"]

                if source_label and target_label and rel_type:
                    node_labels_set.add(source_label)
                    node_labels_set.add(target_label)
                    edges.append({
                        "from": source_label,
                        "to": target_label,
                        "type": rel_type
                    })

            graph_visualization = {
                "nodes": sorted(list(node_labels_set)),
                "edges": edges
            }

            schema = {
                "labels": labels,
                "relationshipTypes": relationship_types,
                "propertyKeys": property_keys,
                "constraints": constraints,
                "indexes": indexes,
                "nodeCount": node_count,
                "relationshipCount": rel_count,
                "labelCounts": label_counts,
                "relationshipTypeCounts": rel_type_counts,
                "graphVisualization": graph_visualization
            }

            # Cache the schema
            _schema_cache[cache_key] = (time.time(), schema)

            logger.info(
                "Generated comprehensive schema",
                labels=len(labels),
                relationships=len(relationship_types),
                properties=len(property_keys)
            )
            return schema

    except Exception as e:
        logger.error("Error getting schema", error=str(e))
        raise


def get_node_properties(label: str) -> Dict[str, Any]:
    """
    Get properties for a specific node label.

    Args:
        label: Node label name

    Returns:
        Dict containing property information
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Sample nodes to get properties
            query = f"MATCH (n:`{label}`) RETURN n LIMIT 100"
            result = session.run(query)

            properties = {}
            count = 0
            for record in result:
                count += 1
                node = record["n"]
                for prop, value in node.items():
                    if prop not in properties:
                        properties[prop] = {
                            "types": set(),
                            "examples": []
                        }
                    properties[prop]["types"].add(type(value).__name__)
                    if len(properties[prop]["examples"]) < 3:
                        properties[prop]["examples"].append(value)

            # Convert sets to lists for JSON serialization
            for prop in properties:
                properties[prop]["types"] = list(properties[prop]["types"])

            result_data = {
                "label": label,
                "properties": properties,
                "sampleSize": count
            }

            logger.info(
                "Retrieved node properties",
                label=label,
                property_count=len(properties)
            )
            return result_data

    except Exception as e:
        logger.error("Error getting node properties", label=label, error=str(e))
        raise


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
        driver = get_neo4j_driver()
        with driver.session() as session:
            query = f"MATCH (n:`{label}`) RETURN n LIMIT {limit}"
            result = session.run(query)

            samples = []
            for record in result:
                node = record["n"]
                # Convert Neo4j node to dict
                node_dict = dict(node)
                # Add node labels
                node_dict["_labels"] = list(node.labels)
                samples.append(node_dict)

            result_data = {
                "label": label,
                "sampleCount": len(samples),
                "samples": samples
            }

            logger.info(
                "Retrieved sample nodes",
                label=label,
                count=len(samples)
            )
            return result_data

    except Exception as e:
        logger.error("Error getting sample nodes", label=label, error=str(e))
        raise


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
        driver = get_neo4j_driver()
        with driver.session() as session:
            query = f"""
            MATCH (source)-[r:`{rel_type}`]->(target)
            RETURN source, r, target
            LIMIT {limit}
            """
            result = session.run(query)

            samples = []
            for record in result:
                source_node = record["source"]
                rel = record["r"]
                target_node = record["target"]

                sample = {
                    "source": {
                        **dict(source_node),
                        "_labels": list(source_node.labels)
                    },
                    "relationship": {
                        **dict(rel),
                        "_type": rel.type
                    },
                    "target": {
                        **dict(target_node),
                        "_labels": list(target_node.labels)
                    }
                }
                samples.append(sample)

            result_data = {
                "relationshipType": rel_type,
                "sampleCount": len(samples),
                "samples": samples
            }

            logger.info(
                "Retrieved sample relationships",
                rel_type=rel_type,
                count=len(samples)
            )
            return result_data

    except Exception as e:
        logger.error("Error getting sample relationships", rel_type=rel_type, error=str(e))
        raise


def get_relationship_properties(rel_type: str) -> Dict[str, Any]:
    """
    Get properties for a specific relationship type.

    Args:
        rel_type: Relationship type name

    Returns:
        Dict containing property information
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Sample relationships to get properties
            query = f"MATCH ()-[r:`{rel_type}`]->() RETURN r LIMIT 100"
            result = session.run(query)

            properties = {}
            count = 0
            for record in result:
                count += 1
                rel = record["r"]
                for prop, value in rel.items():
                    if prop not in properties:
                        properties[prop] = {
                            "types": set(),
                            "examples": []
                        }
                    properties[prop]["types"].add(type(value).__name__)
                    if len(properties[prop]["examples"]) < 3:
                        properties[prop]["examples"].append(value)

            # Convert sets to lists for JSON serialization
            for prop in properties:
                properties[prop]["types"] = list(properties[prop]["types"])

            result_data = {
                "relationshipType": rel_type,
                "properties": properties,
                "sampleSize": count
            }

            logger.info(
                "Retrieved relationship properties",
                rel_type=rel_type,
                property_count=len(properties)
            )
            return result_data

    except Exception as e:
        logger.error("Error getting relationship properties", rel_type=rel_type, error=str(e))
        raise


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
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Add LIMIT if not present and query is a read operation
            if "LIMIT" not in query.upper() and any(keyword in query.upper() for keyword in ["MATCH", "RETURN"]):
                if "RETURN" in query.upper():
                    query = query.rstrip(";") + f" LIMIT {limit}"

            result = session.run(query)

            # Convert results to list of dicts
            records = []
            for record in result:
                # Convert Record to dict
                record_dict = dict(record)
                # Handle Neo4j node/relationship objects
                for key, value in record_dict.items():
                    if hasattr(value, '_properties'):  # Node or Relationship
                        record_dict[key] = dict(value)
                    elif isinstance(value, list):
                        # Handle lists of nodes/relationships
                        record_dict[key] = [
                            dict(item) if hasattr(item, '_properties') else item
                            for item in value
                        ]
                records.append(record_dict)

            response = {
                "query": query,
                "result_count": len(records),
                "results": records
            }

            logger.info(
                "Executed Cypher query",
                result_count=len(records)
            )
            return response

    except neo4j_exceptions.CypherSyntaxError as e:
        logger.error("Cypher syntax error", error=str(e))
        raise ValueError(f"Cypher syntax error: {e}")
    except Exception as e:
        logger.error("Error executing Cypher query", error=str(e))
        raise


def explain_cypher(query: str) -> Dict[str, Any]:
    """
    Get execution plan for a Cypher query without executing it.

    Args:
        query: Cypher query string

    Returns:
        Dict containing query execution plan
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(f"EXPLAIN {query}")

            # Get the plan
            summary = result.consume()
            plan = summary.plan

            # Extract plan information
            plan_info = {
                "query": query,
                "operatorType": plan.operator_type if plan else None,
                "estimatedRows": plan.arguments.get("EstimatedRows") if plan and plan.arguments else None,
                "identifiers": plan.identifiers if plan else None,
                "children": len(plan.children) if plan and plan.children else 0
            }

            logger.info("Explained Cypher query")
            return plan_info

    except Exception as e:
        logger.error("Error explaining Cypher query", error=str(e))
        raise


def validate_cypher(query: str) -> Dict[str, Any]:
    """
    Validate a Cypher query without executing it.

    Args:
        query: Cypher query string

    Returns:
        Dict containing validation results
    """
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Use EXPLAIN to validate syntax
            try:
                result = session.run(f"EXPLAIN {query}")
                result.consume()  # Consume to check for errors

                logger.info("Cypher query validated successfully")
                return {
                    "valid": True,
                    "query": query,
                    "message": "Query is valid"
                }
            except neo4j_exceptions.CypherSyntaxError as e:
                return {
                    "valid": False,
                    "error": str(e),
                    "errorType": "CypherSyntaxError"
                }

    except Exception as e:
        logger.error("Error validating Cypher query", error=str(e))
        return {
            "valid": False,
            "error": str(e),
            "errorType": "ValidationError"
        }


def clear_schema_cache():
    """Clear the Neo4j schema cache."""
    global _schema_cache
    count = len(_schema_cache)
    _schema_cache.clear()
    logger.info(f"Cleared Neo4j schema cache ({count} entries)")


def cleanup():
    """Clean up Neo4j connection and clear cache."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Neo4j connection closed")

    # Clear schema cache on cleanup
    clear_schema_cache()
