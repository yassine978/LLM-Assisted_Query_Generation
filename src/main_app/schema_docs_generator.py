"""Schema Documentation Generator - Creates documentation from database schemas.

This module generates human-readable documentation for MongoDB and Neo4j schemas
in Markdown and JSON formats.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MongoDBSchemaDocGenerator:
    """Generates documentation for MongoDB schemas."""

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize generator.

        Args:
            schema: MongoDB collection schema
        """
        self.schema = schema

    def generate_markdown(self) -> str:
        """
        Generate Markdown documentation for the schema.

        Returns:
            Markdown formatted documentation
        """
        try:
            collection = self.schema.get("collection", "unknown")
            database = self.schema.get("database", "unknown")
            doc_count = self.schema.get("document_count", 0)
            stats = self.schema.get("statistics", {})
            fields = self.schema.get("fields", {})
            indexes = self.schema.get("indexes", [])

            # Build markdown
            md_lines = []

            # Header
            md_lines.append(f"# MongoDB Collection: {database}.{collection}")
            md_lines.append("")
            md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_lines.append("")

            # Statistics
            md_lines.append("## Collection Statistics")
            md_lines.append("")
            md_lines.append(f"- **Documents**: {doc_count:,}")
            md_lines.append(f"- **Size**: {stats.get('size_bytes', 0):,} bytes ({self._format_bytes(stats.get('size_bytes', 0))})")
            md_lines.append(f"- **Storage Size**: {stats.get('storage_size_bytes', 0):,} bytes ({self._format_bytes(stats.get('storage_size_bytes', 0))})")
            md_lines.append(f"- **Average Document Size**: {stats.get('avg_document_size_bytes', 0):.0f} bytes")
            md_lines.append(f"- **Indexes**: {stats.get('index_count', 0)}")
            md_lines.append(f"- **Index Size**: {stats.get('total_index_size_bytes', 0):,} bytes ({self._format_bytes(stats.get('total_index_size_bytes', 0))})")
            md_lines.append(f"- **Capped**: {'Yes' if stats.get('capped') else 'No'}")
            md_lines.append("")

            # Fields
            md_lines.append("## Fields")
            md_lines.append("")
            md_lines.append("| Field | Type(s) | Frequency | Description |")
            md_lines.append("|-------|---------|-----------|-------------|")

            for field_name, field_info in sorted(fields.items()):
                types = ", ".join(field_info.get("types", []))
                frequency = f"{field_info.get('frequency', 0):.1f}%"
                example = self._format_example(field_info.get("example"))
                desc = f"Example: {example}" if example else "-"

                # Check for nested schema
                if "nested_schema" in field_info:
                    desc += " (nested document)"
                elif "array_info" in field_info:
                    array_info = field_info["array_info"]
                    element_types = ", ".join(array_info.get("element_types", []))
                    desc += f" (array of {element_types})"

                md_lines.append(f"| `{field_name}` | {types} | {frequency} | {desc} |")

            md_lines.append("")

            # Nested schemas (if any)
            has_nested = any("nested_schema" in f for f in fields.values())
            if has_nested:
                md_lines.append("## Nested Document Structures")
                md_lines.append("")
                for field_name, field_info in sorted(fields.items()):
                    if "nested_schema" in field_info:
                        md_lines.append(f"### {field_name}")
                        md_lines.append("")
                        md_lines.append("```json")
                        md_lines.append(json.dumps(field_info["nested_schema"], indent=2))
                        md_lines.append("```")
                        md_lines.append("")

            # Indexes
            if indexes:
                md_lines.append("## Indexes")
                md_lines.append("")
                md_lines.append("| Name | Keys | Unique | Sparse |")
                md_lines.append("|------|------|--------|--------|")

                for idx in indexes:
                    name = idx.get("name", "-")
                    keys = json.dumps(idx.get("keys", {}))
                    unique = "Yes" if idx.get("unique") else "No"
                    sparse = "Yes" if idx.get("sparse") else "No"
                    md_lines.append(f"| `{name}` | `{keys}` | {unique} | {sparse} |")

                md_lines.append("")

            markdown = "\n".join(md_lines)

            logger.info("Generated MongoDB schema documentation (markdown)", collection=collection)
            return markdown

        except Exception as e:
            logger.error("Error generating MongoDB markdown documentation", error=str(e))
            return f"# Error generating documentation\n\n{str(e)}"

    def generate_json(self) -> str:
        """
        Generate JSON documentation for the schema.

        Returns:
            JSON formatted documentation
        """
        try:
            doc = {
                "type": "mongodb",
                "database": self.schema.get("database"),
                "collection": self.schema.get("collection"),
                "generated_at": datetime.now().isoformat(),
                "statistics": self.schema.get("statistics", {}),
                "document_count": self.schema.get("document_count", 0),
                "fields": self.schema.get("fields", {}),
                "indexes": self.schema.get("indexes", []),
                "sample_documents": self.schema.get("sample_documents", [])[:3]  # Include 3 samples
            }

            logger.info("Generated MongoDB schema documentation (JSON)", collection=doc["collection"])
            return json.dumps(doc, indent=2)

        except Exception as e:
            logger.error("Error generating MongoDB JSON documentation", error=str(e))
            return json.dumps({"error": str(e)}, indent=2)

    @staticmethod
    def _format_bytes(bytes_count: int) -> str:
        """Format bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} TB"

    @staticmethod
    def _format_example(example: Any) -> str:
        """Format example value for display."""
        if example is None:
            return ""
        elif isinstance(example, str):
            return f'"{example[:50]}"' if len(example) > 50 else f'"{example}"'
        elif isinstance(example, (int, float, bool)):
            return str(example)
        elif isinstance(example, list):
            return f"[{len(example)} items]"
        elif isinstance(example, dict):
            return f"{{object}}"
        else:
            return str(example)[:50]


class Neo4jSchemaDocGenerator:
    """Generates documentation for Neo4j graph schemas."""

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize generator.

        Args:
            schema: Neo4j graph schema
        """
        self.schema = schema

    def generate_markdown(self) -> str:
        """
        Generate Markdown documentation for the graph schema.

        Returns:
            Markdown formatted documentation
        """
        try:
            labels = self.schema.get("labels", [])
            rel_types = self.schema.get("relationshipTypes", [])
            label_counts = self.schema.get("labelCounts", {})
            rel_type_counts = self.schema.get("relationshipTypeCounts", {})
            constraints = self.schema.get("constraints", [])
            indexes = self.schema.get("indexes", [])
            viz = self.schema.get("graphVisualization", {})

            # Build markdown
            md_lines = []

            # Header
            md_lines.append("# Neo4j Graph Schema")
            md_lines.append("")
            md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_lines.append("")

            # Statistics
            md_lines.append("## Graph Statistics")
            md_lines.append("")
            md_lines.append(f"- **Total Nodes**: {self.schema.get('nodeCount', 0):,}")
            md_lines.append(f"- **Total Relationships**: {self.schema.get('relationshipCount', 0):,}")
            md_lines.append(f"- **Node Labels**: {len(labels)}")
            md_lines.append(f"- **Relationship Types**: {len(rel_types)}")
            md_lines.append(f"- **Constraints**: {len(constraints)}")
            md_lines.append(f"- **Indexes**: {len(indexes)}")
            md_lines.append("")

            # Node Labels
            md_lines.append("## Node Labels")
            md_lines.append("")
            md_lines.append("| Label | Count | Percentage |")
            md_lines.append("|-------|-------|------------|")

            total_nodes = self.schema.get('nodeCount', 1)  # Avoid division by zero
            for label in sorted(labels):
                count = label_counts.get(label, 0)
                percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
                md_lines.append(f"| `{label}` | {count:,} | {percentage:.1f}% |")

            md_lines.append("")

            # Relationship Types
            md_lines.append("## Relationship Types")
            md_lines.append("")
            md_lines.append("| Type | Count | Percentage |")
            md_lines.append("|------|-------|------------|")

            total_rels = self.schema.get('relationshipCount', 1)  # Avoid division by zero
            for rel_type in sorted(rel_types):
                count = rel_type_counts.get(rel_type, 0)
                percentage = (count / total_rels * 100) if total_rels > 0 else 0
                md_lines.append(f"| `{rel_type}` | {count:,} | {percentage:.1f}% |")

            md_lines.append("")

            # Graph Structure (Visualization)
            if viz.get("edges"):
                md_lines.append("## Graph Structure")
                md_lines.append("")
                md_lines.append("```mermaid")
                md_lines.append("graph LR")

                # Generate Mermaid diagram
                for edge in viz.get("edges", []):
                    source = edge.get("from", "?")
                    target = edge.get("to", "?")
                    rel_type = edge.get("type", "?")
                    md_lines.append(f"    {source} -->|{rel_type}| {target}")

                md_lines.append("```")
                md_lines.append("")

            # Constraints
            if constraints:
                md_lines.append("## Constraints")
                md_lines.append("")
                md_lines.append("| Name | Type | Entity | Properties |")
                md_lines.append("|------|------|--------|------------|")

                for constraint in constraints:
                    name = constraint.get("name", "-")
                    ctype = constraint.get("type", "-")
                    entity_list = constraint.get("labelsOrTypes", [])
                    entity = ", ".join(entity_list) if isinstance(entity_list, list) else str(entity_list)
                    props_list = constraint.get("properties", [])
                    props = ", ".join(props_list) if isinstance(props_list, list) else str(props_list)
                    md_lines.append(f"| `{name}` | {ctype} | `{entity}` | `{props}` |")

                md_lines.append("")

            # Indexes
            if indexes:
                md_lines.append("## Indexes")
                md_lines.append("")
                md_lines.append("| Name | State | Type | Entity |")
                md_lines.append("|------|-------|------|--------|")

                for index in indexes[:20]:  # Limit to first 20
                    name = index.get("name", "-")
                    state = index.get("state", "-")
                    itype = index.get("type", "-")
                    entity_list = index.get("labelsOrTypes", [])
                    entity = ", ".join(entity_list) if isinstance(entity_list, list) else str(entity_list)
                    md_lines.append(f"| `{name}` | {state} | {itype} | `{entity}` |")

                if len(indexes) > 20:
                    md_lines.append(f"| ... | ... | ... | ... |")
                    md_lines.append(f"| *(Total: {len(indexes)} indexes)* | | | |")

                md_lines.append("")

            markdown = "\n".join(md_lines)

            logger.info("Generated Neo4j schema documentation (markdown)")
            return markdown

        except Exception as e:
            logger.error("Error generating Neo4j markdown documentation", error=str(e))
            return f"# Error generating documentation\n\n{str(e)}"

    def generate_json(self) -> str:
        """
        Generate JSON documentation for the graph schema.

        Returns:
            JSON formatted documentation
        """
        try:
            doc = {
                "type": "neo4j",
                "generated_at": datetime.now().isoformat(),
                "statistics": {
                    "nodeCount": self.schema.get("nodeCount", 0),
                    "relationshipCount": self.schema.get("relationshipCount", 0),
                    "labelCounts": self.schema.get("labelCounts", {}),
                    "relationshipTypeCounts": self.schema.get("relationshipTypeCounts", {})
                },
                "labels": self.schema.get("labels", []),
                "relationshipTypes": self.schema.get("relationshipTypes", []),
                "propertyKeys": self.schema.get("propertyKeys", []),
                "constraints": self.schema.get("constraints", []),
                "indexes": self.schema.get("indexes", []),
                "graphVisualization": self.schema.get("graphVisualization", {})
            }

            logger.info("Generated Neo4j schema documentation (JSON)")
            return json.dumps(doc, indent=2, default=str)

        except Exception as e:
            logger.error("Error generating Neo4j JSON documentation", error=str(e))
            return json.dumps({"error": str(e)}, indent=2)


def generate_mongodb_docs(schema: Dict[str, Any], format: str = "markdown") -> str:
    """
    Generate documentation for MongoDB schema.

    Args:
        schema: MongoDB collection schema
        format: Output format ("markdown" or "json")

    Returns:
        Documentation string
    """
    try:
        generator = MongoDBSchemaDocGenerator(schema)
        if format.lower() == "json":
            return generator.generate_json()
        else:
            return generator.generate_markdown()
    except Exception as e:
        logger.error("Error generating MongoDB documentation", error=str(e))
        return f"Error: {str(e)}"


def generate_neo4j_docs(schema: Dict[str, Any], format: str = "markdown") -> str:
    """
    Generate documentation for Neo4j schema.

    Args:
        schema: Neo4j graph schema
        format: Output format ("markdown" or "json")

    Returns:
        Documentation string
    """
    try:
        generator = Neo4jSchemaDocGenerator(schema)
        if format.lower() == "json":
            return generator.generate_json()
        else:
            return generator.generate_markdown()
    except Exception as e:
        logger.error("Error generating Neo4j documentation", error=str(e))
        return f"Error: {str(e)}"
