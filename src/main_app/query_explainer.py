"""Query Explanation Generator - Creates natural language explanations of database queries.

This module generates human-readable explanations of MongoDB and Neo4j queries
to help users understand what the generated queries do.
"""

import json
import re
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MongoDBQueryExplainer:
    """Generates natural language explanations for MongoDB queries."""

    # Common MongoDB operators and their explanations
    OPERATORS = {
        "$gt": "greater than",
        "$gte": "greater than or equal to",
        "$lt": "less than",
        "$lte": "less than or equal to",
        "$eq": "equal to",
        "$ne": "not equal to",
        "$in": "in the list",
        "$nin": "not in the list",
        "$and": "and",
        "$or": "or",
        "$not": "not",
        "$exists": "exists",
        "$type": "has type",
        "$regex": "matches pattern",
        "$all": "contains all",
        "$elemMatch": "has an element matching",
        "$size": "has size"
    }

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize explainer.

        Args:
            schema: Optional schema for field type information
        """
        self.schema = schema

    def explain_query(self, query_dict: Dict[str, Any], collection: str) -> str:
        """
        Generate natural language explanation of MongoDB query.

        Args:
            query_dict: Parsed MongoDB query dictionary
            collection: Collection name

        Returns:
            Natural language explanation string
        """
        try:
            if not query_dict or query_dict == {}:
                return f"Find all documents in the '{collection}' collection."

            conditions = self._explain_conditions(query_dict)

            if len(conditions) == 1:
                explanation = f"Find documents in '{collection}' where {conditions[0]}."
            else:
                explanation = f"Find documents in '{collection}' where:\n"
                for i, condition in enumerate(conditions, 1):
                    explanation += f"  {i}. {condition}\n"
                explanation = explanation.rstrip()

            logger.info("Generated MongoDB query explanation")
            return explanation

        except Exception as e:
            logger.error("Error explaining MongoDB query", error=str(e))
            return f"Query on '{collection}' collection. (Unable to generate detailed explanation)"

    def _explain_conditions(self, obj: Any, field_path: str = "") -> List[str]:
        """
        Recursively explain query conditions.

        Args:
            obj: Object to explain
            field_path: Current field path

        Returns:
            List of explanation strings
        """
        conditions = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("$"):
                    # This is an operator
                    operator_explanation = self._explain_operator(key, value, field_path)
                    if operator_explanation:
                        conditions.append(operator_explanation)
                else:
                    # This is a field name
                    current_path = f"{field_path}.{key}" if field_path else key

                    if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                        # Field with operators
                        sub_conditions = self._explain_conditions(value, current_path)
                        conditions.extend(sub_conditions)
                    else:
                        # Simple equality
                        formatted_value = self._format_value(value)
                        conditions.append(f"'{key}' is {formatted_value}")

        return conditions

    def _explain_operator(self, operator: str, value: Any, field_path: str) -> Optional[str]:
        """
        Explain a MongoDB operator.

        Args:
            operator: Operator name (e.g., "$gt")
            value: Operator value
            field_path: Field path

        Returns:
            Explanation string or None
        """
        op_text = self.OPERATORS.get(operator, operator)

        if operator in ["$gt", "$gte", "$lt", "$lte", "$eq", "$ne"]:
            formatted_value = self._format_value(value)
            return f"'{field_path}' is {op_text} {formatted_value}"

        elif operator in ["$in", "$nin"]:
            values = ", ".join([self._format_value(v) for v in value])
            return f"'{field_path}' is {op_text} [{values}]"

        elif operator == "$regex":
            return f"'{field_path}' matches the pattern '{value}'"

        elif operator == "$exists":
            exists_text = "exists" if value else "does not exist"
            return f"'{field_path}' {exists_text}"

        elif operator == "$and":
            sub_explanations = []
            for condition in value:
                sub_explanations.extend(self._explain_conditions(condition))
            return "(" + " AND ".join(sub_explanations) + ")"

        elif operator == "$or":
            sub_explanations = []
            for condition in value:
                sub_explanations.extend(self._explain_conditions(condition))
            return "(" + " OR ".join(sub_explanations) + ")"

        elif operator == "$not":
            sub_explanations = self._explain_conditions(value, field_path)
            if sub_explanations:
                return f"NOT ({sub_explanations[0]})"

        elif operator == "$elemMatch":
            sub_explanations = self._explain_conditions(value)
            return f"'{field_path}' has an element where {' AND '.join(sub_explanations)}"

        elif operator == "$size":
            return f"'{field_path}' has {value} elements"

        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a value for display in explanation."""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, list):
            return f"[{', '.join([MongoDBQueryExplainer._format_value(v) for v in value])}]"
        elif value is None:
            return "null"
        else:
            return str(value)


class CypherQueryExplainer:
    """Generates natural language explanations for Cypher queries."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize explainer.

        Args:
            schema: Optional schema for graph structure information
        """
        self.schema = schema

    def explain_query(self, cypher_query: str) -> str:
        """
        Generate natural language explanation of Cypher query.

        Args:
            cypher_query: Cypher query string

        Returns:
            Natural language explanation string
        """
        try:
            # Clean up query
            query = cypher_query.strip()

            # Extract main clauses
            match_clause = self._extract_clause(query, "MATCH")
            where_clause = self._extract_clause(query, "WHERE")
            return_clause = self._extract_clause(query, "RETURN")
            create_clause = self._extract_clause(query, "CREATE")
            merge_clause = self._extract_clause(query, "MERGE")
            set_clause = self._extract_clause(query, "SET")
            delete_clause = self._extract_clause(query, "DELETE")

            # Build explanation based on query type
            explanation_parts = []

            # Determine query type
            if create_clause:
                explanation_parts.append("Create:")
                explanation_parts.append(self._explain_pattern(create_clause))
            elif merge_clause:
                explanation_parts.append("Create or match:")
                explanation_parts.append(self._explain_pattern(merge_clause))
            elif delete_clause:
                explanation_parts.append("Delete:")
                if match_clause:
                    explanation_parts.append(f"  - Find: {self._explain_pattern(match_clause)}")
                if where_clause:
                    explanation_parts.append(f"  - Where: {self._explain_where(where_clause)}")
                explanation_parts.append(f"  - Delete: {delete_clause}")
            elif set_clause:
                explanation_parts.append("Update:")
                if match_clause:
                    explanation_parts.append(f"  - Find: {self._explain_pattern(match_clause)}")
                if where_clause:
                    explanation_parts.append(f"  - Where: {self._explain_where(where_clause)}")
                explanation_parts.append(f"  - Set: {self._explain_set(set_clause)}")
            else:
                # Read query
                if match_clause:
                    explanation_parts.append("Find:")
                    explanation_parts.append(f"  - {self._explain_pattern(match_clause)}")

                if where_clause:
                    explanation_parts.append("Where:")
                    explanation_parts.append(f"  - {self._explain_where(where_clause)}")

                if return_clause:
                    explanation_parts.append("Return:")
                    explanation_parts.append(f"  - {self._explain_return(return_clause)}")

            explanation = "\n".join(explanation_parts)

            if not explanation:
                explanation = "Execute Cypher query. (Unable to generate detailed explanation)"

            logger.info("Generated Cypher query explanation")
            return explanation

        except Exception as e:
            logger.error("Error explaining Cypher query", error=str(e))
            return "Execute Cypher query. (Unable to generate detailed explanation)"

    @staticmethod
    def _extract_clause(query: str, clause_name: str) -> Optional[str]:
        """Extract a clause from Cypher query."""
        pattern = rf'\b{clause_name}\b\s+(.*?)(?:\b(?:WHERE|RETURN|CREATE|MERGE|SET|DELETE|WITH|ORDER|LIMIT)\b|$)'
        match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _explain_pattern(pattern: str) -> str:
        """Explain a Cypher pattern (nodes and relationships)."""
        explanations = []

        # Find node patterns like (n:Label) or (n:Label {property: value})
        node_pattern = r'\((\w+):(\w+)(?:\s*\{([^}]+)\})?\)'
        nodes = re.findall(node_pattern, pattern)

        for var_name, label, props in nodes:
            if props:
                explanations.append(f"Nodes labeled '{label}' (as {var_name}) with properties: {props}")
            else:
                explanations.append(f"Nodes labeled '{label}' (as {var_name})")

        # Find relationship patterns like -[r:TYPE]->
        rel_pattern = r'-\[(\w+):(\w+)(?:\s*\{([^}]+)\})?\]->'
        rels = re.findall(rel_pattern, pattern)

        for var_name, rel_type, props in rels:
            if props:
                explanations.append(f"Connected by '{rel_type}' relationships (as {var_name}) with properties: {props}")
            else:
                explanations.append(f"Connected by '{rel_type}' relationships (as {var_name})")

        return ", ".join(explanations) if explanations else pattern

    @staticmethod
    def _explain_where(where_clause: str) -> str:
        """Explain WHERE clause conditions."""
        # Simple explanation - could be enhanced with more parsing
        return where_clause

    @staticmethod
    def _explain_return(return_clause: str) -> str:
        """Explain RETURN clause."""
        return_items = [item.strip() for item in return_clause.split(",")]
        return ", ".join(return_items)

    @staticmethod
    def _explain_set(set_clause: str) -> str:
        """Explain SET clause for updates."""
        return set_clause


def explain_mongodb_query(
    query: str,
    collection: str,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate natural language explanation of MongoDB query.

    Args:
        query: MongoDB query as JSON string
        collection: Collection name
        schema: Optional schema for context

    Returns:
        Natural language explanation
    """
    try:
        query_dict = json.loads(query)
        explainer = MongoDBQueryExplainer(schema)
        return explainer.explain_query(query_dict, collection)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in MongoDB query", error=str(e))
        return f"Query on '{collection}'. (Invalid JSON format)"
    except Exception as e:
        logger.error("Error explaining MongoDB query", error=str(e))
        return f"Query on '{collection}'. (Unable to generate explanation)"


def explain_cypher_query(
    query: str,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate natural language explanation of Cypher query.

    Args:
        query: Cypher query string
        schema: Optional schema for context

    Returns:
        Natural language explanation
    """
    try:
        explainer = CypherQueryExplainer(schema)
        return explainer.explain_query(query)
    except Exception as e:
        logger.error("Error explaining Cypher query", error=str(e))
        return "Execute Cypher query. (Unable to generate explanation)"
