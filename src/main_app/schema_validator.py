"""Schema-based query validator for MongoDB and Neo4j queries.

This module provides validation logic to check queries against database schemas,
ensuring field names, types, and operations are valid before execution.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError:
    """Represents a validation error with severity and suggestion."""

    def __init__(
        self,
        field: str,
        message: str,
        severity: str = "error",
        suggestion: Optional[str] = None
    ):
        """
        Initialize a validation error.

        Args:
            field: Field name that caused the error
            message: Error message
            severity: Error severity ("error", "warning", "info")
            suggestion: Optional suggestion for fixing the error
        """
        self.field = field
        self.message = message
        self.severity = severity
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "field": self.field,
            "message": self.message,
            "severity": self.severity
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


class MongoDBSchemaValidator:
    """Validates MongoDB queries against collection schema."""

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize validator with schema.

        Args:
            schema: Collection schema from get_collection_schema()
        """
        self.schema = schema
        self.fields = schema.get("fields", {})
        self.field_names = set(self.fields.keys())

    def validate_query(self, query_dict: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a MongoDB query dict against the schema.

        Args:
            query_dict: Parsed MongoDB query dictionary

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Validate field names in query
        self._validate_fields_recursive(query_dict, errors, path="")

        # Check for common mistakes
        self._check_common_mistakes(query_dict, errors)

        is_valid = all(e.severity != "error" for e in errors)

        logger.info(
            "MongoDB query validation complete",
            is_valid=is_valid,
            error_count=len([e for e in errors if e.severity == "error"]),
            warning_count=len([e for e in errors if e.severity == "warning"])
        )

        return is_valid, errors

    def _validate_fields_recursive(
        self,
        obj: Any,
        errors: List[ValidationError],
        path: str = "",
        is_operator: bool = False
    ):
        """
        Recursively validate field names in query.

        Args:
            obj: Object to validate
            errors: List to append errors to
            path: Current path in object hierarchy
            is_operator: Whether current level is inside an operator
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # MongoDB operators start with $
                if key.startswith("$"):
                    # This is an operator, validate its value
                    self._validate_fields_recursive(value, errors, current_path, is_operator=True)
                else:
                    # This is a field name
                    if not is_operator and key not in self.field_names:
                        # Field doesn't exist in schema
                        suggestion = self._find_similar_field(key)
                        error_msg = f"Field '{key}' not found in schema"
                        if suggestion:
                            error_msg += f". Did you mean '{suggestion}'?"
                        errors.append(ValidationError(
                            field=key,
                            message=error_msg,
                            severity="error",
                            suggestion=suggestion
                        ))

                    # Validate nested value
                    self._validate_fields_recursive(value, errors, current_path, is_operator=False)

        elif isinstance(obj, list):
            for item in obj:
                self._validate_fields_recursive(item, errors, path, is_operator)

    def _find_similar_field(self, field: str) -> Optional[str]:
        """
        Find similar field name using Levenshtein distance.

        Args:
            field: Field name to find similar match for

        Returns:
            Most similar field name or None
        """
        min_distance = float('inf')
        similar_field = None

        for schema_field in self.field_names:
            distance = self._levenshtein_distance(field.lower(), schema_field.lower())
            if distance < min_distance and distance <= 2:  # Max 2 character difference
                min_distance = distance
                similar_field = schema_field

        return similar_field

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return MongoDBSchemaValidator._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _check_common_mistakes(self, query_dict: Dict[str, Any], errors: List[ValidationError]):
        """
        Check for common query mistakes.

        Args:
            query_dict: Query dictionary
            errors: List to append errors to
        """
        # Check for empty query
        if not query_dict:
            errors.append(ValidationError(
                field="query",
                message="Empty query will match all documents",
                severity="warning",
                suggestion="Add filter criteria or use limit to restrict results"
            ))

    def validate_field_type(self, field: str, operation: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate that an operation is compatible with the field type.

        Args:
            field: Field name
            operation: Operation name (e.g., "$gt", "$regex")
            value: Value being used

        Returns:
            Tuple of (is_valid, error_message)
        """
        if field not in self.fields:
            return False, f"Field '{field}' not found in schema"

        field_info = self.fields[field]
        field_types = field_info.get("types", [])

        # Numeric operations
        if operation in ["$gt", "$gte", "$lt", "$lte"]:
            if not any(t in ["int", "float"] for t in field_types):
                return False, f"Cannot use {operation} on non-numeric field '{field}' (types: {field_types})"

        # String operations
        elif operation in ["$regex", "$text"]:
            if "str" not in field_types:
                return False, f"Cannot use {operation} on non-string field '{field}' (types: {field_types})"

        # Array operations
        elif operation in ["$in", "$nin", "$all", "$elemMatch"]:
            if "list" not in field_types:
                return False, f"Operation {operation} is typically used with array fields, but '{field}' has types: {field_types}"

        return True, None


class Neo4jSchemaValidator:
    """Validates Cypher queries against Neo4j graph schema."""

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize validator with schema.

        Args:
            schema: Graph schema from get_schema()
        """
        self.schema = schema
        self.labels = set(schema.get("labels", []))
        self.relationship_types = set(schema.get("relationshipTypes", []))
        self.property_keys = set(schema.get("propertyKeys", []))

    def validate_query(self, cypher_query: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a Cypher query against the schema.

        Args:
            cypher_query: Cypher query string

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Extract and validate node labels
        self._validate_node_labels(cypher_query, errors)

        # Extract and validate relationship types
        self._validate_relationship_types(cypher_query, errors)

        # Extract and validate property names
        self._validate_property_names(cypher_query, errors)

        is_valid = all(e.severity != "error" for e in errors)

        logger.info(
            "Cypher query validation complete",
            is_valid=is_valid,
            error_count=len([e for e in errors if e.severity == "error"]),
            warning_count=len([e for e in errors if e.severity == "warning"])
        )

        return is_valid, errors

    def _validate_node_labels(self, query: str, errors: List[ValidationError]):
        """
        Validate node labels in query.

        Args:
            query: Cypher query
            errors: List to append errors to
        """
        # Match node label patterns like (n:Label) or (:Label)
        label_pattern = r'\([^)]*:([A-Za-z_][A-Za-z0-9_]*)'
        matches = re.findall(label_pattern, query)

        for label in matches:
            if label not in self.labels:
                suggestion = self._find_similar_label(label)
                error_msg = f"Node label '{label}' not found in schema"
                if suggestion:
                    error_msg += f". Did you mean '{suggestion}'?"
                errors.append(ValidationError(
                    field=label,
                    message=error_msg,
                    severity="error",
                    suggestion=suggestion
                ))

    def _validate_relationship_types(self, query: str, errors: List[ValidationError]):
        """
        Validate relationship types in query.

        Args:
            query: Cypher query
            errors: List to append errors to
        """
        # Match relationship type patterns like -[r:TYPE]-> or -[:TYPE]-
        rel_pattern = r'-\[[^]]*:([A-Z_][A-Z0-9_]*)\]'
        matches = re.findall(rel_pattern, query)

        for rel_type in matches:
            if rel_type not in self.relationship_types:
                suggestion = self._find_similar_relationship(rel_type)
                error_msg = f"Relationship type '{rel_type}' not found in schema"
                if suggestion:
                    error_msg += f". Did you mean '{suggestion}'?"
                errors.append(ValidationError(
                    field=rel_type,
                    message=error_msg,
                    severity="error",
                    suggestion=suggestion
                ))

    def _validate_property_names(self, query: str, errors: List[ValidationError]):
        """
        Validate property names in query.

        Args:
            query: Cypher query
            errors: List to append errors to
        """
        # Match property patterns like n.property or .property
        prop_pattern = r'\.([a-z_][a-zA-Z0-9_]*)'
        matches = re.findall(prop_pattern, query)

        for prop in matches:
            if prop not in self.property_keys:
                suggestion = self._find_similar_property(prop)
                # Use warning instead of error since properties might be dynamic
                error_msg = f"Property '{prop}' not found in schema"
                if suggestion:
                    error_msg += f". Did you mean '{suggestion}'?"
                errors.append(ValidationError(
                    field=prop,
                    message=error_msg,
                    severity="warning",
                    suggestion=suggestion
                ))

    def _find_similar_label(self, label: str) -> Optional[str]:
        """Find similar label name."""
        return self._find_similar(label, self.labels)

    def _find_similar_relationship(self, rel_type: str) -> Optional[str]:
        """Find similar relationship type."""
        return self._find_similar(rel_type, self.relationship_types)

    def _find_similar_property(self, prop: str) -> Optional[str]:
        """Find similar property name."""
        return self._find_similar(prop, self.property_keys)

    @staticmethod
    def _find_similar(target: str, candidates: set) -> Optional[str]:
        """
        Find similar string in candidates using Levenshtein distance.

        Args:
            target: String to find similar match for
            candidates: Set of candidate strings

        Returns:
            Most similar string or None
        """
        min_distance = float('inf')
        similar = None

        for candidate in candidates:
            distance = MongoDBSchemaValidator._levenshtein_distance(
                target.lower(),
                candidate.lower()
            )
            if distance < min_distance and distance <= 2:  # Max 2 character difference
                min_distance = distance
                similar = candidate

        return similar


def validate_mongodb_query(
    query: str,
    schema: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a MongoDB query string against schema.

    Args:
        query: MongoDB query as JSON string
        schema: Collection schema from get_collection_schema()

    Returns:
        Tuple of (is_valid, list of error dicts)
    """
    try:
        query_dict = json.loads(query)
    except json.JSONDecodeError as e:
        return False, [{
            "field": "query",
            "message": f"Invalid JSON: {str(e)}",
            "severity": "error"
        }]

    validator = MongoDBSchemaValidator(schema)
    is_valid, errors = validator.validate_query(query_dict)

    return is_valid, [e.to_dict() for e in errors]


def validate_cypher_query(
    query: str,
    schema: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a Cypher query string against schema.

    Args:
        query: Cypher query string
        schema: Graph schema from get_schema()

    Returns:
        Tuple of (is_valid, list of error dicts)
    """
    validator = Neo4jSchemaValidator(schema)
    is_valid, errors = validator.validate_query(query)

    return is_valid, [e.to_dict() for e in errors]
