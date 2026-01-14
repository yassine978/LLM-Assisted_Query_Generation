"""Query Translation Engine - Converts natural language to database queries using LLMs.

This module provides the QueryEngine class which handles:
- Natural language query analysis
- Database type detection
- Schema context gathering
- LLM-based query generation
- Query validation and execution
- Result formatting and explanation
"""

import asyncio
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Custom exceptions for better error handling
class QueryEngineError(Exception):
    """Base exception for Query Engine errors."""
    pass


class DatabaseConnectionError(QueryEngineError):
    """Raised when database connection fails."""
    pass


class QueryGenerationError(QueryEngineError):
    """Raised when query generation fails."""
    pass


class QueryValidationError(QueryEngineError):
    """Raised when query validation fails."""
    pass


class QueryExecutionError(QueryEngineError):
    """Raised when query execution fails."""
    pass


class QueryTimeoutError(QueryEngineError):
    """Raised when query execution times out."""
    pass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from groq import Groq
from src.main_app.mcp_manager import MCPManager
from src.main_app.query_history import QueryHistory
from src.main_app.schema_validator import validate_mongodb_query, validate_cypher_query
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Simple LRU cache with time-based expiration.

    This cache stores query results and schema information with automatic
    expiration and size limits.

    Attributes:
        max_size: Maximum number of items to store
        ttl: Time-to-live in seconds for cached items
        cache: OrderedDict storing cached items
    """

    def __init__(self, max_size: int = 100, ttl: int = 300):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to cache (default: 100)
            ttl: Time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments.

        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            Hash string to use as cache key
        """
        # Create a deterministic string from all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)

        # Hash the string for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get an item from the cache.

        Args:
            *args: Arguments used to create cache key
            **kwargs: Keyword arguments used to create cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        key = self._make_key(*args, **kwargs)

        if key not in self.cache:
            return None

        timestamp, value = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value

    def set(self, value: Any, *args, **kwargs) -> None:
        """Store an item in the cache.

        Args:
            value: Value to cache
            *args: Arguments used to create cache key
            **kwargs: Keyword arguments used to create cache key
        """
        key = self._make_key(*args, **kwargs)

        # Add or update item
        self.cache[key] = (time.time(), value)
        self.cache.move_to_end(key)

        # Enforce size limit (remove oldest)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, max_size, ttl)
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class QueryEngine:
    """Translates natural language queries to database-specific queries using LLMs.

    This class orchestrates the entire query translation pipeline:
    1. Analyze natural language to determine target database
    2. Gather schema context from the database
    3. Generate database-specific query using LLM
    4. Validate the generated query
    5. Execute the query
    6. Format and explain results

    Attributes:
        mcp_manager: MCP Manager instance for database access
        groq_client: Groq API client for LLM calls
        settings: Application settings
    """

    def __init__(
        self,
        mcp_manager: MCPManager,
        enable_cache: bool = True,
        enable_history: bool = True,
        history_db_path: str = "query_history.db"
    ):
        """Initialize the Query Engine.

        Args:
            mcp_manager: MCPManager instance for accessing database servers
            enable_cache: Whether to enable query and schema caching (default: True)
            enable_history: Whether to enable query history tracking (default: True)
            history_db_path: Path to SQLite database for history (default: "query_history.db")
        """
        self.mcp_manager = mcp_manager
        self.settings = get_settings()
        self.groq_client = Groq(api_key=self.settings.groq_api_key)

        # Initialize caches
        self.enable_cache = enable_cache
        if enable_cache:
            self.schema_cache = LRUCache(max_size=50, ttl=300)  # 5 minutes
            self.query_cache = LRUCache(max_size=100, ttl=600)  # 10 minutes
        else:
            self.schema_cache = None
            self.query_cache = None

        # Initialize query history
        self.enable_history = enable_history
        if enable_history:
            self.history = QueryHistory(db_path=history_db_path)
        else:
            self.history = None

        # Default timeouts (in seconds)
        self.llm_timeout = 30  # LLM query generation timeout
        self.db_query_timeout = 60  # Database query execution timeout

        logger.info(
            "QueryEngine initialized",
            caching_enabled=enable_cache,
            history_enabled=enable_history,
            llm_timeout=self.llm_timeout,
            db_query_timeout=self.db_query_timeout
        )

    async def _with_timeout(self, coro, timeout: float, error_message: str):
        """Execute a coroutine with a timeout.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            error_message: Error message if timeout occurs

        Returns:
            Result of the coroutine

        Raises:
            QueryTimeoutError: If the coroutine times out
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise QueryTimeoutError(f"{error_message} (timeout: {timeout}s)")

    async def detect_target_database(
        self,
        nl_query: str,
        ask_user_on_low_confidence: bool = False,
        confidence_threshold: float = 0.7
    ) -> Tuple[str, float, Optional[List[Tuple[str, float]]]]:
        """Detect which database type should be used for the query.

        Uses LLM to analyze the natural language query and determine the most
        appropriate database type based on query characteristics. Can handle
        ambiguous queries by returning alternatives when confidence is low.

        Args:
            nl_query: Natural language query from user
            ask_user_on_low_confidence: If True, returns alternatives when confidence is low
            confidence_threshold: Threshold below which query is considered ambiguous

        Returns:
            Tuple of (database_type, confidence_score, alternatives)
            - database_type is one of: "mongodb", "neo4j", "redis", "hbase", "rdf"
            - confidence_score is between 0.0 and 1.0
            - alternatives is list of (db_type, confidence) if confidence is low, else None

        Example:
            >>> engine = QueryEngine(mcp_manager)
            >>> db_type, confidence, alts = await engine.detect_target_database(
            ...     "Find all users older than 25"
            ... )
            >>> print(f"{db_type} (confidence: {confidence})")
            mongodb (confidence: 0.95)
        """
        try:
            # Get list of connected databases
            connected_dbs = self.mcp_manager.get_connected_servers()

            if not connected_dbs:
                logger.warning("No databases connected")
                return ("mongodb", 0.0, None)  # Default to MongoDB

            # For now, if only one database is connected, use it
            if len(connected_dbs) == 1:
                return (connected_dbs[0], 1.0, None)

            # Build prompt for database detection with alternative suggestions
            prompt = f"""Analyze this natural language query and determine which database type is most appropriate.

Connected databases: {', '.join(connected_dbs)}

Query: "{nl_query}"

Database characteristics:
- mongodb: Document store, good for flexible schemas, nested data, general queries, aggregations
- neo4j: Graph database, best for relationship queries, graph traversal, social networks, connected data
- redis: Key-value store, best for simple lookups, caching, counters, real-time operations
- hbase: Column-family store, best for time-series data, wide tables, large-scale data
- rdf: Triple store, best for semantic data, ontologies, linked data, knowledge graphs

Respond with ONLY a JSON object in this format:
{{
    "database": "database_name",
    "confidence": 0.95,
    "reasoning": "brief explanation",
    "alternatives": [
        {{"database": "alternative_db", "confidence": 0.75, "reasoning": "why this could also work"}}
    ]
}}

Include alternatives only if the query could reasonably work with multiple databases.
Confidence should reflect how certain you are about the primary choice."""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5000
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            # Sometimes LLM wraps JSON in markdown code blocks
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                json_lines = [l for l in lines if not l.startswith("```")]
                result_text = "\n".join(json_lines).strip()

            result = json.loads(result_text)
            database = result.get("database", "mongodb")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")

            # Process alternatives
            alternatives = None
            if ask_user_on_low_confidence and confidence < confidence_threshold:
                alt_list = result.get("alternatives", [])
                if alt_list:
                    alternatives = [(alt.get("database"), alt.get("confidence", 0.5)) for alt in alt_list]

            logger.info(
                "Database detected",
                query=nl_query,
                database=database,
                confidence=confidence,
                reasoning=reasoning,
                alternatives=alternatives
            )

            return (database, float(confidence), alternatives)

        except Exception as e:
            logger.error("Error detecting database", error=str(e))
            # Default to MongoDB if detection fails
            return ("mongodb", 0.5, None)

    async def gather_schema_context(
        self,
        database_type: str,
        database_name: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Gather schema information from the database for LLM context.

        Args:
            database_type: Type of database ("mongodb", "neo4j", etc.)
            database_name: Specific database name to query (optional)
            use_cache: Whether to use cached schema if available (default: True)

        Returns:
            Dictionary containing schema information formatted for LLM context

        Raises:
            ValueError: If database_type is not connected
        """
        # RDF doesn't use MCP, so skip connection check
        if database_type != "rdf" and not self.mcp_manager.is_connected(database_type):
            raise ValueError(f"Database '{database_type}' is not connected")

        # Check cache first
        if use_cache and self.enable_cache and self.schema_cache:
            cached = self.schema_cache.get(database_type, database_name or "default")
            if cached is not None:
                logger.debug("Using cached schema", database_type=database_type)
                return cached

        try:
            context = {"database_type": database_type}

            if database_type == "mongodb":
                # List databases
                databases = await self.mcp_manager.call_tool(
                    database_type,
                    "list_databases",
                    {}
                )

                context["databases"] = databases

                # If specific database provided, get its collections and schemas
                if database_name and database_name in databases:
                    collections = await self.mcp_manager.call_tool(
                        database_type,
                        "list_collections",
                        {"database": database_name}
                    )

                    context["collections"] = collections
                    context["schemas"] = {}

                    # Get schema for each collection (limit to first 3 for performance)
                    for collection in collections[:3]:
                        try:
                            schema = await self.mcp_manager.call_tool(
                                database_type,
                                "get_collection_schema",
                                {"database": database_name, "collection": collection}
                            )
                            context["schemas"][collection] = schema
                        except Exception as e:
                            logger.warning(
                                "Failed to get schema",
                                collection=collection,
                                error=str(e)
                            )

                # If no specific database, use first available
                elif databases and len(databases) > 0:
                    # Find first non-system database
                    user_dbs = [db for db in databases if db not in ['admin', 'local', 'config']]
                    if user_dbs:
                        database_name = user_dbs[0]
                        context["default_database"] = database_name

                        collections = await self.mcp_manager.call_tool(
                            database_type,
                            "list_collections",
                            {"database": database_name}
                        )
                        context["collections"] = collections

            elif database_type == "neo4j":
                # Get Neo4j graph schema
                schema = await self.mcp_manager.call_tool(
                    database_type,
                    "get_schema",
                    {}
                )
                # Merge schema into context
                context.update(schema)

            elif database_type == "redis":
                # Get Redis schema (key patterns and types)
                schema = await self.mcp_manager.call_tool(
                    database_type,
                    "get_schema",
                    {"use_cache": True, "max_keys": 1000}
                )
                # Merge schema into context
                context.update(schema)

            elif database_type == "hbase":
                # Get HBase schema (tables, column families, qualifiers)
                schema = await self.mcp_manager.call_tool(
                    database_type,
                    "get_schema",
                    {"use_cache": True, "sample_limit": 100}
                )
                # Merge schema into context
                context.update(schema)

            elif database_type == "rdf":
                # Get RDF ontology (classes, properties, namespaces)
                from src.main_app import rdf_tools
                schema = rdf_tools.get_ontology()
                if schema.get("success"):
                    context.update(schema)
                else:
                    logger.warning("Failed to get RDF ontology", error=schema.get("error"))

            logger.info(
                "Schema context gathered",
                database_type=database_type,
                database_name=database_name
            )

            # Cache the schema context
            if use_cache and self.enable_cache and self.schema_cache:
                self.schema_cache.set(context, database_type, database_name or "default")

            return context

        except Exception as e:
            logger.error(
                "Error gathering schema context",
                database_type=database_type,
                error=str(e)
            )
            return {"database_type": database_type, "error": str(e)}

    def _build_mongodb_prompt(
        self,
        nl_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for MongoDB query generation with improved field clarity."""
        
        # Extract schema information
        databases = context.get("databases", [])
        collections = context.get("collections", [])
        schemas = context.get("schemas", {})
        default_db = context.get("default_database", "")

        # Build schema description with example documents
        schema_desc = []
        for coll_name, schema in schemas.items():
            fields = schema.get("fields", {})
            field_desc = []
            for name, info in fields.items():
                types = ', '.join(info['types'])
                # Add examples if available
                examples = info.get('examples', [])
                example_str = f" (e.g., {examples[0]})" if examples else ""
                field_desc.append(f"  - {name}: {types}{example_str}")
            
            schema_desc.append(f"Collection '{coll_name}':\n" + "\n".join(field_desc))

        schema_text = "\n\n".join(schema_desc) if schema_desc else "No schema information available"

        prompt = f"""You are a MongoDB query expert. Convert the natural language query to a MongoDB query.

    Available Databases: {', '.join(databases)}
    Database to use: {default_db or databases[0] if databases else 'test_db'}
    Collections: {', '.join(collections)}

    Schema Information:
    {schema_text}

    Natural Language Query: "{nl_query}"

    CRITICAL FIELD DISTINCTIONS:
    - "rating" or "score" → use "imdb.rating" (numeric score, e.g., 7.6)
    - "content rating" or "age rating" → use "rated" (string like "PG-13", "R", "G")
    - "genres" → array of strings (e.g., ["Action", "Thriller"])
    - "year" → numeric year (e.g., 2018)

    When the user asks for:
    - "high ratings" → use {{"imdb.rating": {{"$gte": 7.0}}}}
    - "good ratings" → use {{"imdb.rating": {{"$gte": 7.0}}}}
    - "top rated" → use {{"imdb.rating": {{"$gte": 8.0}}}}
    - "highly rated" → use {{"imdb.rating": {{"$gte": 7.5}}}}
    - "rated R" or "PG-13 movies" → use {{"rated": "R"}} or {{"rated": "PG-13"}}

    Instructions:
    1. Generate a valid MongoDB query in JSON format
    2. Prefer existing schema fields when available, but use reasonable generic field names if needed
    3. Choose the most appropriate collection (or infer a reasonable collection name)
    4. Pay attention to nested fields (use dot notation like "imdb.rating")
    5. For numeric comparisons, use operators like $gte, $lte, $gt, $lt
    6. For date/time queries, use reasonable field names like "created_at", "updated_at", "timestamp", "year"
    7. CRITICAL: Do NOT include comments (# or //) in the JSON - JSON does not support comments
    8. Return ONLY a JSON object with this structure:

    For FIND operations (simple queries, filtering):
    {{
        "database": "database_name",
        "collection": "collection_name",
        "operation": "find",
        "query": {{"field": "value"}},
        "projection": {{"field1": 1, "field2": 1}},  # Optional: specify which fields to return
        "explanation": "brief explanation"
    }}

    For AGGREGATE operations (grouping, counting, complex transformations):
    {{
        "database": "database_name",
        "collection": "collection_name",
        "operation": "aggregate",
        "query": [
            {{"$match": {{"field": "value"}}}},
            {{"$group": {{"_id": "$field", "count": {{"$sum": 1}}}}}}
        ],
        "explanation": "brief explanation"
    }}

    CRITICAL: For aggregation, the "query" field must be an ARRAY of pipeline stages, NOT an object.

    Example 1 - Find high-rated action movies:
    {{
        "database": "sample_mflix",
        "collection": "movies",
        "operation": "find",
        "query": {{
            "genres": "Action",
            "imdb.rating": {{"$gte": 7.0}}
        }},
        "projection": {{"title": 1, "year": 1, "imdb.rating": 1, "genres": 1}},
        "explanation": "Finds action movies with IMDb rating of 7.0 or higher"
    }}

    Example 2 - Find movies from 2018 with high ratings:
    {{
        "database": "sample_mflix",
        "collection": "movies",
        "operation": "find",
        "query": {{
            "year": {{"$gte": 2018}},
            "imdb.rating": {{"$gte": 7.5}}
        }},
        "explanation": "Finds movies from 2018 or later with high IMDb ratings (7.5+)"
    }}

    Example 3 - Count movies by rating category:
    {{
        "database": "sample_mflix",
        "collection": "movies",
        "operation": "aggregate",
        "query": [
            {{"$match": {{"imdb.rating": {{"$exists": true}}}}}},
            {{"$bucket": {{
                "groupBy": "$imdb.rating",
                "boundaries": [0, 5, 7, 8, 10],
                "default": "Other",
                "output": {{"count": {{"$sum": 1}}}}
            }}}}
        ],
        "explanation": "Groups movies by rating ranges: 0-5 (low), 5-7 (medium), 7-8 (good), 8-10 (excellent)"
    }}

    Example 4 - Find R-rated movies (content rating):
    {{
        "database": "sample_mflix",
        "collection": "movies",
        "operation": "find",
        "query": {{"rated": "R"}},
        "explanation": "Finds movies with R content rating (restricted)"
    }}

    Generate the MongoDB query now:"""

        return prompt

    def _build_neo4j_prompt(
        self,
        nl_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for Neo4j Cypher query generation.

        Args:
            nl_query: Natural language query
            context: Schema context from gather_schema_context()

        Returns:
            Formatted prompt string for LLM
        """
        # Extract schema information
        labels = context.get("labels", [])
        relationship_types = context.get("relationshipTypes", [])
        property_keys = context.get("propertyKeys", [])
        node_count = context.get("nodeCount", 0)
        rel_count = context.get("relationshipCount", 0)

        # Build schema description
        labels_text = ", ".join(labels) if labels else "No labels available"
        rels_text = ", ".join(relationship_types) if relationship_types else "No relationships available"

        prompt = f"""You are a Neo4j Cypher query expert. Convert the natural language query to a Cypher query.

Graph Schema:
- Node Labels: {labels_text}
- Relationship Types: {rels_text}
- Total Nodes: {node_count}
- Total Relationships: {rel_count}

Property Keys available: {', '.join(property_keys[:20])}

Natural Language Query: "{nl_query}"

Instructions:
1. Generate a valid Cypher query
2. Use only node labels and relationships that exist in the schema
3. Return ONLY a JSON object with this structure:
{{
    "query": "MATCH (n:Label) RETURN n",
    "explanation": "brief explanation of what the query does"
}}

Common patterns:
- Find nodes: MATCH (n:Person) WHERE n.age > 25 RETURN n
- Find relationships: MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person) RETURN a, r, b
- Traversal: MATCH (a:Person {{name: "Alice"}})-[:FRIENDS_WITH*1..2]-(friend) RETURN friend
- Aggregation: MATCH (p:Person) RETURN p.city, count(p) as count

Example response:
{{
    "query": "MATCH (p:Person)-[:WORKS_AT]->(c:Company) WHERE c.location = 'New York' RETURN p.name, c.name",
    "explanation": "Finds all people who work at companies located in New York"
}}

Generate the Cypher query now:"""

        return prompt

    def _build_redis_prompt(
        self,
        nl_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for Redis MCP tool translation.

        Args:
            nl_query: Natural language query
            context: Schema context from gather_schema_context()

        Returns:
            Formatted prompt string for LLM
        """
        # Extract schema information
        total_keys = context.get("total_keys", 0)
        key_patterns = context.get("key_patterns", [])[:10]
        type_dist = context.get("type_distribution", {})

        # Format schema information
        patterns_text = "\n".join([
            f"  - {p['pattern']} ({p['count']} keys)"
            for p in key_patterns
        ]) if key_patterns else "No patterns found"

        # Format type distribution
        import json
        types_text = json.dumps(type_dist)

        prompt = f"""You are a Redis MCP tool translator. Your job is to translate natural language queries into MCP tool calls.

Redis Schema Information:
- Total Keys: {total_keys}
- Type Distribution: {types_text}

Key Patterns in Database:
{patterns_text}

Available MCP Tools:
- ping: Test Redis connection
  Parameters: {{}}
- get_db_info: Get Redis database information and statistics
  Parameters: {{}}
- get_key: Get value for a Redis key
  Parameters: {{"key": "string (required)"}}
- set_key: Set a string key in Redis
  Parameters: {{"key": "string (required)", "value": "string (required)", "ex": "integer (optional) - expiration in seconds"}}
- delete_key: Delete a Redis key
  Parameters: {{"key": "string (required)"}}
- get_keys: Get Redis keys matching a pattern
  Parameters: {{"pattern": "string (default: *)", "limit": "integer (default: 100)"}}
- get_schema: Get comprehensive Redis schema information
  Parameters: {{"use_cache": "boolean (default: true)", "max_keys": "integer (default: 1000)"}}
- clear_schema_cache: Clear the Redis schema cache
  Parameters: {{}}
- execute_command: Execute a raw Redis command
  Parameters: {{"command": "string (required)", "args": "array (default: [])"}}

Natural Language Query: "{nl_query}"

IMPORTANT INSTRUCTIONS:
1. Analyze the query and determine which MCP tool to use
2. Extract the necessary parameters from the query
3. Match key patterns from the schema when looking for keys
4. For queries requiring filtering (like "users in New York"), explain this is not directly supported and use get_keys to get all matching keys
5. For TTL/expiration queries, explain that Redis doesn't support querying by TTL directly
6. Return ONLY a JSON object with this structure:

{{
    "command": "tool_name",
    "args": ["arg1", "arg2"] OR {{"param1": "value1", "param2": "value2"}},
    "explanation": "Brief explanation of what this tool call will do"
}}

Note: For execute_command, use "args" as array. For other tools, use parameter object format.

Examples:

Example 1 - "Show me all user profiles"
{{
    "command": "get_keys",
    "args": {{"pattern": "user:*", "limit": 100}},
    "explanation": "Get all keys matching the user profile pattern"
}}

Example 2 - "Get user 1001's information"
{{
    "command": "get_key",
    "args": {{"key": "user:1001"}},
    "explanation": "Retrieve the user:1001 hash with all user information"
}}

Example 3 - "What's in the database?"
{{
    "command": "get_db_info",
    "args": {{}},
    "explanation": "Get comprehensive database statistics and information"
}}

Example 4 - "Show me the database structure"
{{
    "command": "get_schema",
    "args": {{"use_cache": true, "max_keys": 1000}},
    "explanation": "Get full schema with key patterns and data types"
}}

Example 5 - "Find all session keys"
{{
    "command": "get_keys",
    "args": {{"pattern": "session:*", "limit": 100}},
    "explanation": "Find all keys matching the session pattern"
}}

Example 6 - "Create a new key called test with value hello"
{{
    "command": "set_key",
    "args": {{"key": "test", "value": "hello"}},
    "explanation": "Set a new string key 'test' with value 'hello'"
}}

Example 7 - "Delete the key test"
{{
    "command": "delete_key",
    "args": {{"key": "test"}},
    "explanation": "Delete the key named 'test'"
}}

Example 8 - "Get all products"
{{
    "command": "get_keys",
    "args": {{"pattern": "product:*", "limit": 100}},
    "explanation": "Find all product keys in the database"
}}

Now translate this query: "{nl_query}"

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _build_hbase_prompt(
        self,
        nl_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for HBase operation generation with enhanced guidance.

        Args:
            nl_query: Natural language query
            context: Schema context from gather_schema_context()

        Returns:
            Formatted prompt string for LLM
        """
        # Extract schema information
        total_tables = context.get("total_tables", 0)
        tables = context.get("tables", [])

        # Build tables description with better formatting
        tables_text = []
        for table in tables[:10]:  # Top 10 tables
            table_name = table["table_name"]
            families = table["column_families"]
            qualifiers = table.get("column_qualifiers", {})
            row_count = table.get("estimated_row_count", 0)
            row_key_pattern = table.get("row_key_pattern")  # NEW
            sample_rows = table.get("sample_rows", [])

            # Format qualifiers
            qual_text = []
            for family, quals in qualifiers.items():
                qual_text.append(f"    - {family}: {', '.join(quals[:10])}")

            table_desc = f"  - {table_name} ({row_count} rows)\n"
            table_desc += f"    Column Families: {', '.join(families)}\n"
            if qual_text:
                table_desc += "    Column Qualifiers:\n" + "\n".join(qual_text)

            # Add row key pattern info to help LLM
            if row_key_pattern:
                table_desc += f"\n    Row Key Pattern: {row_key_pattern}\n"

            # Add sample row keys to show actual examples
            if sample_rows:
                row_keys = [r.get("row_key", "") for r in sample_rows[:3]]
                table_desc += f"    Example Row Keys: {', '.join(row_keys)}\n"

            tables_text.append(table_desc)

        tables_desc = "\n".join(tables_text) if tables_text else "No tables found"

        prompt = f"""You are an HBase expert. Convert the natural language query to HBase operations.

    HBase Database Schema:
    - Total Tables: {total_tables}

    Tables and Structure:
    {tables_desc}

    Natural Language Query: "{nl_query}"

    HBase Key Concepts:
    - Row Keys: Used for get operations (exact match) or scan ranges (lexicographic)
    - Column Families: Logical grouping of data (e.g., "profile", "contact", "preferences")
    - Column Qualifiers: Individual columns within families (e.g., "name", "email", "theme")
    - HBase does NOT support native sorting - results return in row key lexicographic order
    - HBase does NOT support WHERE clauses - filtering is client-side

    Common HBase Operations:
    - get_row: Retrieve a specific row by exact row key. Use ONLY if you have the exact key from the query
    - scan_table: Retrieve multiple rows with optional row range and column/value filtering
    - count_rows: Count rows matching filter criteria WITHOUT returning row data. Use for "count", "how many" queries
    - put_row: Insert or update a row with specified columns
    - delete_row: Delete a row by row key

    CRITICAL: You must respond with ONLY valid JSON. No explanation text before or after.
    Your response must be a single JSON object with this exact structure:
    {{
        "operation": "OPERATION_NAME",
        "table": "table_name",
        "row_key": "optional_row_key",
        "row_start": "optional_start_key",
        "row_stop": "optional_stop_key",
        "limit": 100,
        "columns": {{}},
        "filter_column": "optional_column_name",
        "filter_value": "optional_value",
        "filter_operator": "optional_comparison_operator",
        "explanation": "brief explanation of what the operation does"
    }}

    COMPARISON OPERATORS (use with filter_column and filter_value):
    - "=" : Exact equality (default if not specified)
    - ">" : Greater than (works for numeric and string values)
    - "<" : Less than (works for numeric and string values)
    - ">=" : Greater than or equal
    - "<=" : Less than or equal
    - "!=" : Not equal

    IMPORTANT RULES FOR QUERY GENERATION:
    1. Use get_row ONLY if the query specifies an EXACT row key that matches table examples
    2. Use count_rows for queries asking "count", "how many", "number of" - it returns ONLY the count
    3. Use scan_table for queries asking to "show", "list", "find", "get" rows with data
    4. For comparison filters (e.g., "older than 30", "age > 25"), use:
       - "filter_column": "full_column_name_with_family:qualifier"
       - "filter_value": "threshold_value"
       - "filter_operator": ">", "<", ">=", "<=", "=", or "!="
    5. For exact match filters (e.g., "dark theme", "status = active"), use:
       - "filter_column": "full_column_name_with_family:qualifier"
       - "filter_value": "exact_value_to_match"
       - "filter_operator": "=" (or omit, defaults to "=")
    6. Column specifications use format: {{"family": ["qual1", "qual2"]}} or {{"family:qualifier": ""}}
    7. Row key ranges are INCLUSIVE on row_start, EXCLUSIVE on row_stop
    8. Respect HBase limitations - you cannot sort by non-key columns

    Example 1 - Query: "Get user 1001"
    {{
        "operation": "get_row",
        "table": "users",
        "row_key": "user_1001",
        "explanation": "Retrieve user with exact row key user_1001"
    }}

    Example 2 - Query: "Find users with dark theme"
    {{
        "operation": "scan_table",
        "table": "users",
        "limit": 100,
        "filter_column": "preferences:theme",
        "filter_value": "dark",
        "explanation": "Scan users table and filter for rows where preferences:theme column equals 'dark'"
    }}

    Example 3 - Query: "Get first 10 users"
    {{
        "operation": "scan_table",
        "table": "users",
        "limit": 10,
        "explanation": "Scan and return first 10 rows from users table"
    }}

    Example 4 - Query: "Get users from user_1001 to user_2000"
    {{
        "operation": "scan_table",
        "table": "users",
        "row_start": "user_1001",
        "row_stop": "user_2000",
        "limit": 1000,
        "explanation": "Scan users table in the specified row key range [user_1001, user_2000)"
    }}

    Example 5 - Query: "Get all products with category electronics"
    {{
        "operation": "scan_table",
        "table": "products",
        "limit": 1000,
        "filter_column": "info:category",
        "filter_value": "electronics",
        "explanation": "Scan all products and filter for category = electronics"
    }}

    Example 6 - Query: "Get specific columns for user 1001"
    {{
        "operation": "get_row",
        "table": "users",
        "row_key": "user_1001",
        "columns": {{"profile": ["name", "age"]}},
        "explanation": "Get user_1001 but only retrieve name and age from profile family"
    }}

    Example 7 - Query: "Show users older than 30"
    {{
        "operation": "scan_table",
        "table": "users",
        "limit": 100,
        "filter_column": "profile:age",
        "filter_value": "30",
        "filter_operator": ">",
        "explanation": "Scan users table and filter for rows where profile:age > 30"
    }}

    Example 8 - Query: "Count all users"
    {{
        "operation": "count_rows",
        "table": "users",
        "explanation": "Count total number of rows in users table"
    }}

    Example 9 - Query: "How many users have dark theme"
    {{
        "operation": "count_rows",
        "table": "users",
        "filter_column": "preferences:theme",
        "filter_value": "dark",
        "explanation": "Count users where preferences:theme equals 'dark'"
    }}

    Example 10 - Query: "Count users aged 35 or older"
    {{
        "operation": "count_rows",
        "table": "users",
        "filter_column": "profile:age",
        "filter_value": "35",
        "filter_operator": ">=",
        "explanation": "Count users where profile:age >= 35"
    }}

    Now generate ONLY the JSON for: "{nl_query}" """

        return prompt

    def _build_sparql_prompt(
        self,
        nl_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for SPARQL query generation.

        Args:
            nl_query: Natural language query
            context: Schema context from gather_schema_context()

        Returns:
            Formatted prompt string for LLM
        """
        # Extract schema information
        classes = context.get("classes", [])
        properties = context.get("properties", [])
        namespaces = context.get("namespaces", [])

        # Build classes description
        classes_text = []
        for cls in classes[:20]:  # Top 20 classes
            uri = cls.get("class", "")
            count = cls.get("count", 0)
            # Extract local name from URI
            local_name = uri.split("/")[-1] if "/" in uri else uri
            classes_text.append(f"  - {local_name} ({count} instances) - {uri}")

        classes_desc = "\n".join(classes_text) if classes_text else "No classes found"

        # Build properties description
        props_text = []
        for prop in properties[:30]:  # Top 30 properties
            uri = prop.get("property", "")
            count = prop.get("count", 0)
            # Extract local name from URI
            local_name = uri.split("/")[-1] if "/" in uri else uri
            props_text.append(f"  - {local_name} ({count} uses) - {uri}")

        props_desc = "\n".join(props_text) if props_text else "No properties found"

        # Build namespaces description
        ns_text = "\n".join([f"  - {ns}" for ns in namespaces[:10]]) if namespaces else "No namespaces found"

        prompt = f"""You are a SPARQL expert. Convert the natural language query to a SPARQL query.

RDF Dataset Schema:
- Total Classes: {len(classes)}
- Total Properties: {len(properties)}
- Namespaces: {len(namespaces)}

Common Classes:
{classes_desc}

Common Properties:
{props_desc}

Namespaces:
{ns_text}

Natural Language Query: "{nl_query}"

CRITICAL REQUIREMENTS:
1. You must respond with ONLY valid JSON. No explanation text before or after.
2. Your SPARQL query MUST include ALL necessary PREFIX declarations.
3. ALWAYS include these prefixes at the start of every query:
   PREFIX wd: <http://www.wikidata.org/entity/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX schema: <http://schema.org/>

Your response must be a single JSON object with this exact structure:
{{
    "query": "SPARQL query string with ALL prefixes",
    "explanation": "brief explanation"
}}

Example 1 - "Find all physicists":
{{
    "query": "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?person ?name WHERE {{ ?person wdt:P106 wd:Q169470 . ?person rdfs:label ?name }} LIMIT 10",
    "explanation": "Query to find persons whose occupation is physicist (Q169470)"
}}

Example 2 - "Count all humans":
{{
    "query": "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT (COUNT(?person) as ?count) WHERE {{ ?person wdt:P31 wd:Q5 }}",
    "explanation": "Count all instances of human (Q5)"
}}

Example 3 - "Get Einstein's information":
{{
    "query": "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?property ?value WHERE {{ wd:Q937 ?property ?value }} LIMIT 20",
    "explanation": "Get all properties and values for Albert Einstein (Q937)"
}}

Important Notes:
- Always include PREFIX declarations for namespaces used
- Use LIMIT to restrict results (default: 10)
- Use rdfs:label to get human-readable names
- Common Wikidata properties: P31 (instance of), P106 (occupation), P27 (country of citizenship), P569 (date of birth)
- Common Wikidata classes: Q5 (human), Q169470 (physicist), Q170790 (mathematician)

Now generate ONLY the JSON for: "{nl_query}" """

        return prompt

    async def generate_query(
        self,
        nl_query: str,
        database_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a database-specific query from natural language.

        Args:
            nl_query: Natural language query
            database_type: Type of database ("mongodb", "neo4j", etc.)
            context: Schema context from gather_schema_context()

        Returns:
            Dictionary containing generated query and metadata

        Raises:
            ValueError: If database type is not supported
        """
        try:
            if database_type == "mongodb":
                prompt = self._build_mongodb_prompt(nl_query, context)
            elif database_type == "neo4j":
                prompt = self._build_neo4j_prompt(nl_query, context)
            elif database_type == "redis":
                prompt = self._build_redis_prompt(nl_query, context)
            elif database_type == "hbase":
                prompt = self._build_hbase_prompt(nl_query, context)
            elif database_type == "rdf":
                prompt = self._build_sparql_prompt(nl_query, context)
            else:
                raise ValueError(f"Database type '{database_type}' not yet supported")

            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5000
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            # Sometimes LLM wraps JSON in markdown code blocks or adds explanatory text
            if result_text.startswith("```"):
                # Extract JSON from code block
                lines = result_text.split("\n")
                json_lines = [l for l in lines if not l.startswith("```")]
                result_text = "\n".join(json_lines).strip()

            # If response doesn't start with {, try to find JSON object
            if not result_text.startswith("{"):
                # Look for first { and try to extract JSON from there
                json_start = result_text.find("{")
                if json_start != -1:
                    result_text = result_text[json_start:]
                    # Try to find matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(result_text):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end != -1:
                        result_text = result_text[:json_end]

            result = json.loads(result_text)

            logger.info(
                "Query generated",
                database_type=database_type,
                collection=result.get("collection"),
                operation=result.get("operation")
            )

            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e), response=result_text[:200])
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            logger.error("Error generating query", error=str(e))
            raise

    async def validate_query(
        self,
        database_type: str,
        database: str,
        collection: str,
        query: str
    ) -> Dict[str, Any]:
        """Validate a generated query before execution.

        Performs two levels of validation:
        1. Syntax validation using database's native validator
        2. Schema validation using our schema validator

        Args:
            database_type: Type of database
            database: Database name
            collection: Collection/table name
            query: Query to validate (as JSON string for MongoDB, Cypher string for Neo4j)

        Returns:
            Validation result dictionary with format:
            {
                "valid": bool,
                "syntax_valid": bool,
                "schema_valid": bool,
                "errors": list,  # Schema validation errors
                "warnings": list,  # Schema validation warnings
                "error": str (if syntax validation fails)
            }
        """
        try:
            validation_result = {
                "valid": False,
                "syntax_valid": False,
                "schema_valid": False,
                "errors": [],
                "warnings": []
            }

            if database_type == "mongodb":
                # Step 1: Syntax validation using MongoDB's native validator
                syntax_result = await self.mcp_manager.call_tool(
                    database_type,
                    "validate_query",
                    {
                        "database": database,
                        "collection": collection,
                        "query": query
                    }
                )

                validation_result["syntax_valid"] = syntax_result.get("valid", False)

                if not validation_result["syntax_valid"]:
                    validation_result["error"] = syntax_result.get("error", "Syntax validation failed")
                    return validation_result

                # Step 2: Schema validation
                # Get schema
                schema = await self.mcp_manager.call_tool(
                    database_type,
                    "get_collection_schema",
                    {
                        "database": database,
                        "collection": collection
                    }
                )

                # Validate against schema
                schema_valid, schema_errors = validate_mongodb_query(query, schema)
                validation_result["schema_valid"] = schema_valid

                # Separate errors and warnings
                validation_result["errors"] = [e for e in schema_errors if e.get("severity") == "error"]
                validation_result["warnings"] = [e for e in schema_errors if e.get("severity") == "warning"]

                # Overall validity requires both syntax and schema to be valid (errors only, warnings are OK)
                validation_result["valid"] = validation_result["syntax_valid"] and len(validation_result["errors"]) == 0

                if not validation_result["schema_valid"]:
                    logger.warning("Schema validation issues found", errors=validation_result["errors"], warnings=validation_result["warnings"])

                return validation_result

            elif database_type == "neo4j":
                # Step 1: Syntax validation using Neo4j's native validator
                syntax_result = await self.mcp_manager.call_tool(
                    database_type,
                    "validate_cypher",
                    {"query": query}
                )

                validation_result["syntax_valid"] = syntax_result.get("valid", False)

                if not validation_result["syntax_valid"]:
                    validation_result["error"] = syntax_result.get("error", "Syntax validation failed")
                    return validation_result

                # Step 2: Schema validation
                # Get schema
                schema = await self.mcp_manager.call_tool(
                    database_type,
                    "get_schema",
                    {}
                )

                # Validate against schema
                schema_valid, schema_errors = validate_cypher_query(query, schema)
                validation_result["schema_valid"] = schema_valid

                # Separate errors and warnings
                validation_result["errors"] = [e for e in schema_errors if e.get("severity") == "error"]
                validation_result["warnings"] = [e for e in schema_errors if e.get("severity") == "warning"]

                # Overall validity requires both syntax and schema to be valid (errors only, warnings are OK)
                validation_result["valid"] = validation_result["syntax_valid"] and len(validation_result["errors"]) == 0

                if not validation_result["schema_valid"]:
                    logger.warning("Schema validation issues found", errors=validation_result["errors"], warnings=validation_result["warnings"])

                return validation_result

            else:
                raise ValueError(f"Validation not implemented for {database_type}")

        except Exception as e:
            logger.error("Error validating query", error=str(e))
            return {"valid": False, "error": str(e)}

    async def execute_query(
        self,
        database_type: str,
        database: str,
        collection: str,
        query: Any,
        operation: str = "find",
        limit: int = 100
    ) -> Dict[str, Any]:
        """Execute a database query.

        Args:
            database_type: Type of database
            database: Database name
            collection: Collection/table name
            query: Query to execute
            operation: Operation type ("find", "aggregate", etc.)
            limit: Maximum results to return

        Returns:
            Query execution results
        """
        try:
            if database_type == "mongodb":
                if operation == "find":
                    # Convert query to JSON string if it's a dict
                    query_str = json.dumps(query) if isinstance(query, dict) else query

                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "execute_query",
                        {
                            "database": database,
                            "collection": collection,
                            "query": query_str,
                            "limit": limit
                        }
                    )
                elif operation == "aggregate":
                    # For aggregation, query is already a list (pipeline)
                    # Convert to JSON string (list or dict)
                    pipeline_str = json.dumps(query) if isinstance(query, (dict, list)) else query

                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "aggregate",
                        {
                            "database": database,
                            "collection": collection,
                            "pipeline": pipeline_str
                        }
                    )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                return result

            elif database_type == "neo4j":
                # For Neo4j, query is the Cypher string
                result = await self.mcp_manager.call_tool(
                    database_type,
                    "run_cypher",
                    {
                        "query": query,
                        "limit": limit
                    }
                )
                return result

            elif database_type == "redis":
                # For Redis, call the MCP tool directly
                tool_name = query.get("tool") if isinstance(query, dict) else operation
                arguments = query.get("arguments", {}) if isinstance(query, dict) else {}

                logger.info(
                    "Executing Redis MCP tool",
                    tool=tool_name,
                    arguments=arguments
                )

                result = await self.mcp_manager.call_tool(
                    database_type,
                    tool_name,
                    arguments
                )

                # Format result to match expected structure
                # get_schema doesn't return success flag, just data
                has_success_flag = "success" in result
                is_successful = result.get("success", True) if has_success_flag else (tool_name == "get_schema")

                if is_successful:
                    # Handle different result types based on tool
                    if tool_name == "get_keys":
                        keys = result.get("keys", [])
                        results = [{"key": k} for k in keys]
                        result_count = len(keys)
                    elif tool_name == "get_key":
                        # Single key result
                        results = [result]
                        result_count = 1
                    elif tool_name == "get_db_info":
                        # Database info
                        results = [result]
                        result_count = 1
                    elif tool_name == "get_schema":
                        # Schema info - returns data directly without success flag
                        results = [result]
                        result_count = 1
                    elif tool_name in ["set_key", "delete_key"]:
                        # Write operations
                        results = [{"message": result.get("message", "Operation successful")}]
                        result_count = 1
                    elif tool_name == "execute_command":
                        # Raw command result
                        redis_result = result.get("result", [])
                        if isinstance(redis_result, list):
                            results = [{"key": k} for k in redis_result] if redis_result else []
                            result_count = len(redis_result)
                        elif isinstance(redis_result, dict):
                            results = [redis_result]
                            result_count = 1
                        else:
                            results = [{"value": redis_result}]
                            result_count = 1
                    else:
                        # Generic handling
                        results = [result]
                        result_count = 1

                    return {
                        "results": results,
                        "result_count": result_count
                    }
                else:
                    raise ValueError(result.get("error", "Redis tool call failed"))

            elif database_type == "hbase":
                table_name = query.get("table")
                
                if not table_name:
                    raise ValueError("table name is required for HBase operations")
                
                operation = query.get("operation", "scan_table")
                
                if operation == "get_row":
                    row_key = query.get("row_key")
                    
                    if not row_key:
                        raise ValueError("row_key is required for get_row operation")
                    
                    logger.info(
                        "Executing HBase get_row",
                        table=table_name,
                        row_key=row_key
                    )
                    
                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "get_row",
                        {
                            "table_name": table_name,
                            "row_key": row_key
                        }
                    )
                    
                    # Check if row was found
                    if not result.get("success"):
                        # Instead of raising error, return empty results for graceful handling
                        logger.info(
                            f"Row not found - returning empty result set",
                            table=table_name,
                            row_key=row_key
                        )
                        return {
                            "results": [],
                            "result_count": 0
                        }
                    
                    # Format result as list of rows for consistency
                    return {
                        "results": [{
                            "row_key": result.get("row_key"),
                            "columns": result.get("columns", {})
                        }],
                        "result_count": 1
                    }
                
                elif operation == "scan_table":
                    row_start = query.get("row_start")
                    row_stop = query.get("row_stop")
                    scan_limit = query.get("limit", 100)
                    columns = query.get("columns", {})
                    filter_column = query.get("filter_column")
                    filter_value = query.get("filter_value")
                    filter_operator = query.get("filter_operator")  # NEW: Support comparison operators

                    logger.info(
                        "Executing HBase scan_table",
                        table=table_name,
                        limit=scan_limit,
                        filter_column=filter_column,
                        filter_operator=filter_operator,
                        has_range=bool(row_start or row_stop)
                    )

                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "scan_table",
                        {
                            "table_name": table_name,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "limit": scan_limit,
                            "columns": columns if columns else None,
                            "filter_column": filter_column,
                            "filter_value": filter_value,
                            "filter_operator": filter_operator
                        }
                    )
                    
                    # Format result
                    if result.get("success"):
                        rows = result.get("rows", [])
                        logger.info(
                            "HBase scan completed successfully",
                            table=table_name,
                            returned_rows=len(rows),
                            scanned_rows=result.get("scanned_count", len(rows))
                        )
                        return {
                            "results": rows,
                            "result_count": len(rows)
                        }
                    else:
                        raise ValueError(result.get("error", "Failed to scan table"))
                
                elif operation == "put_row":
                    row_key = query.get("row_key")
                    columns = query.get("columns", {})
                    
                    if not row_key:
                        raise ValueError("row_key is required for put_row operation")
                    if not columns:
                        raise ValueError("columns are required for put_row operation")
                    
                    logger.info(
                        "Executing HBase put_row",
                        table=table_name,
                        row_key=row_key,
                        column_count=len(columns)
                    )
                    
                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "put_row",
                        {
                            "table_name": table_name,
                            "row_key": row_key,
                            "columns": columns
                        }
                    )
                    
                    if result.get("success"):
                        return {
                            "results": [{
                                "operation": "put_row",
                                "row_key": row_key,
                                "message": result.get("message", "Row inserted/updated successfully")
                            }],
                            "result_count": 1
                        }
                    else:
                        raise ValueError(result.get("error", "Failed to insert/update row"))
                
                elif operation == "delete_row":
                    row_key = query.get("row_key")

                    if not row_key:
                        raise ValueError("row_key is required for delete_row operation")

                    logger.info(
                        "Executing HBase delete_row",
                        table=table_name,
                        row_key=row_key
                    )

                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "delete_row",
                        {
                            "table_name": table_name,
                            "row_key": row_key
                        }
                    )

                    if result.get("success"):
                        return {
                            "results": [{
                                "operation": "delete_row",
                                "row_key": row_key,
                                "message": result.get("message", "Row deleted successfully")
                            }],
                            "result_count": 1
                        }
                    else:
                        raise ValueError(result.get("error", "Failed to delete row"))

                elif operation == "count_rows":
                    row_start = query.get("row_start")
                    row_stop = query.get("row_stop")
                    filter_column = query.get("filter_column")
                    filter_value = query.get("filter_value")
                    filter_operator = query.get("filter_operator")

                    logger.info(
                        "Executing HBase count_rows",
                        table=table_name,
                        filter_column=filter_column,
                        filter_operator=filter_operator
                    )

                    result = await self.mcp_manager.call_tool(
                        database_type,
                        "count_rows",
                        {
                            "table_name": table_name,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "filter_column": filter_column,
                            "filter_value": filter_value,
                            "filter_operator": filter_operator
                        }
                    )

                    if result.get("success"):
                        count = result.get("count", 0)
                        return {
                            "results": [{
                                "operation": "count_rows",
                                "count": count,
                                "message": result.get("message", f"Count: {count}")
                            }],
                            "result_count": count
                        }
                    else:
                        raise ValueError(result.get("error", "Failed to count rows"))

                else:
                    raise ValueError(f"Unknown HBase operation: {operation}")

            elif database_type == "rdf":
                # For RDF, execute the SPARQL query
                from src.main_app import rdf_tools

                # Query is the SPARQL string
                sparql_query = query if isinstance(query, str) else query.get("query", "")

                if not sparql_query:
                    raise ValueError("SPARQL query is required")

                result = rdf_tools.run_sparql(sparql_query, limit=limit)

                # Format result
                if result.get("success"):
                    bindings = result.get("results", [])
                    return {
                        "results": bindings,
                        "result_count": result.get("result_count", 0),
                        "variables": result.get("variables", [])
                    }
                else:
                    raise ValueError(result.get("error", "SPARQL query execution failed"))

            else:
                raise ValueError(f"Execution not implemented for {database_type}")

        except Exception as e:
            logger.error("Error executing query", error=str(e))
            raise

    async def process_natural_language_query(
        self,
        nl_query: str,
        target_database: Optional[str] = None,
        use_cache: bool = True,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """Process a natural language query end-to-end.

        This is the main entry point that orchestrates the entire pipeline:
        1. Detect target database (if not specified)
        2. Gather schema context
        3. Generate query using LLM
        4. Validate query (optional)
        5. Execute query
        6. Return results with explanation

        Args:
            nl_query: Natural language query from user
            target_database: Optional specific database to target
            use_cache: Whether to use cached results if available (default: True)
            skip_validation: Skip query validation (useful for cross-database comparison)

        Returns:
            Dictionary containing:
            - query: Generated database query
            - results: Query execution results
            - explanation: Natural language explanation
            - metadata: Additional information about the process
            - cached: Boolean indicating if result was from cache

        Example:
            >>> results = await engine.process_natural_language_query(
            ...     "Show me all users older than 25"
            ... )
            >>> print(results['explanation'])
            >>> print(f"Found {results['result_count']} users")
        """
        # Check query cache first
        if use_cache and self.enable_cache and self.query_cache:
            cached_result = self.query_cache.get(nl_query, target_database or "auto")
            if cached_result is not None:
                logger.info("Using cached query result", nl_query=nl_query[:50])
                cached_result["cached"] = True
                return cached_result

        try:
            start_time = time.time()

            # Step 1: Detect target database
            if target_database:
                database_type = target_database
                confidence = 1.0
                alternatives = None
            else:
                database_type, confidence, alternatives = await self.detect_target_database(nl_query)

            logger.info(
                "Processing query",
                nl_query=nl_query,
                target_database=database_type,
                confidence=confidence
            )

            # Step 2: Gather schema context
            context = await self.gather_schema_context(database_type)

            # Step 3: Generate query
            query_result = await self.generate_query(nl_query, database_type, context)

            # Extract query components based on database type
            if database_type == "mongodb":
                database = query_result.get("database")
                collection = query_result.get("collection")
                operation = query_result.get("operation", "find")
                query = query_result.get("query")
                explanation = query_result.get("explanation", "")
            elif database_type == "neo4j":
                database = None  # Neo4j doesn't use database selection the same way
                collection = None
                operation = "cypher"
                query = query_result.get("query")  # This is the Cypher string
                explanation = query_result.get("explanation", "")
            elif database_type == "redis":
                database = None  # Redis doesn't use database selection the same way
                collection = None
                operation = query_result.get("command", "get_key")  # Redis MCP tool name

                # Handle both old command format and new MCP tool format
                command = query_result.get("command", "")
                args = query_result.get("args", {})

                # Convert to MCP tool call format
                query = {
                    "tool": command,
                    "arguments": args if isinstance(args, dict) else {}
                }
                explanation = query_result.get("explanation", "")
            elif database_type == "hbase":
                database = None  # HBase doesn't use database selection
                collection = query_result.get("table", "")  # Table name
                operation = query_result.get("operation", "scan_table")  # HBase operation
                query = query_result  # Full query dict with operation, table, row_key, filter_column, etc.
                explanation = query_result.get("explanation", "")
                
                # Log the generated HBase query for debugging
                logger.info(
                    "Generated HBase query",
                    table=collection,
                    operation=operation,
                    has_filter=bool(query.get("filter_column")),
                    has_range=bool(query.get("row_start") or query.get("row_stop"))
                )
            elif database_type == "rdf":
                database = None  # RDF doesn't use database selection
                collection = None  # RDF doesn't use collections
                operation = "sparql"  # Always SPARQL for RDF
                query = query_result.get("query", "")  # SPARQL query string
                explanation = query_result.get("explanation", "")
            else:
                raise ValueError(f"Unsupported database type: {database_type}")

            # Step 4: Validate query (skip for Redis, HBase, RDF, MongoDB aggregations, or if explicitly disabled)
            if skip_validation:
                # Skip validation for cross-database comparison or when explicitly requested
                validation = {"valid": True}
            elif database_type in ["redis", "hbase", "rdf"]:
                # Redis, HBase, and RDF validation happens during execution
                validation = {"valid": True}
            elif database_type == "mongodb" and operation == "aggregate":
                # MongoDB aggregation validation happens during execution
                validation = {"valid": True}
            elif database_type == "neo4j":
                validation = await self.validate_query(
                    database_type,
                    None,
                    None,
                    query  # For Neo4j, query is already a string
                )
            else:
                validation = await self.validate_query(
                    database_type,
                    database,
                    collection,
                    json.dumps(query)
                )

            if not validation.get("valid", False):
                error_msg = validation.get("error", "Query validation failed")
                logger.warning("Query validation failed", error=error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "query": query,
                    "explanation": explanation
                }

            # Step 5: Execute query
            execution_result = await self.execute_query(
                database_type,
                database,
                collection,
                query,
                operation
            )

            # Step 6: Format response
            execution_time_ms = (time.time() - start_time) * 1000
            response = {
                "success": True,
                "database_type": database_type,
                "database": database,
                "collection": collection,
                "operation": operation,
                "query": query,
                "results": execution_result.get("results", []),
                "result_count": execution_result.get("result_count", 0),
                "explanation": explanation,
                "confidence": confidence,
                "validated": validation.get("valid", False),
                "validation_message": validation.get("message", "Query validated successfully"),
                "cached": False,
                "execution_time_ms": execution_time_ms
            }

            logger.info(
                "Query processed successfully",
                result_count=response["result_count"],
                execution_time_ms=execution_time_ms
            )

            # Cache the successful result
            if use_cache and self.enable_cache and self.query_cache:
                self.query_cache.set(response, nl_query, target_database or "auto")

            # Add to history
            if self.enable_history and self.history:
                self.history.add_query(
                    nl_query=nl_query,
                    database_type=database_type,
                    database_name=database,
                    collection=collection,
                    operation=operation,
                    generated_query=query,
                    success=True,
                    result_count=response["result_count"],
                    execution_time_ms=execution_time_ms,
                    confidence=confidence,
                    cached=False
                )

            return response

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0

            logger.error("Error processing natural language query", error=str(e))

            # Add failed query to history
            if self.enable_history and self.history:
                self.history.add_query(
                    nl_query=nl_query,
                    database_type=target_database or "unknown",
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time_ms
                )

            return {
                "success": False,
                "error": str(e),
                "query": nl_query
            }

    async def process_multi_database_query(
        self,
        nl_query: str,
        target_databases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a natural language query across multiple databases in parallel.

        This method allows a single NL query to be executed on multiple databases
        simultaneously, enabling cross-database comparison and analysis.

        Args:
            nl_query: Natural language query from user
            target_databases: Optional list of database types to query.
                            If None, queries all connected databases.

        Returns:
            Dictionary containing:
            - success: Overall success status
            - results: Dict mapping database_type to query results
            - errors: Dict mapping database_type to error messages (if any)
            - summary: Aggregated statistics across all databases

        Example:
            >>> results = await engine.process_multi_database_query(
            ...     "Find all users",
            ...     target_databases=["mongodb", "neo4j"]
            ... )
            >>> print(f"MongoDB: {results['results']['mongodb']['result_count']} users")
            >>> print(f"Neo4j: {results['results']['neo4j']['result_count']} users")
        """
        try:
            # Determine which databases to query
            if target_databases is None:
                target_databases = self.mcp_manager.get_connected_servers()

            if not target_databases:
                return {
                    "success": False,
                    "error": "No databases connected",
                    "results": {},
                    "errors": {}
                }

            logger.info(
                "Processing multi-database query",
                nl_query=nl_query,
                target_databases=target_databases
            )

            # Create tasks for parallel execution
            tasks = []
            db_names = []
            for db_type in target_databases:
                if self.mcp_manager.is_connected(db_type):
                    task = self.process_natural_language_query(nl_query, target_database=db_type)
                    tasks.append(task)
                    db_names.append(db_type)
                else:
                    logger.warning(f"Database '{db_type}' not connected, skipping")

            # Execute all queries in parallel
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            results = {}
            errors = {}
            total_results = 0
            successful_queries = 0

            for db_name, result in zip(db_names, results_list):
                if isinstance(result, Exception):
                    errors[db_name] = str(result)
                    logger.error(f"Error querying {db_name}", error=str(result))
                elif isinstance(result, dict) and not result.get("success", False):
                    errors[db_name] = result.get("error", "Unknown error")
                else:
                    results[db_name] = result
                    total_results += result.get("result_count", 0)
                    successful_queries += 1

            # Create summary
            summary = {
                "total_databases_queried": len(db_names),
                "successful_queries": successful_queries,
                "failed_queries": len(errors),
                "total_results": total_results,
                "databases_with_results": [
                    db for db, res in results.items()
                    if res.get("result_count", 0) > 0
                ]
            }

            response = {
                "success": successful_queries > 0,
                "nl_query": nl_query,
                "results": results,
                "errors": errors,
                "summary": summary
            }

            logger.info(
                "Multi-database query completed",
                successful=successful_queries,
                failed=len(errors),
                total_results=total_results
            )

            return response

        except Exception as e:
            logger.error("Error processing multi-database query", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "nl_query": nl_query,
                "results": {},
                "errors": {}
            }

    def clear_cache(self, cache_type: str = "all") -> None:
        """Clear the query engine caches.

        Args:
            cache_type: Type of cache to clear ("query", "schema", or "all")
        """
        if not self.enable_cache:
            logger.warning("Caching is disabled")
            return

        if cache_type in ["query", "all"] and self.query_cache:
            self.query_cache.clear()
            logger.info("Query cache cleared")

        if cache_type in ["schema", "all"] and self.schema_cache:
            self.schema_cache.clear()
            logger.info("Schema cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the caches.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_cache:
            return {"caching_enabled": False}

        stats = {"caching_enabled": True}

        if self.query_cache:
            stats["query_cache"] = self.query_cache.stats()

        if self.schema_cache:
            stats["schema_cache"] = self.schema_cache.stats()

        return stats

    async def explain_query(
        self,
        query: str,
        database_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate natural language explanation of a database query using LLM.

        This method uses the database-specific prompts to generate explanations
        for queries in any of the supported database types.

        Args:
            query: The database query to explain (string or JSON)
            database_type: Type of database ("mongodb", "neo4j", "redis", "hbase", "rdf")
            context: Optional context (database name, collection, table, etc.)

        Returns:
            Natural language explanation of what the query does

        Example:
            >>> explanation = await engine.explain_query(
            ...     query='{"age": {"$gt": 25}}',
            ...     database_type="mongodb",
            ...     context={"collection": "users"}
            ... )
        """
        try:
            # Build explanation prompt based on database type
            if database_type == "mongodb":
                collection = context.get("collection", "documents") if context else "documents"
                prompt = f"""Explain this MongoDB query in simple, clear natural language.

Query: {query}
Collection: {collection}

Provide a concise explanation of:
1. What documents will be found/modified
2. What conditions/filters are applied
3. What operations are performed (if any aggregation)

Keep the explanation simple and non-technical. Respond with ONLY the explanation text, no additional formatting."""

            elif database_type == "neo4j":
                prompt = f"""Explain this Cypher query in simple, clear natural language.

Query: {query}

Provide a concise explanation of:
1. What nodes/relationships are being matched
2. What conditions are applied
3. What is being returned or modified

Keep the explanation simple and non-technical. Respond with ONLY the explanation text, no additional formatting."""

            elif database_type == "redis":
                prompt = f"""Explain this Redis command in simple, clear natural language.

Command: {query}

Provide a concise explanation of:
1. What operation is being performed
2. What key(s) are affected
3. What the expected result is

Keep the explanation simple and non-technical. Respond with ONLY the explanation text, no additional formatting."""

            elif database_type == "hbase":
                table = context.get("table", "table") if context else "table"
                row_key = context.get("row_key") if context else None
                operation = context.get("operation") if context else None
                filter_column = context.get("filter_column") if context else None
                filter_value = context.get("filter_value") if context else None

                # Build detailed context for HBase queries
                context_parts = [f"Table: {table}"]

                if row_key:
                    context_parts.append(f"Row Key: {row_key}")
                    context_parts.append("Operation: Get specific row")
                elif filter_column and filter_value:
                    context_parts.append(f"Operation: {operation or 'Scan'}")
                    context_parts.append(f"Filter: {filter_column} = {filter_value}")
                else:
                    context_parts.append(f"Operation: {operation or 'Scan all rows'}")

                context_str = "\n".join(context_parts)

                prompt = f"""Explain this HBase operation in simple, clear natural language.

{context_str}
Query: {query}

Provide a concise explanation of:
1. What data will be retrieved from the table
2. What filters or conditions are applied
3. What the operation does

Keep the explanation simple and non-technical. Respond with ONLY the explanation text, no additional formatting."""

            elif database_type == "rdf":
                prompt = f"""Explain this SPARQL query in simple, clear natural language.

Query: {query}

Provide a concise explanation of:
1. What RDF triples are being matched
2. What patterns are being searched
3. What data is being returned

Keep the explanation simple and non-technical. Respond with ONLY the explanation text, no additional formatting."""

            else:
                return f"Explanation not available for database type: {database_type}"

            # Call LLM for explanation
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a database query explanation assistant. Provide clear, concise explanations in natural language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            explanation = response.choices[0].message.content.strip()

            logger.info(
                "Query explanation generated",
                database_type=database_type,
                query_length=len(query)
            )

            return explanation

        except Exception as e:
            logger.error(
                "Error generating query explanation",
                error=str(e),
                database_type=database_type
            )
            return f"Unable to generate explanation: {str(e)}"
