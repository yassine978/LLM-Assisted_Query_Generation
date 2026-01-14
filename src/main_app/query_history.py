"""Query History Storage - Tracks query execution history using SQLite.

This module provides the QueryHistory class which persists query execution
metadata and results for analysis and replay.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryHistory:
    """Manages query execution history using SQLite database.

    This class provides functionality to:
    - Store query execution records
    - Track success/failure metrics
    - Retrieve query history with filtering
    - Analyze query patterns

    Attributes:
        db_path: Path to SQLite database file
        connection: SQLite connection object
    """

    def __init__(self, db_path: str = "query_history.db"):
        """Initialize the Query History manager.

        Args:
            db_path: Path to SQLite database file (default: "query_history.db")
        """
        self.db_path = Path(db_path)
        self.connection = None
        self._initialize_database()
        logger.info("QueryHistory initialized", db_path=str(self.db_path))

    def _initialize_database(self) -> None:
        """Create the database and tables if they don't exist."""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name

        cursor = self.connection.cursor()

        # Create queries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                nl_query TEXT NOT NULL,
                database_type TEXT NOT NULL,
                database_name TEXT,
                collection TEXT,
                operation TEXT,
                generated_query TEXT,
                success INTEGER NOT NULL,
                error TEXT,
                result_count INTEGER,
                execution_time_ms REAL,
                confidence REAL,
                cached INTEGER DEFAULT 0
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON queries(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_database_type
            ON queries(database_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success
            ON queries(success)
        """)

        self.connection.commit()
        logger.debug("Database initialized")

    def add_query(
        self,
        nl_query: str,
        database_type: str,
        database_name: Optional[str] = None,
        collection: Optional[str] = None,
        operation: Optional[str] = None,
        generated_query: Optional[Any] = None,
        success: bool = True,
        error: Optional[str] = None,
        result_count: int = 0,
        execution_time_ms: Optional[float] = None,
        confidence: Optional[float] = None,
        cached: bool = False
    ) -> int:
        """Add a query execution record to history.

        Args:
            nl_query: Natural language query
            database_type: Type of database queried
            database_name: Name of database (optional)
            collection: Collection/table name (optional)
            operation: Operation type (find, aggregate, etc.)
            generated_query: The generated database query
            success: Whether query executed successfully
            error: Error message if failed
            result_count: Number of results returned
            execution_time_ms: Execution time in milliseconds
            confidence: Confidence score for database detection
            cached: Whether result was from cache

        Returns:
            ID of the inserted record
        """
        cursor = self.connection.cursor()

        # Convert generated_query to JSON string if it's a dict
        query_str = None
        if generated_query is not None:
            if isinstance(generated_query, (dict, list)):
                query_str = json.dumps(generated_query)
            else:
                query_str = str(generated_query)

        cursor.execute("""
            INSERT INTO queries (
                timestamp, nl_query, database_type, database_name,
                collection, operation, generated_query, success, error,
                result_count, execution_time_ms, confidence, cached
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            nl_query,
            database_type,
            database_name,
            collection,
            operation,
            query_str,
            1 if success else 0,
            error,
            result_count,
            execution_time_ms,
            confidence,
            1 if cached else 0
        ))

        self.connection.commit()
        query_id = cursor.lastrowid

        logger.info(
            "Query added to history",
            query_id=query_id,
            database_type=database_type,
            success=success
        )

        return query_id

    def get_query(self, query_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific query by ID.

        Args:
            query_id: ID of the query to retrieve

        Returns:
            Dictionary with query details, or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM queries WHERE id = ?", (query_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent queries.

        Args:
            limit: Maximum number of queries to return (default: 10)

        Returns:
            List of query dictionaries, most recent first
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM queries
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_queries_by_database(
        self,
        database_type: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get queries for a specific database type.

        Args:
            database_type: Type of database to filter by
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of query dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM queries
            WHERE database_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (database_type, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_failed_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get queries that failed.

        Args:
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of failed query dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM queries
            WHERE success = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def search_queries(
        self,
        search_term: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for queries by natural language text.

        Args:
            search_term: Term to search for in nl_query
            limit: Maximum number of queries to return (default: 20)

        Returns:
            List of matching query dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM queries
            WHERE nl_query LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{search_term}%", limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall query execution statistics.

        Returns:
            Dictionary with statistics including:
            - total_queries: Total number of queries
            - successful_queries: Number of successful queries
            - failed_queries: Number of failed queries
            - success_rate: Percentage of successful queries
            - queries_by_database: Count per database type
            - avg_result_count: Average number of results
            - cached_queries: Number of cached queries
        """
        cursor = self.connection.cursor()

        # Total queries
        cursor.execute("SELECT COUNT(*) as total FROM queries")
        total = cursor.fetchone()["total"]

        # Successful queries
        cursor.execute("SELECT COUNT(*) as successful FROM queries WHERE success = 1")
        successful = cursor.fetchone()["successful"]

        # Failed queries
        failed = total - successful

        # Success rate
        success_rate = (successful / total * 100) if total > 0 else 0

        # Queries by database
        cursor.execute("""
            SELECT database_type, COUNT(*) as count
            FROM queries
            GROUP BY database_type
        """)
        queries_by_db = {row["database_type"]: row["count"] for row in cursor.fetchall()}

        # Average result count
        cursor.execute("SELECT AVG(result_count) as avg_results FROM queries WHERE success = 1")
        avg_results = cursor.fetchone()["avg_results"] or 0

        # Cached queries
        cursor.execute("SELECT COUNT(*) as cached FROM queries WHERE cached = 1")
        cached = cursor.fetchone()["cached"]

        return {
            "total_queries": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "success_rate": round(success_rate, 2),
            "queries_by_database": queries_by_db,
            "avg_result_count": round(avg_results, 2),
            "cached_queries": cached,
            "cache_hit_rate": round((cached / total * 100) if total > 0 else 0, 2)
        }

    def clear_history(self, older_than_days: Optional[int] = None) -> int:
        """Clear query history.

        Args:
            older_than_days: If specified, only clear queries older than this many days.
                           If None, clears all history.

        Returns:
            Number of queries deleted
        """
        cursor = self.connection.cursor()

        if older_than_days is None:
            cursor.execute("DELETE FROM queries")
        else:
            cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            cursor.execute("DELETE FROM queries WHERE timestamp < ?", (cutoff_iso,))

        deleted_count = cursor.rowcount
        self.connection.commit()

        logger.info("History cleared", deleted_count=deleted_count)
        return deleted_count

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
