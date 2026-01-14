"""Cross-Database Comparison Module.

This module enables comparing how the same natural language query translates
and executes across different database types (MongoDB, Neo4j, Redis, HBase, RDF).

Features:
- Parallel query generation for all databases
- Syntax comparison across query languages
- Result set comparison
- Performance metrics comparison
- Comprehensive comparison reports
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of executing a query on a specific database."""
    database_type: str
    query: Any  # Query can be dict, string, or other format
    query_str: str  # String representation for comparison
    success: bool
    execution_time_ms: float
    result_count: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class ComparisonResult:
    """Complete comparison across all databases."""
    nl_query: str
    timestamp: datetime
    databases_compared: List[str]
    query_results: Dict[str, QueryResult]
    syntax_comparison: Dict[str, Any] = field(default_factory=dict)
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    result_comparison: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class CrossDatabaseComparator:
    """Compare query execution across multiple database types."""

    def __init__(self, query_engine):
        """Initialize the comparator.

        Args:
            query_engine: QueryEngine instance for executing queries
        """
        self.query_engine = query_engine
        self.supported_databases = ["mongodb", "neo4j", "redis", "hbase", "rdf"]

    async def compare_query(
        self,
        nl_query: str,
        databases: Optional[List[str]] = None,
        limit: int = 10
    ) -> ComparisonResult:
        """Compare the same natural language query across multiple databases.

        Args:
            nl_query: Natural language query
            databases: List of database types to compare (default: all supported)
            limit: Maximum results to return per database

        Returns:
            ComparisonResult with complete comparison data
        """
        if databases is None:
            databases = self.supported_databases

        logger.info(
            "Starting cross-database comparison",
            nl_query=nl_query,
            databases=databases
        )

        # Execute queries in parallel for all databases
        query_results = await self._execute_parallel(nl_query, databases, limit)

        # Perform comparisons
        syntax_comparison = self._compare_syntax(query_results)
        performance_comparison = self._compare_performance(query_results)
        result_comparison = self._compare_results(query_results)

        # Generate summary
        summary = self._generate_summary(
            query_results,
            syntax_comparison,
            performance_comparison,
            result_comparison
        )

        comparison = ComparisonResult(
            nl_query=nl_query,
            timestamp=datetime.now(),
            databases_compared=databases,
            query_results=query_results,
            syntax_comparison=syntax_comparison,
            performance_comparison=performance_comparison,
            result_comparison=result_comparison,
            summary=summary
        )

        logger.info(
            "Cross-database comparison complete",
            nl_query=nl_query,
            successful_dbs=len([r for r in query_results.values() if r.success])
        )

        return comparison

    async def _execute_parallel(
        self,
        nl_query: str,
        databases: List[str],
        limit: int
    ) -> Dict[str, QueryResult]:
        """Execute query on all databases in parallel.

        Args:
            nl_query: Natural language query
            databases: List of database types
            limit: Result limit

        Returns:
            Dictionary mapping database type to QueryResult
        """
        # Create tasks for parallel execution
        tasks = [
            self._execute_on_database(nl_query, db_type, limit)
            for db_type in databases
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results to database types
        query_results = {}
        for db_type, result in zip(databases, results):
            if isinstance(result, Exception):
                # Handle exception
                query_results[db_type] = QueryResult(
                    database_type=db_type,
                    query=None,
                    query_str="",
                    success=False,
                    execution_time_ms=0.0,
                    result_count=0,
                    error=str(result)
                )
            else:
                query_results[db_type] = result

        return query_results

    async def _execute_on_database(
        self,
        nl_query: str,
        database_type: str,
        limit: int
    ) -> QueryResult:
        """Execute query on a specific database.

        Args:
            nl_query: Natural language query
            database_type: Target database type
            limit: Result limit

        Returns:
            QueryResult with execution details
        """
        start_time = time.time()

        try:
            # Execute query through query engine
            # Note: Skip validation for cross-database comparison since we're demonstrating
            # syntax translation, not validating against actual database schemas
            result = await self.query_engine.process_natural_language_query(
                nl_query=nl_query,
                target_database=database_type,
                use_cache=False,
                skip_validation=True  # Skip schema validation for syntax demonstration
            )

            execution_time_ms = (time.time() - start_time) * 1000

            # Extract query and convert to string
            query = result.get("query")
            if isinstance(query, str):
                query_str = query
            elif isinstance(query, dict):
                query_str = str(query)
            elif isinstance(query, list):
                query_str = str(query)
            else:
                query_str = str(query)

            return QueryResult(
                database_type=database_type,
                query=query,
                query_str=query_str,
                success=result.get("success", True),
                execution_time_ms=execution_time_ms,
                result_count=result.get("result_count", 0),
                results=result.get("results", [])[:limit],  # Limit results
                explanation=result.get("explanation")
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "Error executing query on database",
                database_type=database_type,
                error=str(e)
            )
            return QueryResult(
                database_type=database_type,
                query=None,
                query_str="",
                success=False,
                execution_time_ms=execution_time_ms,
                result_count=0,
                error=str(e)
            )

    def _compare_syntax(
        self,
        query_results: Dict[str, QueryResult]
    ) -> Dict[str, Any]:
        """Compare query syntax across databases.

        Args:
            query_results: Dictionary of query results

        Returns:
            Syntax comparison data
        """
        comparison = {
            "queries": {},
            "syntax_types": {},
            "complexity": {}
        }

        for db_type, result in query_results.items():
            if result.success:
                comparison["queries"][db_type] = result.query_str

                # Determine syntax type
                if db_type == "mongodb":
                    syntax_type = "JSON-based query"
                elif db_type == "neo4j":
                    syntax_type = "Cypher (graph query language)"
                elif db_type == "redis":
                    syntax_type = "Command-based"
                elif db_type == "hbase":
                    syntax_type = "Row-key based operations"
                elif db_type == "rdf":
                    syntax_type = "SPARQL (semantic query language)"
                else:
                    syntax_type = "Unknown"

                comparison["syntax_types"][db_type] = syntax_type

                # Estimate query complexity (based on string length as proxy)
                comparison["complexity"][db_type] = len(result.query_str)

        return comparison

    def _compare_performance(
        self,
        query_results: Dict[str, QueryResult]
    ) -> Dict[str, Any]:
        """Compare performance metrics across databases.

        Args:
            query_results: Dictionary of query results

        Returns:
            Performance comparison data
        """
        successful_results = {
            db: result for db, result in query_results.items()
            if result.success
        }

        if not successful_results:
            return {"error": "No successful query executions"}

        execution_times = {
            db: result.execution_time_ms
            for db, result in successful_results.items()
        }

        # Find fastest and slowest
        fastest_db = min(execution_times, key=execution_times.get)
        slowest_db = max(execution_times, key=execution_times.get)

        # Calculate average
        avg_time = sum(execution_times.values()) / len(execution_times)

        return {
            "execution_times_ms": execution_times,
            "fastest": {
                "database": fastest_db,
                "time_ms": execution_times[fastest_db]
            },
            "slowest": {
                "database": slowest_db,
                "time_ms": execution_times[slowest_db]
            },
            "average_time_ms": avg_time,
            "performance_ranking": sorted(
                execution_times.items(),
                key=lambda x: x[1]
            )
        }

    def _compare_results(
        self,
        query_results: Dict[str, QueryResult]
    ) -> Dict[str, Any]:
        """Compare result sets across databases.

        Args:
            query_results: Dictionary of query results

        Returns:
            Result comparison data
        """
        successful_results = {
            db: result for db, result in query_results.items()
            if result.success
        }

        if not successful_results:
            return {"error": "No successful query executions"}

        result_counts = {
            db: result.result_count
            for db, result in successful_results.items()
        }

        # Check if counts are consistent
        unique_counts = set(result_counts.values())
        counts_consistent = len(unique_counts) == 1

        return {
            "result_counts": result_counts,
            "counts_consistent": counts_consistent,
            "unique_count_values": list(unique_counts),
            "total_results": sum(result_counts.values()),
            "note": "Result counts may differ due to different data models and schemas"
        }

    def _generate_summary(
        self,
        query_results: Dict[str, QueryResult],
        syntax_comparison: Dict[str, Any],
        performance_comparison: Dict[str, Any],
        result_comparison: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of comparison.

        Args:
            query_results: Query results
            syntax_comparison: Syntax comparison
            performance_comparison: Performance comparison
            result_comparison: Result comparison

        Returns:
            Summary string
        """
        successful = [db for db, r in query_results.items() if r.success]
        failed = [db for db, r in query_results.items() if not r.success]

        summary_parts = []

        # Success rate
        summary_parts.append(
            f"Successfully executed on {len(successful)}/{len(query_results)} databases"
        )

        if successful:
            summary_parts.append(f"Successful: {', '.join(successful)}")

        if failed:
            summary_parts.append(f"Failed: {', '.join(failed)}")

        # Performance
        if "fastest" in performance_comparison:
            fastest = performance_comparison["fastest"]
            summary_parts.append(
                f"Fastest: {fastest['database']} ({fastest['time_ms']:.2f}ms)"
            )

        # Results
        if "result_counts" in result_comparison:
            counts = result_comparison["result_counts"]
            if counts:
                summary_parts.append(
                    f"Results returned: {', '.join(f'{db}={count}' for db, count in counts.items())}"
                )

        return " | ".join(summary_parts)

    def format_comparison_report(
        self,
        comparison: ComparisonResult,
        include_queries: bool = True,
        include_results: bool = False
    ) -> str:
        """Format comparison result as human-readable report.

        Args:
            comparison: ComparisonResult to format
            include_queries: Whether to include full queries
            include_results: Whether to include result samples

        Returns:
            Formatted report string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("CROSS-DATABASE COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Natural Language Query: \"{comparison.nl_query}\"")
        lines.append(f"Timestamp: {comparison.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Databases Compared: {', '.join(comparison.databases_compared)}")
        lines.append("")

        # Summary
        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(comparison.summary)
        lines.append("")

        # Generated Queries
        if include_queries and "queries" in comparison.syntax_comparison:
            lines.append("-" * 80)
            lines.append("GENERATED QUERIES")
            lines.append("-" * 80)
            for db, query in comparison.syntax_comparison["queries"].items():
                lines.append(f"\n{db.upper()}:")
                lines.append(f"  Type: {comparison.syntax_comparison['syntax_types'].get(db, 'Unknown')}")
                lines.append(f"  Query: {query[:200]}{'...' if len(query) > 200 else ''}")

            lines.append("")

        # Performance Comparison
        if "execution_times_ms" in comparison.performance_comparison:
            lines.append("-" * 80)
            lines.append("PERFORMANCE COMPARISON")
            lines.append("-" * 80)
            perf = comparison.performance_comparison

            lines.append(f"  Average Execution Time: {perf.get('average_time_ms', 0):.2f}ms")
            lines.append(f"  Fastest: {perf['fastest']['database']} ({perf['fastest']['time_ms']:.2f}ms)")
            lines.append(f"  Slowest: {perf['slowest']['database']} ({perf['slowest']['time_ms']:.2f}ms)")

            lines.append("\n  Ranking (fastest to slowest):")
            for i, (db, time_ms) in enumerate(perf["performance_ranking"], 1):
                lines.append(f"    {i}. {db}: {time_ms:.2f}ms")

            lines.append("")

        # Result Comparison
        if "result_counts" in comparison.result_comparison:
            lines.append("-" * 80)
            lines.append("RESULT COMPARISON")
            lines.append("-" * 80)
            result_comp = comparison.result_comparison

            lines.append(f"  Total Results: {result_comp.get('total_results', 0)}")
            lines.append(f"  Counts Consistent: {'Yes' if result_comp.get('counts_consistent') else 'No'}")

            lines.append("\n  Result Counts by Database:")
            for db, count in result_comp["result_counts"].items():
                lines.append(f"    {db}: {count} results")

            if result_comp.get("note"):
                lines.append(f"\n  Note: {result_comp['note']}")

            lines.append("")

        # Errors (if any)
        errors = [
            (db, result.error)
            for db, result in comparison.query_results.items()
            if not result.success and result.error
        ]

        if errors:
            lines.append("-" * 80)
            lines.append("ERRORS")
            lines.append("-" * 80)
            for db, error in errors:
                lines.append(f"{db}: {error}")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)
