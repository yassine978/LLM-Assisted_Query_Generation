"""Streamlit Frontend for LLM-Assisted NoSQL Query Generation.

This application provides a web interface for:
1. Natural Language Query Translation across multiple NoSQL databases
2. Database Schema and Metadata Exploration
3. Query Validation and Explanation
4. Cross-Database Comparison

Architecture:
- Frontend (Streamlit) -> MCP Client -> Query Engine -> MCP Servers -> Databases
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import src
from src.main_app.mcp_manager import MCPManager
from src.main_app.query_engine import QueryEngine
from src.main_app.cross_db_compare import CrossDatabaseComparator

# Page configuration
st.set_page_config(
    page_title="NoSQL Query Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state."""
    if 'mcp_manager' not in st.session_state:
        st.session_state.mcp_manager = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'comparator' not in st.session_state:
        st.session_state.comparator = None
    if 'connected_databases' not in st.session_state:
        st.session_state.connected_databases = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


async def initialize_mcp_connection():
    """Initialize MCP Manager and connect to database servers."""
    try:
        # Create MCP Manager
        mcp_manager = MCPManager()

        # Register all database MCP servers
        databases = {
            "mongodb": ["python", "-m", "src.mcp_servers.mongodb_mcp.server"],
            "neo4j": ["python", "-m", "src.mcp_servers.neo4j_mcp.server"],
            "redis": ["python", "-m", "src.mcp_servers.redis_mcp.server"],
            "hbase": ["python", "-m", "src.mcp_servers.hbase_mcp.server"],
        }

        connected = []
        for db_name, cmd_args in databases.items():
            mcp_manager.register_server(
                name=db_name,
                command=cmd_args[0],
                args=cmd_args[1:]
            )
            success = await mcp_manager.connect_server(db_name)
            if success:
                connected.append(db_name)

        # RDF doesn't use MCP - it's direct integration
        connected.append("rdf")

        # Create Query Engine and Comparator
        query_engine = QueryEngine(mcp_manager)
        comparator = CrossDatabaseComparator(query_engine)

        return mcp_manager, query_engine, comparator, connected

    except Exception as e:
        st.error(f"Error initializing MCP connection: {str(e)}")
        return None, None, None, []


def render_sidebar():
    """Render the sidebar with navigation and database status."""
    st.sidebar.markdown("## 🔍 NoSQL Query Assistant")
    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select Feature",
        [
            "🏠 Home",
            "💬 Natural Language Query",
            "🗂️ Schema Explorer",
            "✅ Query Validation",
            "⚖️ Cross-Database Comparison"
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Database Status
    st.sidebar.markdown("### Database Status")
    if st.session_state.connected_databases:
        for db in ["mongodb", "neo4j", "redis", "hbase", "rdf"]:
            if db in st.session_state.connected_databases:
                st.sidebar.success(f"✓ {db.upper()}")
            else:
                st.sidebar.error(f"✗ {db.upper()}")
    else:
        st.sidebar.warning("Not connected")

    st.sidebar.markdown("---")

    # Connection button
    if st.sidebar.button("🔄 Reconnect to Databases"):
        with st.spinner("Connecting to databases..."):
            result = asyncio.run(initialize_mcp_connection())
            if result[0]:
                st.session_state.mcp_manager = result[0]
                st.session_state.query_engine = result[1]
                st.session_state.comparator = result[2]
                st.session_state.connected_databases = result[3]
                st.sidebar.success(f"Connected to {len(result[3])} databases!")
                st.rerun()

    return page


def render_home_page():
    """Render the home page."""
    st.markdown('<div class="main-header">🔍 LLM-Assisted NoSQL Query Generation</div>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the NoSQL Query Assistant! This application helps you interact with multiple NoSQL databases
    using natural language queries powered by Large Language Models.
    """)

    # Project Objectives
    st.markdown("### 🎯 Project Objectives")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>1️⃣ Natural Language Query Translation</h4>
        Write queries in plain English and get them translated to:<br>
        • MongoDB (JSON queries)<br>
        • Neo4j (Cypher)<br>
        • Redis (Commands)<br>
        • HBase (Row operations)<br>
        • RDF (SPARQL)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h4>2️⃣ Schema & Metadata Exploration</h4>
        Explore database structures:<br>
        • View collections/tables<br>
        • Examine field types<br>
        • Check indexes and constraints<br>
        • See sample data
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>3️⃣ Query Validation & Explanation</h4>
        Understand your queries:<br>
        • Validate syntax before execution<br>
        • Get field-level explanations<br>
        • Map queries to schema<br>
        • Receive performance hints
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <h4>4️⃣ Cross-Database Comparison</h4>
        Compare across databases:<br>
        • Same query, multiple databases<br>
        • Syntax comparison<br>
        • Performance metrics<br>
        • Result analysis
        </div>
        """, unsafe_allow_html=True)

    # Architecture
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    <div class="info-box">
    <b>Model Context Protocol (MCP) Architecture:</b><br><br>
    Frontend (Streamlit) → MCP Client (Query Engine) → MCP Servers (MongoDB, Neo4j, Redis, HBase) → Databases<br><br>
    <i>Note: RDF uses direct integration for simplicity</i>
    </div>
    """, unsafe_allow_html=True)

    # Quick Start
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Check Database Status** in the sidebar
    2. **Choose a Feature** from the navigation menu
    3. **Enter your query** or explore schemas
    4. **View results** and explanations
    """)

    # Statistics
    if st.session_state.query_history:
        st.markdown("### 📊 Session Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Queries Executed", len(st.session_state.query_history))

        with col2:
            successful = sum(1 for q in st.session_state.query_history if q.get('success'))
            st.metric("Successful Queries", successful)

        with col3:
            databases_used = set()
            for q in st.session_state.query_history:
                databases_used.add(q.get('database', 'unknown'))
            st.metric("Databases Used", len(databases_used))


def render_nl_query_page():
    """Render the Natural Language Query page."""
    st.markdown('<div class="main-header">💬 Natural Language Query Translation</div>', unsafe_allow_html=True)

    st.markdown("""
    Enter a query in plain English, select your target database, and watch as the LLM translates
    it into the appropriate query language!
    """)

    # Database selection
    col1, col2 = st.columns([2, 1])

    with col1:
        nl_query = st.text_area(
            "Enter your natural language query:",
            height=100,
            placeholder="Example: Find all users who registered in 2023"
        )

    with col2:
        database = st.selectbox(
            "Target Database:",
            ["mongodb", "neo4j", "redis", "hbase", "rdf"],
            format_func=str.upper
        )

    # Execute button
    if st.button("🚀 Execute Query", type="primary"):
        if not nl_query:
            st.warning("Please enter a query!")
            return

        if not st.session_state.query_engine:
            st.error("Query engine not initialized. Please reconnect to databases.")
            return

        with st.spinner(f"Executing query on {database.upper()}..."):
            try:
                result = asyncio.run(
                    st.session_state.query_engine.process_natural_language_query(
                        nl_query=nl_query,
                        target_database=database,
                        use_cache=False
                    )
                )

                # Save to history
                st.session_state.query_history.append({
                    'nl_query': nl_query,
                    'database': database,
                    'success': result.get('success', True),
                    'timestamp': datetime.now()
                })

                # Display results
                st.markdown("### Results")

                # Generated Query
                st.markdown("#### Generated Query")
                query = result.get('query')

                # Format query based on database type
                if isinstance(query, (dict, list)):
                    # MongoDB or JSON-based queries
                    query_str = json.dumps(query, indent=2)
                    language = "json"
                elif database == "rdf":
                    # SPARQL query - format with proper line breaks
                    query_str = str(query)
                    # Add line breaks after common SPARQL keywords for better readability
                    keywords = ["PREFIX", "SELECT", "WHERE", "FILTER", "OPTIONAL", "UNION", "LIMIT", "ORDER BY", "GROUP BY"]
                    for keyword in keywords:
                        if keyword in query_str:
                            query_str = query_str.replace(f" {keyword} ", f"\n{keyword} ")
                    # Clean up opening braces
                    query_str = query_str.replace(" { ", " {\n  ").replace(" } ", "\n}")
                    language = "sparql"
                elif database == "neo4j":
                    # Cypher query
                    query_str = str(query)
                    language = "cypher"
                else:
                    # Other queries
                    query_str = str(query)
                    language = "sql"

                st.code(query_str, language=language)

                # Validation status
                if result.get('validated'):
                    st.success("✓ Query is valid!")
                    if result.get('validation_message'):
                        st.info(result['validation_message'])

                # Explanation
                if result.get('explanation'):
                    st.markdown("#### Explanation")
                    st.info(result['explanation'])

                # Results
                st.markdown(f"#### Query Results ({result.get('result_count', 0)} found)")

                if result.get('results'):
                    # Display as JSON (show first 10 results)
                    st.json(result['results'][:10])
                else:
                    st.warning("No results found")

                # Success message
                if result.get('success', True):
                    st.success("✓ Query executed successfully!")

            except Exception as e:
                st.error(f"Error executing query: {str(e)}")


def render_schema_explorer_page():
    """Render the Schema Explorer page."""
    st.markdown('<div class="main-header">🗂️ Database Schema Explorer</div>', unsafe_allow_html=True)

    st.markdown("""
    Explore database structures, view schemas, and examine sample data.
    """)

    # Database selection (only databases with schema exploration)
    database = st.selectbox(
        "Select Database:",
        ["mongodb", "neo4j", "redis", "hbase", "rdf"],
        format_func=str.upper
    )

    if not st.session_state.mcp_manager:
        st.warning("Please connect to databases first")
        return

    # MongoDB Schema
    if database == "mongodb":
        render_mongodb_schema()

    # Neo4j Schema
    elif database == "neo4j":
        render_neo4j_schema()

    # Redis Schema
    elif database == "redis":
        render_redis_schema()

    # HBase Schema
    elif database == "hbase":
        render_hbase_schema()

    # RDF Schema
    elif database == "rdf":
        render_rdf_schema()


def render_mongodb_schema():
    """Render MongoDB schema information."""
    st.markdown("### MongoDB Schema")

    try:
        # List databases
        databases = asyncio.run(
            st.session_state.mcp_manager.call_tool("mongodb", "list_databases", {})
        )

        db_name = st.selectbox("Select Database:", databases)

        # List collections
        collections = asyncio.run(
            st.session_state.mcp_manager.call_tool("mongodb", "list_collections", {
                "database": db_name
            })
        )

        collection_name = st.selectbox("Select Collection:", collections)

        # Get schema
        if st.button("Load Schema"):
            with st.spinner("Loading schema..."):
                schema = asyncio.run(
                    st.session_state.mcp_manager.call_tool("mongodb", "get_collection_schema", {
                        "database": db_name,
                        "collection": collection_name,
                        "sample_size": 100,
                        "use_cache": False
                    })
                )

                # Display schema
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Document Count", schema.get('document_count', 0))
                    st.metric("Fields", len(schema.get('fields', {})))

                with col2:
                    st.metric("Sample Size", schema.get('sample_size', 0))
                    st.metric("Indexes", len(schema.get('statistics', {}).get('indexes', [])))

                # Fields
                st.markdown("#### Fields")
                for field, info in schema.get('fields', {}).items():
                    with st.expander(f"📄 {field}"):
                        st.write(f"**Types:** {', '.join(info.get('types', []))}")
                        st.write(f"**Frequency:** {info.get('frequency', 0)}%")
                        if info.get('nested_schema'):
                            st.json(info['nested_schema'])

                # Sample documents
                if schema.get('sample_documents'):
                    st.markdown("#### Sample Documents")
                    st.json(schema['sample_documents'][:3])

    except Exception as e:
        st.error(f"Error loading MongoDB schema: {str(e)}")


def render_neo4j_schema():
    """Render Neo4j schema information."""
    st.markdown("### Neo4j Graph Schema")

    try:
        # Get schema
        schema = asyncio.run(
            st.session_state.mcp_manager.call_tool("neo4j", "get_schema", {})
        )

        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", schema.get('nodeCount', 0))
        with col2:
            st.metric("Total Relationships", schema.get('relationshipCount', 0))
        with col3:
            st.metric("Property Keys", len(schema.get('propertyKeys', [])))

        # Node labels with counts
        st.markdown("#### Node Labels")
        labels = schema.get('labels', [])
        label_counts = schema.get('labelCounts', {})

        for label in labels:
            count = label_counts.get(label, 0)
            with st.expander(f"🔵 {label} ({count} nodes)"):
                props = asyncio.run(
                    st.session_state.mcp_manager.call_tool("neo4j", "get_node_properties", {
                        "label": label
                    })
                )
                st.json(props)

        # Relationship types with counts
        st.markdown("#### Relationship Types")
        rel_types = schema.get('relationshipTypes', [])
        rel_counts = schema.get('relationshipTypeCounts', {})

        for rel_type in rel_types:
            count = rel_counts.get(rel_type, 0)
            with st.expander(f"➡️ {rel_type} ({count} relationships)"):
                props = asyncio.run(
                    st.session_state.mcp_manager.call_tool("neo4j", "get_relationship_properties", {
                        "rel_type": rel_type
                    })
                )
                st.json(props)

        # Graph visualization data
        if schema.get('graphVisualization'):
            st.markdown("#### Graph Structure")
            viz = schema['graphVisualization']
            st.write(f"**Nodes:** {', '.join(viz.get('nodes', []))}")
            if viz.get('edges'):
                st.write("**Connections:**")
                for edge in viz['edges'][:10]:  # Show first 10
                    st.write(f"  - {edge['from']} --[{edge['type']}]--> {edge['to']}")

    except Exception as e:
        st.error(f"Error loading Neo4j schema: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_redis_schema():
    """Render Redis schema information."""
    st.markdown("### Redis Key Patterns & Schema")

    try:
        # Get schema
        schema = asyncio.run(
            st.session_state.mcp_manager.call_tool("redis", "get_schema", {})
        )

        # Display database statistics
        db_info = schema.get('database_info', {})
        if db_info:
            st.markdown("#### Database Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Keys", db_info.get('keys_count', 0))
            with col2:
                st.metric("Redis Version", db_info.get('redis_version', 'N/A'))
            with col3:
                st.metric("Memory Used", db_info.get('used_memory', 'N/A'))
            with col4:
                st.metric("Connected Clients", db_info.get('connected_clients', 0))

        # Display type distribution
        st.markdown("#### Type Distribution")
        type_dist = schema.get('type_distribution', {})
        if type_dist:
            import pandas as pd
            df = pd.DataFrame(list(type_dist.items()), columns=['Type', 'Count'])
            st.bar_chart(df.set_index('Type'))

        # Display key patterns
        st.markdown("#### Key Patterns")
        patterns = schema.get('key_patterns', [])
        if patterns:
            import pandas as pd
            df = pd.DataFrame(patterns)
            st.dataframe(df, use_container_width=True)

            # Show pattern details
            st.markdown("#### Pattern Details")
            pattern_details = schema.get('pattern_details', {})
            selected_pattern = st.selectbox("Select a pattern to view details:", list(pattern_details.keys()))

            if selected_pattern:
                details = pattern_details[selected_pattern]
                st.write(f"**Count:** {details.get('count', 0)}")
                st.write(f"**Types:** {', '.join(details.get('types', {}).keys())}")

                st.write("**Examples:**")
                for example in details.get('examples', [])[:5]:
                    st.json(example)

    except Exception as e:
        st.error(f"Error loading Redis schema: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_hbase_schema():
    """Render HBase schema information."""
    st.markdown("### HBase Table Schema")

    try:
        # List tables
        tables_result = asyncio.run(
            st.session_state.mcp_manager.call_tool("hbase", "list_tables", {})
        )

        # Extract table names
        if isinstance(tables_result, dict) and "tables" in tables_result:
            tables = tables_result["tables"]
        elif isinstance(tables_result, list):
            tables = tables_result
        else:
            tables = []

        if tables:
            st.markdown("#### Available Tables")
            st.write(f"**Total Tables:** {len(tables)}")

            # Display tables
            for table in tables:
                st.write(f"- {table}")

            st.info("💡 Note: HBase is schema-less with flexible column families. Detailed schema information (column families, qualifiers) can be obtained by scanning sample data from each table.")

            # Option to scan sample data
            selected_table = st.selectbox("Select a table to scan sample data:", tables)

            if st.button("Scan Sample Data"):
                with st.spinner(f"Scanning {selected_table}..."):
                    scan_result = asyncio.run(
                        st.session_state.mcp_manager.call_tool("hbase", "scan_table", {
                            "table_name": selected_table,
                            "limit": 5
                        })
                    )
                    st.markdown(f"#### Sample Data from `{selected_table}`")
                    st.json(scan_result)
        else:
            st.warning("No tables found in HBase")

    except Exception as e:
        st.error(f"Error loading HBase schema: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_rdf_schema():
    """Render RDF ontology information."""
    st.markdown("### RDF Ontology")

    try:
        from src.main_app import rdf_tools

        # Get ontology
        ontology = rdf_tools.get_ontology()

        if ontology.get('success'):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Classes", ontology.get('class_count', 0))

            with col2:
                st.metric("Properties", ontology.get('property_count', 0))

            # Classes
            st.markdown("#### Classes")
            for cls in ontology.get('classes', [])[:10]:
                st.write(f"- {cls.get('class', '')} ({cls.get('count', 0)} instances)")

            # Properties
            st.markdown("#### Properties")
            for prop in ontology.get('properties', [])[:10]:
                st.write(f"- {prop.get('property', '')} ({prop.get('count', 0)} uses)")

    except Exception as e:
        st.error(f"Error loading RDF ontology: {str(e)}")


def render_validation_page():
    """Render the Query Validation page."""
    st.markdown('<div class="main-header">✅ Query Validation & Explanation</div>', unsafe_allow_html=True)

    st.markdown("""
    Validate your queries and get detailed explanations of how they map to the database schema.
    """)

    database = st.selectbox(
        "Select Database:",
        ["mongodb", "neo4j", "redis", "hbase", "rdf"],
        format_func=str.upper
    )

    # Database-specific inputs
    if database == "mongodb":
        col1, col2 = st.columns(2)
        with col1:
            db_name = st.text_input("Database Name:", value="sample_mflix")
        with col2:
            collection_name = st.text_input("Collection Name:", value="movies")
        query_placeholder = '{"year": {"$gte": 2020}}'
    elif database == "neo4j":
        query_placeholder = 'MATCH (n:Person) WHERE n.age > 25 RETURN n'
    elif database == "redis":
        query_placeholder = 'GET user:1001  (or any Redis command)'
    elif database == "hbase":
        col1, col2 = st.columns(2)
        with col1:
            table_name = st.text_input("Table Name:", value="users")
        with col2:
            row_key = st.text_input("Row Key (optional):", value="")
        query_placeholder = 'Enter row key, or JSON query like: {"operation": "scan_table", "filter_column": "preferences:theme", "filter_value": "dark"}'
    elif database == "rdf":
        query_placeholder = 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10'

    query_input = st.text_area(
        "Enter Query:",
        height=150,
        placeholder=query_placeholder
    )

    if st.button("Validate Query"):
        if not query_input:
            st.warning("Please enter a query")
            return

        with st.spinner("Validating query..."):
            try:
                validation_result = None

                if database == "mongodb":
                    # Validate MongoDB query - support both find queries and aggregation pipelines
                    try:
                        # Test if it's valid JSON
                        parsed_query = json.loads(query_input)

                        # Check if it's an aggregation pipeline (array) or find query (object)
                        if isinstance(parsed_query, list):
                            # Aggregation pipeline - just validate JSON structure
                            validation_result = {
                                "valid": True,
                                "message": f"Valid MongoDB aggregation pipeline with {len(parsed_query)} stages",
                                "query_type": "aggregation"
                            }
                        elif isinstance(parsed_query, dict):
                            # Find query - call validation tool
                            validation_result = asyncio.run(
                                st.session_state.mcp_manager.call_tool("mongodb", "validate_query", {
                                    "database": db_name,
                                    "collection": collection_name,
                                    "query": query_input
                                })
                            )
                            validation_result["query_type"] = "find"
                        else:
                            validation_result = {
                                "valid": False,
                                "error": "MongoDB query must be a JSON object {} or array [] for aggregation"
                            }
                    except json.JSONDecodeError as e:
                        validation_result = {
                            "valid": False,
                            "error": f"Invalid JSON format: {str(e)}"
                        }

                elif database == "neo4j":
                    # Validate Cypher query
                    validation_result = asyncio.run(
                        st.session_state.mcp_manager.call_tool("neo4j", "validate_cypher", {
                            "query": query_input
                        })
                    )

                elif database == "redis":
                    # Redis validation - support both raw commands and JSON MCP tool format
                    try:
                        # Try to parse as JSON (MCP tool format)
                        redis_query = json.loads(query_input)
                        if isinstance(redis_query, dict) and "tool" in redis_query:
                            tool_name = redis_query.get("tool", "")
                            valid_tools = ["get_key", "set_key", "delete_key", "get_keys", "get_schema",
                                         "get_db_info", "ping", "execute_command"]
                            if tool_name in valid_tools:
                                validation_result = {
                                    "valid": True,
                                    "message": f"Valid Redis MCP tool: {tool_name}"
                                }
                            else:
                                validation_result = {
                                    "valid": False,
                                    "error": f"Unknown Redis tool: {tool_name}. Valid tools: {', '.join(valid_tools)}"
                                }
                        else:
                            validation_result = {
                                "valid": False,
                                "error": "Redis JSON query must have 'tool' field"
                            }
                    except json.JSONDecodeError:
                        # Not JSON, treat as raw Redis command
                        parts = query_input.strip().split(None, 1)
                        if parts:
                            command = parts[0].upper()
                            valid_commands = ["GET", "SET", "DEL", "HGET", "HGETALL", "LRANGE", "SMEMBERS",
                                            "ZRANGE", "KEYS", "SCAN", "INFO", "PING"]
                            if command in valid_commands:
                                validation_result = {
                                    "valid": True,
                                    "message": f"Valid Redis command: {command}"
                                }
                            else:
                                validation_result = {
                                    "valid": False,
                                    "error": f"Unknown or unsupported Redis command: {command}"
                                }
                        else:
                            validation_result = {
                                "valid": False,
                                "error": "Empty command"
                            }

                elif database == "hbase":
                    # HBase validation - check if table exists
                    if table_name:
                        # List tables to check if it exists
                        tables_result = asyncio.run(
                            st.session_state.mcp_manager.call_tool("hbase", "list_tables", {})
                        )

                        if isinstance(tables_result, dict) and tables_result.get("success"):
                            tables = tables_result.get("tables", [])
                            if table_name in tables:
                                validation_result = {
                                    "valid": True,
                                    "message": f"Table '{table_name}' exists"
                                }
                            else:
                                validation_result = {
                                    "valid": False,
                                    "error": f"Table '{table_name}' not found. Available tables: {', '.join(tables)}"
                                }
                        else:
                            validation_result = {
                                "valid": False,
                                "error": "Could not list HBase tables"
                            }
                    else:
                        validation_result = {
                            "valid": False,
                            "error": "Table name is required"
                        }

                elif database == "rdf":
                    # RDF/SPARQL validation - basic syntax check
                    query_upper = query_input.strip().upper()
                    if any(keyword in query_upper for keyword in ["SELECT", "CONSTRUCT", "ASK", "DESCRIBE"]):
                        if "WHERE" in query_upper and "{" in query_input and "}" in query_input:
                            validation_result = {
                                "valid": True,
                                "message": "SPARQL query syntax appears valid"
                            }
                        else:
                            validation_result = {
                                "valid": False,
                                "error": "SPARQL query missing WHERE clause or triple patterns"
                            }
                    else:
                        validation_result = {
                            "valid": False,
                            "error": "SPARQL query must start with SELECT, CONSTRUCT, ASK, or DESCRIBE"
                        }

                # Display validation result
                if validation_result and validation_result.get('valid'):
                    st.success("✓ Query is valid!")
                    if validation_result.get('message'):
                        st.info(validation_result['message'])

                    # Generate and display explanation using QueryEngine
                    st.markdown("---")
                    st.markdown("### Query Explanation")

                    try:
                        # Build context for explanation
                        context = {}
                        if database == "mongodb":
                            context = {"collection": collection_name, "database": db_name}
                        elif database == "hbase":
                            context = {"table": table_name}
                            if row_key:
                                context["row_key"] = row_key
                            # Try to parse query as JSON to extract additional context
                            try:
                                hbase_query = json.loads(query_input)
                                if isinstance(hbase_query, dict):
                                    context["operation"] = hbase_query.get("operation", "scan")
                                    context["filter_column"] = hbase_query.get("filter_column")
                                    context["filter_value"] = hbase_query.get("filter_value")
                            except:
                                pass

                        # Use QueryEngine's explain_query method for LLM-powered explanation
                        if st.session_state.query_engine:
                            with st.spinner("Generating explanation..."):
                                explanation = asyncio.run(
                                    st.session_state.query_engine.explain_query(
                                        query=query_input,
                                        database_type=database,
                                        context=context
                                    )
                                )

                                st.markdown(f"""
                                <div class="info-box">
                                <h4>What this query does:</h4>
                                {explanation}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("Query engine not available for generating explanations")

                    except Exception as e:
                        st.warning(f"Could not generate explanation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

                else:
                    st.error("✗ Query is invalid!")
                    if validation_result and validation_result.get('error'):
                        st.error(validation_result['error'])

            except Exception as e:
                st.error(f"Error validating query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_comparison_page():
    """Render the Cross-Database Comparison page."""
    st.markdown('<div class="main-header">⚖️ Cross-Database Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    Compare how the same natural language query translates and executes across multiple databases!
    """)

    # Query input
    nl_query = st.text_area(
        "Enter your natural language query:",
        height=100,
        placeholder="Example: Find all items created in the last year"
    )

    # Database selection
    st.markdown("#### Select Databases to Compare")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        use_mongodb = st.checkbox("MongoDB", value=True)
    with col2:
        use_neo4j = st.checkbox("Neo4j", value=True)
    with col3:
        use_redis = st.checkbox("Redis", value=True)
    with col4:
        use_hbase = st.checkbox("HBase", value=True)
    with col5:
        use_rdf = st.checkbox("RDF", value=True)

    databases = []
    if use_mongodb:
        databases.append("mongodb")
    if use_neo4j:
        databases.append("neo4j")
    if use_redis:
        databases.append("redis")
    if use_hbase:
        databases.append("hbase")
    if use_rdf:
        databases.append("rdf")

    # Compare button
    if st.button("🔄 Compare Across Databases", type="primary"):
        if not nl_query:
            st.warning("Please enter a query!")
            return

        if not databases:
            st.warning("Please select at least one database!")
            return

        if not st.session_state.comparator:
            st.error("Comparator not initialized. Please reconnect to databases.")
            return

        with st.spinner(f"Comparing across {len(databases)} databases..."):
            try:
                comparison = asyncio.run(
                    st.session_state.comparator.compare_query(
                        nl_query=nl_query,
                        databases=databases
                    )
                )

                # Display comparison results
                st.markdown("### Comparison Results")

                # Summary
                st.markdown(f"**Summary:** {comparison.summary}")

                # Generated Queries
                st.markdown("### Generated Queries")
                for db, query_result in comparison.query_results.items():
                    with st.expander(f"📝 {db.upper()}", expanded=True):
                        if query_result.success:
                            st.code(query_result.query_str, language="json")
                            st.caption(f"Execution Time: {query_result.execution_time_ms:.2f}ms")
                            st.caption(f"Results: {query_result.result_count}")
                            if query_result.explanation:
                                st.info(query_result.explanation)
                        else:
                            st.error(f"Error: {query_result.error}")

                # Performance Comparison
                if comparison.performance_comparison.get('execution_times_ms'):
                    st.markdown("### Performance Comparison")
                    perf = comparison.performance_comparison

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Fastest",
                            perf['fastest']['database'].upper(),
                            f"{perf['fastest']['time_ms']:.2f}ms"
                        )
                    with col2:
                        st.metric(
                            "Slowest",
                            perf['slowest']['database'].upper(),
                            f"{perf['slowest']['time_ms']:.2f}ms"
                        )
                    with col3:
                        st.metric(
                            "Average",
                            f"{perf['average_time_ms']:.2f}ms"
                        )

                    # Performance chart
                    st.bar_chart({
                        db: time_ms
                        for db, time_ms in perf['execution_times_ms'].items()
                    })

                # Result Comparison
                if comparison.result_comparison.get('result_counts'):
                    st.markdown("### Result Counts")
                    result_comp = comparison.result_comparison

                    st.bar_chart(result_comp['result_counts'])

                    if not result_comp.get('counts_consistent'):
                        st.info(result_comp.get('note', ''))

                # Download report
                report = st.session_state.comparator.format_comparison_report(comparison)
                st.download_button(
                    "📥 Download Full Report",
                    report,
                    file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error comparing queries: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Auto-connect on first load
    if st.session_state.mcp_manager is None:
        with st.spinner("Initializing connection to databases..."):
            result = asyncio.run(initialize_mcp_connection())
            if result[0]:
                st.session_state.mcp_manager = result[0]
                st.session_state.query_engine = result[1]
                st.session_state.comparator = result[2]
                st.session_state.connected_databases = result[3]

    # Render sidebar
    page = render_sidebar()

    # Render selected page
    if "Home" in page:
        render_home_page()
    elif "Natural Language Query" in page:
        render_nl_query_page()
    elif "Schema Explorer" in page:
        render_schema_explorer_page()
    elif "Query Validation" in page:
        render_validation_page()
    elif "Cross-Database Comparison" in page:
        render_comparison_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    LLM-Assisted NoSQL Query Generation | Built with MCP Architecture | Powered by Groq
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
