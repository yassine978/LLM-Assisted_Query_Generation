"""MCP Client Manager for managing connections to multiple MCP servers.

This module provides the MCPManager class which handles:
- Registration of MCP server configurations
- Connecting to and disconnecting from MCP servers
- Tool discovery and invocation
- Resource access
- Multi-server management
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fallback mode - use tools.py directly if MCP protocol fails
USE_DIRECT_TOOLS = True  # Set to False to use full MCP protocol


class MCPManager:
    """Manages connections to multiple MCP servers.

    This class provides a centralized interface for:
    - Registering MCP server configurations
    - Establishing and managing connections
    - Discovering available tools
    - Invoking tools on connected servers
    - Accessing MCP resources

    Attributes:
        servers: Dictionary mapping server names to their configurations
        connections: Dictionary mapping server names to active MCP client sessions
        tools_cache: Dictionary caching tool schemas by server name
    """

    def __init__(self):
        """Initialize the MCP Manager."""
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, Any] = {}
        self.tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("MCPManager initialized")

    def register_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> None:
        """Register an MCP server configuration.

        Args:
            name: Unique name for the server (e.g., "mongodb", "neo4j")
            command: Command to start the MCP server (e.g., "python", "node")
            args: List of arguments for the command
            env: Optional environment variables for the server

        Raises:
            ValueError: If a server with the same name is already registered

        Example:
            manager.register_server(
                name="mongodb",
                command="python",
                args=["-m", "src.mcp_servers.mongodb_mcp.server"]
            )
        """
        if name in self.servers:
            raise ValueError(f"Server '{name}' is already registered")

        self.servers[name] = {
            "command": command,
            "args": args or [],
            "env": env or {}
        }

        logger.info("Server registered", server_name=name, command=command)

    async def connect_server(self, name: str) -> bool:
        """Connect to a registered MCP server.

        Args:
            name: Name of the server to connect to

        Returns:
            True if connection successful, False otherwise

        Raises:
            ValueError: If server is not registered

        Note:
            This method uses the MCP Python client to establish a stdio-based
            connection to the server process.
        """
        if name not in self.servers:
            raise ValueError(f"Server '{name}' is not registered")

        if name in self.connections:
            logger.warning("Server already connected", server_name=name)
            return True

        # Use direct tools mode if enabled (bypass MCP protocol)
        if USE_DIRECT_TOOLS and name in ["mongodb", "neo4j", "redis", "hbase"]:
            logger.info("Using direct tools mode (bypassing MCP protocol)", server_name=name)
            try:
                if name == "mongodb":
                    from src.mcp_servers.mongodb_mcp import tools
                elif name == "neo4j":
                    from src.mcp_servers.neo4j_mcp import tools
                elif name == "redis":
                    from src.mcp_servers.redis_mcp import tools
                elif name == "hbase":
                    from src.mcp_servers.hbase_mcp import tools

                # Store tools module as "connection"
                self.connections[name] = {
                    "type": "direct",
                    "tools": tools,
                    "database_type": name
                }
                logger.info("Direct tools connection successful", server_name=name)
                return True
            except Exception as e:
                logger.error("Failed to import tools module", server_name=name, error=str(e))
                return False

        try:
            # Import MCP client here to avoid import errors if not installed
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_config = self.servers[name]
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config["env"] if server_config["env"] else None
            )

            logger.info("Starting MCP server subprocess", server_name=name)

            # Create stdio transport using async context manager
            # Note: We store the context manager to properly close it later
            stdio_context = stdio_client(server_params)

            # Add timeout to prevent hanging
            try:
                read_stream, write_stream = await asyncio.wait_for(
                    stdio_context.__aenter__(),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for server to start", server_name=name)
                return False

            logger.info("Server subprocess started", server_name=name)

            # Wait a moment for server to fully initialize
            await asyncio.sleep(3.0)
            logger.info("Waited for server initialization", server_name=name)

            # Create client session
            session = ClientSession(read_stream, write_stream)

            # Initialize the session with timeout
            logger.info("Initializing MCP session", server_name=name)
            try:
                result = await asyncio.wait_for(
                    session.initialize(),
                    timeout=30.0  # 30 second timeout (increased for debugging)
                )
                logger.info("Session initialized successfully", server_name=name, result=str(result)[:200])
            except asyncio.TimeoutError:
                logger.error("Timeout during session initialization", server_name=name)
                # Clean up the context
                try:
                    await stdio_context.__aexit__(None, None, None)
                except:
                    pass
                return False

            # Store the connection with context manager for cleanup
            self.connections[name] = {
                "session": session,
                "read_stream": read_stream,
                "write_stream": write_stream,
                "context": stdio_context
            }

            logger.info("Successfully connected to server", server_name=name)
            return True

        except Exception as e:
            logger.error("Failed to connect to server", server_name=name, error=str(e))
            import traceback
            logger.error("Connection error traceback", traceback=traceback.format_exc())
            return False

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            name: Name of the server to disconnect from
        """
        if name not in self.connections:
            logger.warning("Server not connected", server_name=name)
            return

        try:
            # Clean up connection
            connection = self.connections[name]

            # Properly exit the context manager if it exists
            if "context" in connection:
                try:
                    await connection["context"].__aexit__(None, None, None)
                except Exception as ctx_error:
                    logger.warning("Error closing context", error=str(ctx_error))

            del self.connections[name]
            if name in self.tools_cache:
                del self.tools_cache[name]

            logger.info("Disconnected from server", server_name=name)

        except Exception as e:
            logger.error("Error disconnecting from server", server_name=name, error=str(e))

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List all available tools from a connected MCP server.

        Args:
            server_name: Name of the server to query

        Returns:
            List of tool schemas

        Raises:
            ValueError: If server is not connected
        """
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' is not connected")

        # Check cache first
        if server_name in self.tools_cache:
            return self.tools_cache[server_name]

        # Handle direct tools mode
        connection = self.connections[server_name]
        if connection.get("type") == "direct":
            logger.info("Listing tools in direct mode", server_name=server_name)

            # Get database-specific tools
            db_type = connection.get("database_type", server_name)

            if db_type == "mongodb":
                tool_list = [
                    {"name": "ping", "description": "Test MongoDB connection"},
                    {"name": "list_databases", "description": "List all databases in MongoDB"},
                    {"name": "list_collections", "description": "List all collections in a database"},
                    {"name": "get_collection_schema", "description": "Infer schema from sample documents"},
                    {"name": "execute_query", "description": "Execute a MongoDB query"},
                    {"name": "validate_query", "description": "Validate a MongoDB query without executing"},
                    {"name": "get_indexes", "description": "Get all indexes for a collection"},
                    {"name": "aggregate", "description": "Execute an aggregation pipeline"},
                    {"name": "clear_schema_cache", "description": "Clear schema cache entries"}
                ]
            elif db_type == "neo4j":
                tool_list = [
                    {"name": "get_node_labels", "description": "List all node labels in Neo4j"},
                    {"name": "get_relationship_types", "description": "List all relationship types"},
                    {"name": "get_schema", "description": "Get comprehensive graph schema"},
                    {"name": "get_node_properties", "description": "Get properties for a specific node label"},
                    {"name": "get_relationship_properties", "description": "Get properties for a relationship type"},
                    {"name": "run_cypher", "description": "Execute a Cypher query"},
                    {"name": "explain_cypher", "description": "Get execution plan for a Cypher query"},
                    {"name": "validate_cypher", "description": "Validate Cypher query syntax"}
                ]
            elif db_type == "redis":
                tool_list = [
                    {"name": "ping", "description": "Test Redis connection"},
                    {"name": "get_db_info", "description": "Get Redis database information and statistics"},
                    {"name": "get_key", "description": "Get value for a Redis key"},
                    {"name": "set_key", "description": "Set a string key in Redis"},
                    {"name": "delete_key", "description": "Delete a Redis key"},
                    {"name": "get_keys", "description": "Get Redis keys matching a pattern"},
                    {"name": "get_schema", "description": "Get comprehensive Redis schema information"},
                    {"name": "clear_schema_cache", "description": "Clear the Redis schema cache"},
                    {"name": "execute_command", "description": "Execute a raw Redis command"}
                ]
            elif db_type == "hbase":
                tool_list = [
                    {"name": "ping", "description": "Test HBase connection"},
                    {"name": "list_tables", "description": "List all tables in HBase"},
                    {"name": "get_table_info", "description": "Get information about a table"},
                    {"name": "get_row", "description": "Get a specific row from a table"},
                    {"name": "scan_table", "description": "Scan rows from a table"},
                    {"name": "put_row", "description": "Insert/update a row in a table"},
                    {"name": "delete_row", "description": "Delete a row from a table"},
                    {"name": "get_schema", "description": "Infer HBase schema by analyzing tables"}
                ]
            else:
                tool_list = []

            self.tools_cache[server_name] = tool_list
            logger.info("Listed tools (direct mode)", server_name=server_name, tool_count=len(tool_list))
            return tool_list

        try:
            session = connection["session"]

            # List tools using MCP protocol
            tools_result = await session.list_tools()
            tools = tools_result.tools if hasattr(tools_result, 'tools') else []

            # Convert to dict format
            tool_list = [
                {
                    "name": tool.name,
                    "description": tool.description if hasattr(tool, 'description') else "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
                for tool in tools
            ]

            # Cache the results
            self.tools_cache[server_name] = tool_list

            logger.info("Listed tools", server_name=server_name, tool_count=len(tool_list))
            return tool_list

        except Exception as e:
            logger.error("Error listing tools", server_name=server_name, error=str(e))
            return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a tool on a connected MCP server.

        Args:
            server_name: Name of the server hosting the tool
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution results

        Raises:
            ValueError: If server is not connected
        """
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' is not connected")

        # Handle direct tools mode
        connection = self.connections[server_name]
        if connection.get("type") == "direct":
            logger.info("Calling tool in direct mode", server_name=server_name, tool_name=tool_name)
            try:
                tools_module = connection["tools"]
                tool_func = getattr(tools_module, tool_name)

                # Special handling for Redis execute_command which uses *args
                if tool_name == "execute_command" and server_name == "redis" and arguments:
                    command = arguments.get("command")
                    args = arguments.get("args", [])
                    result = tool_func(command, *args)
                # Call the function with arguments
                elif arguments:
                    result = tool_func(**arguments)
                else:
                    result = tool_func()

                logger.info("Direct tool call successful", server_name=server_name, tool_name=tool_name)
                return result
            except Exception as e:
                logger.error("Direct tool call failed", server_name=server_name, tool_name=tool_name, error=str(e))
                raise

        try:
            session = connection["session"]

            # Call the tool using MCP protocol
            result = await session.call_tool(tool_name, arguments or {})

            # Extract content from result
            if hasattr(result, 'content') and result.content:
                # MCP returns content as a list of content items
                content_items = result.content
                if content_items:
                    # Get the first content item (typically text)
                    first_item = content_items[0]
                    if hasattr(first_item, 'text'):
                        import json
                        # Parse the JSON text content
                        return json.loads(first_item.text)

            logger.info(
                "Tool called successfully",
                server_name=server_name,
                tool_name=tool_name
            )
            return {}

        except Exception as e:
            logger.error(
                "Error calling tool",
                server_name=server_name,
                tool_name=tool_name,
                error=str(e)
            )
            raise

    async def get_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Get a resource from a connected MCP server.

        Args:
            server_name: Name of the server hosting the resource
            uri: Resource URI (e.g., "mongodb://test_db/users/schema")

        Returns:
            Resource data

        Raises:
            ValueError: If server is not connected
        """
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' is not connected")

        try:
            session = self.connections[server_name]["session"]

            # Read the resource using MCP protocol
            result = await session.read_resource(uri)

            # Extract content from result
            if hasattr(result, 'contents') and result.contents:
                content_items = result.contents
                if content_items:
                    first_item = content_items[0]
                    if hasattr(first_item, 'text'):
                        import json
                        return json.loads(first_item.text)

            logger.info(
                "Resource retrieved",
                server_name=server_name,
                uri=uri
            )
            return {}

        except Exception as e:
            logger.error(
                "Error getting resource",
                server_name=server_name,
                uri=uri,
                error=str(e)
            )
            raise

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered servers.

        Returns:
            Dictionary mapping server names to connection success status
        """
        results = {}
        for name in self.servers.keys():
            results[name] = await self.connect_server(name)

        successful = sum(1 for status in results.values() if status)
        logger.info(
            "Connected to servers",
            total=len(results),
            successful=successful
        )
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        server_names = list(self.connections.keys())
        for name in server_names:
            await self.disconnect_server(name)

        logger.info("Disconnected from all servers")

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names.

        Returns:
            List of connected server names
        """
        return list(self.connections.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is currently connected.

        Args:
            server_name: Name of the server to check

        Returns:
            True if connected, False otherwise
        """
        return server_name in self.connections
