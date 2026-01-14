"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # LLM API Configuration
    groq_api_key: str = Field(..., description="Groq API key for LLM calls")

    # MongoDB Configuration
    mongodb_uri: Optional[str] = Field(default=None, description="MongoDB connection URI (takes precedence over individual settings)")
    mongodb_host: str = Field(default="localhost", description="MongoDB host")
    mongodb_port: int = Field(default=27017, description="MongoDB port")
    mongodb_username: str = Field(default="admin", description="MongoDB username")
    mongodb_password: str = Field(default="adminpass", description="MongoDB password")
    mongodb_database: str = Field(default="test_db", description="MongoDB default database")
    mongodb_auth_source: str = Field(default="admin", description="MongoDB auth source")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password123", description="Neo4j password")

    # Redis Configuration
    redis_uri: Optional[str] = Field(default=None, description="Redis connection URI")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default="redispass123", description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")

    # HBase Configuration
    hbase_uri: Optional[str] = Field(default=None, description="HBase Thrift connection URI")
    hbase_host: str = Field(default="localhost", description="HBase host")
    hbase_port: int = Field(default=9090, description="HBase Thrift port")
    hbase_thrift_protocol: str = Field(default="binary", description="HBase Thrift protocol (compact or binary)")

    # RDF/Fuseki Configuration
    fuseki_endpoint: str = Field(
        default="http://localhost:3030",
        description="Fuseki SPARQL endpoint"
    )
    fuseki_dataset: str = Field(default="ds", description="Fuseki dataset name")
    fuseki_username: Optional[str] = Field(default="admin", description="Fuseki username")
    fuseki_password: Optional[str] = Field(default="admin123", description="Fuseki password")

    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_query_results: int = Field(default=100, description="Maximum query results to return")
    query_timeout: int = Field(default=30, description="Query timeout in seconds")

    # MCP Server Configuration
    mcp_server_timeout: int = Field(default=60, description="MCP server timeout in seconds")
    mcp_max_retries: int = Field(default=3, description="Maximum MCP retry attempts")

    def get_mongodb_uri(self) -> str:
        """Get or construct MongoDB connection URI."""
        if self.mongodb_uri:
            return self.mongodb_uri

        # Construct URI from components
        return (
            f"mongodb://{self.mongodb_username}:{self.mongodb_password}"
            f"@{self.mongodb_host}:{self.mongodb_port}/"
            f"?authSource={self.mongodb_auth_source}"
        )

    def get_redis_uri(self) -> str:
        """Get or construct Redis connection URI."""
        if self.redis_uri:
            return self.redis_uri

        # Construct URI from components
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def get_hbase_uri(self) -> str:
        """Get or construct HBase Thrift connection URI."""
        if self.hbase_uri:
            return self.hbase_uri

        # Construct URI from components
        # Format: hbase+thrift://host:port?protocol=compact
        return f"hbase+thrift://{self.hbase_host}:{self.hbase_port}?protocol={self.hbase_thrift_protocol}"

    def get_sparql_endpoint(self) -> str:
        """Get full SPARQL endpoint URL."""
        # Format: http://localhost:3030/{dataset}/sparql
        return f"{self.fuseki_endpoint}/{self.fuseki_dataset}/sparql"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()
