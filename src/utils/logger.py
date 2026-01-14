"""Logging configuration using structlog and colorlog."""

import logging
import sys
from typing import Optional

import colorlog
import structlog
from structlog.typing import FilteringBoundLogger

from .config import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure structured logging with color output.

    Args:
        log_level: Optional log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    settings = get_settings()
    level = log_level or settings.log_level

    # Configure standard logging
    log_level_int = getattr(logging, level.upper(), logging.INFO)

    # Create color formatter for console
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    # Configure root logger
    # Use stderr to avoid interfering with MCP protocol on stdout
    handler = colorlog.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level_int)

    # Configure structlog
    # Use stderr for all logging to avoid interfering with MCP protocol
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> FilteringBoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional logger name (usually __name__)

    Returns:
        FilteringBoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


# Initialize logging on module import
setup_logging()
