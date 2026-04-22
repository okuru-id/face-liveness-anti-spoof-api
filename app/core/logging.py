import logging
import sys

structlog_available = False
try:
    import structlog

    structlog_available = True
except ImportError:
    pass


def setup_logging() -> None:
    if structlog_available:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stdout,
        )


def get_logger(name: str):
    if structlog_available:
        return structlog.get_logger(name)
    return logging.getLogger(name)
