import logging
import logging.config
from pathlib import Path

from core.config import settings


def setup_logging() -> None:
    """
    Configure application logging.
    """

    log_directory = Path(settings.log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s | "
                    "%(levelname)s | "
                    "%(name)s | "
                    "%(filename)s:%(lineno)d | "
                    "%(message)s"
                )
            },
            "access": {
                "format": (
                    "%(asctime)s | "
                    "%(levelname)s | "
                    "%(client_addr)s - "
                    "\"%(request_line)s\" "
                    "%(status_code)s"
                )
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "default",
                "filename": str(log_directory / "app.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """
    Return configured logger instance.
    """
    return logging.getLogger(name)