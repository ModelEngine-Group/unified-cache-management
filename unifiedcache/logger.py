import logging
import os


def init_logger(name: str = "UNIFIED_CACHE") -> logging.Logger:
    log_level = os.getenv("UNIFIED_CACHE_LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    os.environ["UNIFIED_CACHE_LOG_LEVEL"] = "DEBUG"
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")