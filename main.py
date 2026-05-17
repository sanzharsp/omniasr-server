"""
Entry point for the Omnilingual-ASR FastAPI server.
"""

import logging
import os
from pathlib import Path

os.environ.setdefault(
    "FAIRSEQ2_USER_ASSET_DIR",
    str(Path(__file__).resolve().parent / "fairseq2-assets"),
)

import uvicorn

from app.config import OMNILINGUAL_HOST, OMNILINGUAL_PORT, OMNILINGUAL_ROOT_PATH
from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info(
        "Starting Omnilingual-ASR server",
        extra={
            "host": OMNILINGUAL_HOST,
            "port": OMNILINGUAL_PORT,
            "root_path": OMNILINGUAL_ROOT_PATH or "/",
        },
    )

    uvicorn.run(
        "app.server:app",
        host=OMNILINGUAL_HOST,
        port=OMNILINGUAL_PORT,
        reload=False,
        log_config=None,  # we already installed structured logging above
    )


if __name__ == "__main__":
    main()
