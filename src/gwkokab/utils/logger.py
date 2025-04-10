# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import sys

from loguru import logger


def set_log_level(log_level: str) -> None:
    """Set the log level for the logger. The preset log level when initialising GWKokab
    is the value of the GWKOKAB_LOG_LEVEL environment variable, or 'WARNING' if the
    environment variable is unset.

    Parameters
    ----------
    log_level : str
        The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING',
        'ERROR', 'CRITICAL'
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>",
    )
    logger.debug(f"Setting LogLevel to {log_level}")
