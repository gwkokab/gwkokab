# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from loguru import logger


def time_now() -> str:
    """Get the current time as a string in the format YYYYMMDDHHMMSS.

    Returns
    -------
    str
        The current time as a string.
    """
    import datetime

    return datetime.datetime.now().strftime(r"%Y%m%d%H%M%S")


def set_log_level() -> None:
    """Set the log level for the logger. The preset log level when initialising GWKokab
    is the value of the `GWKOKAB_LOG_LEVEL` environment variable, or 'WARNING' if the
    environment variable is unset.

    Valid options of `GWKOKAB_LOG_LEVEL` are 'TRACE', 'DEBUG', 'INFO', 'SUCCESS',
    'WARNING', 'ERROR', and 'CRITICAL'.
    """
    import os

    valid_levels = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")

    if (log_level := os.environ.get("GWKOKAB_LOG_LEVEL", "TRACE")) not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Valid options are {', '.join(valid_levels)}"
        )

    current_time = time_now()

    GWKOKAB_LOG_DIR = os.getenv("GWKOKAB_LOG_DIR", "./logs")
    GWKOKAB_LOG_FILE = os.getenv("GWKOKAB_LOG_FILE", f"gwkokab_{current_time}.log")
    log_filename = os.path.join(GWKOKAB_LOG_DIR, GWKOKAB_LOG_FILE)

    os.makedirs(GWKOKAB_LOG_DIR, exist_ok=True)

    logger.remove()
    logger.add(
        log_filename,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>",
    )
    logger.debug(f"Setting LogLevel to {log_level}")

    del os


def log_gwkokab_info() -> None:
    """Prints the GWKokab version and the Python version."""

    import sys

    import gwkokab as gwk

    logger.info("=" * 60)
    logger.info("GWKokab INFO")
    logger.info("=" * 60)

    logger.info("GWKokab version: {gwk_version}", gwk_version=gwk.__version__)
    logger.info("Python version: {python_version}", python_version=sys.version)
    logger.info("Python platform: {python_platform}", python_platform=sys.platform)
    logger.info("Python build: {python_build}", python_build=sys.version_info)

    del gwk
    del sys


def log_device_info() -> None:
    """Prints the device information."""
    import jax as _jax
    import jaxlib as _jaxlib
    from jax.extend.backend import get_backend

    logger.info("=" * 60)
    logger.info("JAX CUDA ENVIRONMENT INFO")
    logger.info("=" * 60)

    logger.info("Devices count: {n_devices}", n_devices=_jax.device_count())
    logger.info("Devices: {devices}", devices=_jax.devices())

    logger.info("jax version: {jax_version}", jax_version=_jax.__version__)
    logger.info("jaxlib version: {jaxlib_version}", jaxlib_version=_jaxlib.__version__)

    try:
        backend = get_backend()
        logger.info("JAX platform: {}", backend.platform)
        logger.info("JAX backend: {}", type(backend).__name__)

        platform_version = getattr(backend, "platform_version", None)
        if platform_version:
            logger.info("JAX platform version: {}", platform_version)
        else:
            logger.warning("No platform version info found on backend.")
    except Exception as e:
        logger.warning("Could not retrieve CUDA info from XLA backend: {}", e)

    for device in _jax.devices():
        logger.info(
            "Device: {}, process_index: {}, id: {}, platform: {}",
            device.device_kind,
            device.process_index,
            device.id,
            device.platform,
        )

    del _jax
    del _jaxlib
    del get_backend


def log_info(start: bool = False) -> None:
    """Log the information about the package and the device information.

    Parameters
    ----------
    start : bool, optional
        If True, log the start message, by default False
    """
    set_log_level()
    log_gwkokab_info()
    log_device_info()
    if start:
        logger.info("=" * 60)
        logger.info("GWKokab STARTING")
        logger.info("=" * 60)
