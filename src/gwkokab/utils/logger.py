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


def log_gwkokab_info() -> None:
    """Prints the GWKokab version and the Python version."""

    import gwkokab as gwk

    logger.info("=" * 60)
    logger.info("GWKokab INFO")
    logger.info("=" * 60)

    logger.info("GWKokab version: {gwk_version}", gwk_version=gwk.__version__)
    logger.info("Python version: {python_version}", python_version=sys.version)
    logger.info("Python platform: {python_platform}", python_platform=sys.platform)
    logger.info("Python build: {python_build}", python_build=sys.version_info)


def log_device_info() -> None:
    """Prints the device information."""
    import jax
    import jaxlib
    from jax.lib import xla_bridge

    logger.info("=" * 60)
    logger.info("JAX CUDA ENVIRONMENT INFO")
    logger.info("=" * 60)

    logger.info("Devices count: {n_devices}", n_devices=jax.device_count())
    logger.info("Devices: {devices}", devices=jax.devices())

    logger.info("jax version: {jax_version}", jax_version=jax.__version__)
    logger.info("jaxlib version: {jaxlib_version}", jaxlib_version=jaxlib.__version__)

    try:
        backend = xla_bridge.get_backend()
        logger.info("JAX platform: {}", backend.platform)
        logger.info("JAX backend: {}", type(backend).__name__)

        platform_version = getattr(backend, "platform_version", None)
        if platform_version:
            logger.info("JAX platform version: {}", platform_version)
        else:
            logger.warning("No platform version info found on backend.")
    except Exception as e:
        logger.warning("Could not retrieve CUDA info from XLA backend: {}", e)

    for device in jax.devices():
        logger.info(
            "Device: {}, process_index: {}, id: {}, platform: {}",
            device.device_kind,
            device.process_index,
            device.id,
            device.platform,
        )
