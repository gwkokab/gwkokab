# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auto-logging exception and warning registry.

Raising one of these must do three things at once: behave as the built-in exception it
mirrors, carry a message formatted from its arguments, and emit a loguru record
attributed to the *raising* site rather than to the mixin. Each is checked below.
"""

import warnings

import pytest
from loguru import logger

from gwkokab.utils import exceptions
from gwkokab.utils.exceptions import (
    LoggedAssertionError,
    LoggedAttributeError,
    LoggedDeprecationWarning,
    LoggedFileNotFoundError,
    LoggedImportError,
    LoggedIndexError,
    LoggedKeyError,
    LoggedMixinException,
    LoggedMixinWarning,
    LoggedNotImplementedError,
    LoggedRuntimeWarning,
    LoggedTypeError,
    LoggedUserWarning,
    LoggedValueError,
)


EXCEPTIONS = [
    (LoggedAssertionError, AssertionError),
    (LoggedAttributeError, AttributeError),
    (LoggedFileNotFoundError, FileNotFoundError),
    (LoggedImportError, ImportError),
    (LoggedIndexError, IndexError),
    (LoggedKeyError, KeyError),
    (LoggedNotImplementedError, NotImplementedError),
    (LoggedTypeError, TypeError),
    (LoggedValueError, ValueError),
]

WARNINGS = [
    (LoggedUserWarning, UserWarning),
    (LoggedRuntimeWarning, RuntimeWarning),
    (LoggedDeprecationWarning, DeprecationWarning),
]


@pytest.fixture
def records():
    """Collect the loguru records emitted inside a test."""
    collected = []
    handler_id = logger.add(
        lambda message: collected.append(message.record), level="WARNING"
    )
    yield collected
    logger.remove(handler_id)


@pytest.mark.parametrize(("logged", "builtin"), EXCEPTIONS)
def test_exception_subclasses_its_builtin(logged, builtin):
    """Callers catch ``ValueError``, not ``LoggedValueError``, so the bases matter."""
    assert issubclass(logged, builtin)
    assert issubclass(logged, LoggedMixinException)

    with pytest.raises(builtin):
        raise logged("boom")


@pytest.mark.parametrize(("logged", "builtin"), WARNINGS)
def test_warning_subclasses_its_builtin(logged, builtin):
    assert issubclass(logged, builtin)
    assert issubclass(logged, LoggedMixinWarning)

    with pytest.warns(builtin, match="careful"):
        warnings.warn(logged("careful"))


def test_positional_arguments_are_formatted():
    with pytest.raises(ValueError, match=r"expected 3, got 4"):
        raise LoggedValueError("expected {}, got {}", 3, 4)


def test_keyword_arguments_are_formatted():
    with pytest.raises(ValueError, match=r"seed must be non-negative, got -1"):
        raise LoggedValueError("seed must be non-negative, got {seed}", seed=-1)


def test_a_message_without_arguments_is_left_alone():
    """No arguments means no ``str.format`` call, so braces survive verbatim."""
    with pytest.raises(ValueError, match=r"\{not_a_field\}"):
        raise LoggedValueError("{not_a_field}")


def test_the_formatted_message_is_the_exception_argument():
    error = LoggedValueError("x = {x}", x=2)

    assert error.args == ("x = 2",)


def test_warning_message_is_formatted():
    with pytest.warns(UserWarning, match="dropping 2 events"):
        warnings.warn(LoggedUserWarning("dropping {n} events", n=2))


def test_exception_is_logged_at_error_level(records):
    with pytest.raises(ValueError):
        raise LoggedValueError("bad value {v}", v=7)

    assert [(r["level"].name, r["message"]) for r in records] == [
        ("ERROR", "bad value 7")
    ]


def test_warning_is_logged_at_warning_level(records):
    with pytest.warns(UserWarning):
        warnings.warn(LoggedUserWarning("watch out {n}", n=1))

    assert [(r["level"].name, r["message"]) for r in records] == [
        ("WARNING", "watch out 1")
    ]


def test_the_record_points_at_the_raising_site(records):
    """``depth=1`` is what keeps ``exceptions.py`` out of every log line."""
    with pytest.raises(ValueError):
        raise LoggedValueError("boom")

    (record,) = records
    assert record["function"] == "test_the_record_points_at_the_raising_site"
    assert record["name"] == __name__


def test_loguru_options_are_overridable(records):
    """``loguru_opt`` is forwarded to ``logger.opt``, e.g. to unwind one frame
    further.
    """

    def raise_it():
        raise LoggedValueError("boom", loguru_opt={"depth": 2})

    with pytest.raises(ValueError):
        raise_it()

    (record,) = records
    assert record["function"] == "test_loguru_options_are_overridable"


def test_only_one_record_per_instantiation(records):
    for i in range(3):
        LoggedValueError("attempt {i}", i=i)

    assert len(records) == 3


@pytest.mark.parametrize("name", exceptions.__all__)
def test_everything_exported_exists(name):
    assert hasattr(exceptions, name)
