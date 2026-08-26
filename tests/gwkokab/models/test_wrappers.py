# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.wrappers`."""

import pytest

from gwkokab.models.wrappers import ModelRegistry


def test_a_fresh_registry_is_empty():
    assert ModelRegistry().registry == {}


def test_register_with_a_single_string_parameter():
    registry = ModelRegistry()

    def model(parameter):
        return f"built {parameter}"

    # a bare string is normalised to a one-element tuple
    assert registry.register("mass_1_source", model) is model
    assert registry.registry == {("mass_1_source",): model}


def test_register_with_a_tuple_of_parameters():
    registry = ModelRegistry()

    def model(parameter):
        return f"built {parameter}"

    registry.register(("mass_1_source", "mass_2_source"), model)
    assert registry.registry == {("mass_1_source", "mass_2_source"): model}


def test_register_as_a_decorator():
    registry = ModelRegistry()

    @registry.register("redshift")
    def model(parameter):
        return f"built {parameter}"

    assert registry.registry == {("redshift",): model}


def test_registering_the_same_parameter_twice_overwrites():
    registry = ModelRegistry()

    def first(parameter):
        return "first"

    def second(parameter):
        return "second"

    registry.register("redshift", first)
    registry.register("redshift", second)
    assert registry.registry == {("redshift",): second}


def test_calling_the_registry_builds_the_model():
    # __call__ looks the factory up and invokes it with the key it was registered under
    registry = ModelRegistry()
    registry.register(("mass_1_source", "mass_2_source"), lambda parameter: parameter)
    key = ("mass_1_source", "mass_2_source")
    assert registry(key) == key


def test_the_registry_property_exposes_the_mapping():
    registry = ModelRegistry()

    def model(parameter):
        return None

    registry.register("redshift", model)
    assert registry.registry == {("redshift",): model}


@pytest.mark.parametrize("parameter", [1, 1.5, None, ["a"], {"a": 1}])
def test_rejects_a_parameter_that_is_neither_a_string_nor_a_tuple(parameter):
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="must be a string or tuple of strings"):
        registry.register(parameter, lambda p: None)


def test_rejects_a_tuple_that_is_not_all_strings():
    registry = ModelRegistry()
    with pytest.raises(AssertionError):
        registry.register(("mass_1_source", 2), lambda p: None)


def test_looking_up_an_unregistered_parameter_raises():
    registry = ModelRegistry()
    with pytest.raises(NotImplementedError):
        registry(("nope",))


def test_a_string_key_does_not_match_the_tuple_it_was_registered_under():
    # register() normalises a bare string to a one-element tuple, so lookups must use
    # the tuple form
    registry = ModelRegistry()
    registry.register("redshift", lambda parameter: None)
    with pytest.raises(NotImplementedError):
        registry("redshift")


def test_registries_are_independent():
    first, second = ModelRegistry(), ModelRegistry()
    first.register("redshift", lambda parameter: None)
    assert second.registry == {}
