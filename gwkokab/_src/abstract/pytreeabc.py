#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import inspect
from typing_extensions import Hashable, Iterable, Tuple, TypeVar

from jax import tree_util


PyTree = TypeVar("PyTree", bound="PyTreeABC")


class PyTreeABC:
    r"""Abstract base class for PyTree objects. It is taken from the implementation of
    :class:`numpyro.distributions.distribution.Distribution`
    `implementation
    <https://github.com/pyro-ppl/numpyro/blob/f6eb6ce152bd8e903dd56eeb5909ae0b59e24abe/numpyro/distributions/distribution.py#L103-L218>`_
    """

    pytree_data_fields = ()
    pytree_aux_fields = ()

    # register PyTreeABC as a pytree
    # ref: https://github.com/google/jax/issues/2916
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tree_util.register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @classmethod
    def gather_pytree_data_fields(cls) -> Tuple[str]:
        bases = inspect.getmro(cls)

        all_pytree_data_fields = ()
        for base in bases:
            if issubclass(base, PyTreeABC):
                all_pytree_data_fields += base.__dict__.get("pytree_data_fields")
        # remove duplicates
        all_pytree_data_fields = tuple(set(all_pytree_data_fields))
        return all_pytree_data_fields

    @classmethod
    def gather_pytree_aux_fields(cls) -> Tuple[str]:
        bases = inspect.getmro(cls)

        all_pytree_aux_fields = ()
        for base in bases:
            if issubclass(base, PyTreeABC):
                all_pytree_aux_fields += base.__dict__.get("pytree_aux_fields", ())
        # remove duplicates
        all_pytree_aux_fields = tuple(set(all_pytree_aux_fields))
        return all_pytree_aux_fields

    def tree_flatten(self) -> Tuple[Iterable[PyTree], Hashable]:
        r"""
        return: A JAX PyTree of values representing the object, and
            Data that will be treated as constant through JAX operations.
        """
        all_pytree_data_fields_names = type(self).gather_pytree_data_fields()
        all_pytree_data_fields_vals = tuple(
            self.__dict__.get(attr_name) for attr_name in all_pytree_data_fields_names
        )
        all_pytree_aux_fields_names = type(self).gather_pytree_aux_fields()
        all_pytree_aux_fields_vals = tuple(
            self.__dict__.get(attr_name) for attr_name in all_pytree_aux_fields_names
        )
        return (all_pytree_data_fields_vals, all_pytree_aux_fields_vals)

    @classmethod
    def tree_unflatten(cls, aux_data: Hashable, params: Iterable[PyTree]) -> PyTree:
        """
        :param aux_data: Data that will be treated as constant through JAX operations.
        :type aux_data: Hashable
        :param params: A JAX PyTree of values from which the object is constructed.
        :type params: Iterable[PyTree]
        :return: A constructed object.
        :rtype: PyTree
        """
        pytree_data_fields = cls.gather_pytree_data_fields()
        pytree_aux_fields = cls.gather_pytree_aux_fields()

        pytree_data_fields_dict = dict(zip(pytree_data_fields, params))
        pytree_aux_fields_dict = dict(zip(pytree_aux_fields, aux_data))

        d = cls.__new__(cls)

        for k, v in pytree_data_fields_dict.items():
            if v is not None:
                setattr(d, k, v)

        for k, v in pytree_aux_fields_dict.items():
            if v is not None:
                setattr(d, k, v)

        PyTreeABC.__init__(d)
        return d
