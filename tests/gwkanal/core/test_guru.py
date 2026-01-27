# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Set

import pytest

from gwkanal.core.guru import _check_cycles


# fmt: off
@pytest.mark.parametrize(
    "graph, has_cycle",
    [
        ({"A": {"B"}, "B": {"C"}, "C": {"D"}, "D": set()}, False),             # No cycle
        ({"A": {"B"}, "B": {"C"}, "C": {"A"}}, True),                          # Simple cycle
        ({"A": {"B"}, "B": {"C"}, "C": {"D"}, "D": {"B"}}, True),              # Complex cycle
        ({"A": set(), "B": set(), "C": set()}, False),                         # Disconnected graph
        ({"A": {"B"}, "B": {"C"}, "C": {"D"}, "D": {"E"}, "E": set()}, False), # Long chain
        ({"A": {"B", "C"}, "B": {"D"}, "C": {"D"}, "D": set()}, False),        # Multiple paths
        ({"A": {"B"}, "B": {"C"}, "C": {"A"}, "D": set()}, True),              # Cycle with disconnected node
    ],
)
# fmt: on
def test_check_cycles(graph: Dict[str, Set[str]], has_cycle: bool) -> None:
    assert _check_cycles(graph) == has_cycle, f"Failed for graph: {graph}"
