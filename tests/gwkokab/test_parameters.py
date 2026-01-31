# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import math

import numpy as np
import pytest

from gwkokab.parameters import RelationMesh


class TestRelationMeshBasic:
    """Test basic functionality of RelationMesh."""

    def test_empty_mesh(self):
        """Test that an empty mesh returns initial state unchanged."""
        mesh = RelationMesh()
        initial = {"x": 5}
        result = mesh.resolve(initial)
        assert result == {"x": 5}

    def test_single_rule_simple(self):
        """Test a single simple rule: y = 2x."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: 2 * x)

        result = mesh.resolve({"x": 3})
        assert result == {"x": 3, "y": 6}

    def test_no_overwrite(self):
        """Test that existing values are not overwritten."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: 2 * x)

        result = mesh.resolve({"x": 3, "y": 100})
        assert result == {"x": 3, "y": 100}  # y should not change


class TestRelationMeshMathematical:
    """Test mathematical relations."""

    def test_pythagorean_theorem(self):
        """Test Pythagorean theorem: c² = a² + b²."""
        mesh = RelationMesh()

        # c from a and b
        mesh.add_rule(("a", "b"), "c", lambda a, b: math.sqrt(a**2 + b**2))
        # a from b and c
        mesh.add_rule(("b", "c"), "a", lambda b, c: math.sqrt(c**2 - b**2))
        # b from a and c
        mesh.add_rule(("a", "c"), "b", lambda a, c: math.sqrt(c**2 - a**2))

        # Test different starting points
        result1 = mesh.resolve({"a": 3, "b": 4})
        assert abs(result1["c"] - 5) < 1e-10

        result2 = mesh.resolve({"b": 4, "c": 5})
        assert abs(result2["a"] - 3) < 1e-10

        result3 = mesh.resolve({"a": 3, "c": 5})
        assert abs(result3["b"] - 4) < 1e-10

    def test_quadratic_formula(self):
        """Test solving quadratic equations: ax² + bx + c = 0."""
        mesh = RelationMesh()

        # Calculate discriminant
        mesh.add_rule(("a", "b", "c"), "discriminant", lambda a, b, c: b**2 - 4 * a * c)

        # Calculate roots
        mesh.add_rule(
            ("a", "b", "discriminant"),
            ("x1", "x2"),
            lambda a, b, d: (
                (-b + math.sqrt(d)) / (2 * a),
                (-b - math.sqrt(d)) / (2 * a),
            ),
        )

        # For x² - 5x + 6 = 0, roots are 2 and 3
        result = mesh.resolve({"a": 1, "b": -5, "c": 6})
        assert abs(result["x1"] - 3) < 1e-10
        assert abs(result["x2"] - 2) < 1e-10
        assert result["discriminant"] == 1

    def test_circle_geometry(self):
        """Test circle relations: area, circumference, radius, diameter."""
        mesh = RelationMesh()

        # Diameter from radius
        mesh.add_rule(("radius",), "diameter", lambda r: 2 * r)
        # Radius from diameter
        mesh.add_rule(("diameter",), "radius", lambda d: d / 2)
        # Area from radius
        mesh.add_rule(("radius",), "area", lambda r: math.pi * r**2)
        # Circumference from radius
        mesh.add_rule(("radius",), "circumference", lambda r: 2 * math.pi * r)
        # Radius from area
        mesh.add_rule(("area",), "radius", lambda a: math.sqrt(a / math.pi))

        result = mesh.resolve({"radius": 5})
        assert abs(result["diameter"] - 10) < 1e-10
        assert abs(result["area"] - 25 * math.pi) < 1e-10
        assert abs(result["circumference"] - 10 * math.pi) < 1e-10

        # Start from area
        result2 = mesh.resolve({"area": math.pi * 9})
        assert abs(result2["radius"] - 3) < 1e-10
        assert abs(result2["diameter"] - 6) < 1e-10

    def test_coordinate_transformations(self):
        """Test Cartesian to Polar coordinate conversion and vice versa."""
        mesh = RelationMesh()

        # Cartesian to Polar
        mesh.add_rule(
            ("x", "y"),
            ("r", "theta"),
            lambda x, y: (math.sqrt(x**2 + y**2), math.atan2(y, x)),
        )

        # Polar to Cartesian
        mesh.add_rule(
            ("r", "theta"),
            ("x", "y"),
            lambda r, theta: (r * math.cos(theta), r * math.sin(theta)),
        )

        # Test Cartesian to Polar
        result1 = mesh.resolve({"x": 3, "y": 4})
        assert abs(result1["r"] - 5) < 1e-10
        assert abs(result1["theta"] - math.atan2(4, 3)) < 1e-10

        # Test Polar to Cartesian
        result2 = mesh.resolve({"r": 5, "theta": math.pi / 4})
        assert abs(result2["x"] - 5 * math.cos(math.pi / 4)) < 1e-10
        assert abs(result2["y"] - 5 * math.sin(math.pi / 4)) < 1e-10

    def test_triangle_relations(self):
        """Test triangle area and perimeter calculations."""
        mesh = RelationMesh()

        # Perimeter
        mesh.add_rule(
            ("side_a", "side_b", "side_c"), "perimeter", lambda a, b, c: a + b + c
        )

        # Semi-perimeter
        mesh.add_rule(("perimeter",), "semi_perimeter", lambda p: p / 2)

        # Heron's formula for area
        mesh.add_rule(
            ("side_a", "side_b", "side_c", "semi_perimeter"),
            "area",
            lambda a, b, c, s: math.sqrt(s * (s - a) * (s - b) * (s - c)),
        )

        # Right triangle with sides 3, 4, 5
        result = mesh.resolve({"side_a": 3, "side_b": 4, "side_c": 5})
        assert result["perimeter"] == 12
        assert result["semi_perimeter"] == 6
        assert abs(result["area"] - 6) < 1e-10


class TestRelationMeshPhysics:
    """Test physics relations."""

    def test_kinematic_equations(self):
        """Test basic kinematic equations: v = u + at, s = ut + 0.5at²."""
        mesh = RelationMesh()

        # Final velocity from initial velocity, acceleration, time
        mesh.add_rule(("u", "a", "t"), "v", lambda u, a, t: u + a * t)

        # Displacement from initial velocity, acceleration, time
        mesh.add_rule(("u", "a", "t"), "s", lambda u, a, t: u * t + 0.5 * a * t**2)

        # v² = u² + 2as
        mesh.add_rule(("u", "a", "s"), "v", lambda u, a, s: math.sqrt(u**2 + 2 * a * s))

        result = mesh.resolve({"u": 0, "a": 10, "t": 3})
        assert result["v"] == 30
        assert result["s"] == 45

    def test_newtons_second_law(self):
        """Test F = ma and related derivations."""
        mesh = RelationMesh()

        # Force from mass and acceleration
        mesh.add_rule(("mass", "acceleration"), "force", lambda m, a: m * a)

        # Acceleration from force and mass
        mesh.add_rule(("force", "mass"), "acceleration", lambda f, m: f / m)

        # Mass from force and acceleration
        mesh.add_rule(("force", "acceleration"), "mass", lambda f, a: f / a)

        result1 = mesh.resolve({"mass": 10, "acceleration": 5})
        assert result1["force"] == 50

        result2 = mesh.resolve({"force": 50, "mass": 10})
        assert result2["acceleration"] == 5

        result3 = mesh.resolve({"force": 50, "acceleration": 5})
        assert result3["mass"] == 10

    def test_energy_relations(self):
        """Test kinetic and potential energy relations."""
        mesh = RelationMesh()

        # Kinetic energy: KE = 0.5 * m * v²
        mesh.add_rule(
            ("mass", "velocity"), "kinetic_energy", lambda m, v: 0.5 * m * v**2
        )

        # Potential energy: PE = m * g * h
        mesh.add_rule(
            ("mass", "g", "height"), "potential_energy", lambda m, g, h: m * g * h
        )

        # Total mechanical energy
        mesh.add_rule(
            ("kinetic_energy", "potential_energy"),
            "total_energy",
            lambda ke, pe: ke + pe,
        )

        # Velocity from kinetic energy
        mesh.add_rule(
            ("kinetic_energy", "mass"), "velocity", lambda ke, m: math.sqrt(2 * ke / m)
        )

        result = mesh.resolve({"mass": 2, "velocity": 10, "g": 10, "height": 5})
        assert result["kinetic_energy"] == 100
        assert result["potential_energy"] == 100
        assert result["total_energy"] == 200

    def test_ohms_law(self):
        """Test Ohm's law: V = IR and power relations."""
        mesh = RelationMesh()

        # Voltage from current and resistance
        mesh.add_rule(("current", "resistance"), "voltage", lambda i, r: i * r)

        # Current from voltage and resistance
        mesh.add_rule(("voltage", "resistance"), "current", lambda v, r: v / r)

        # Resistance from voltage and current
        mesh.add_rule(("voltage", "current"), "resistance", lambda v, i: v / i)

        # Power: P = VI
        mesh.add_rule(("voltage", "current"), "power", lambda v, i: v * i)

        # Power: P = I²R
        mesh.add_rule(("current", "resistance"), "power", lambda i, r: i**2 * r)

        # Power: P = V²/R
        mesh.add_rule(("voltage", "resistance"), "power", lambda v, r: v**2 / r)

        result = mesh.resolve({"voltage": 12, "resistance": 4})
        assert result["current"] == 3
        assert result["power"] == 36

    def test_ideal_gas_law(self):
        """Test ideal gas law: PV = nRT."""
        mesh = RelationMesh()

        R = 8.314  # Gas constant

        # Pressure from n, T, V
        mesh.add_rule(("n", "T", "V"), "P", lambda n, t, v: n * R * t / v)

        # Volume from n, T, P
        mesh.add_rule(("n", "T", "P"), "V", lambda n, t, p: n * R * t / p)

        # Temperature from P, V, n
        mesh.add_rule(("P", "V", "n"), "T", lambda p, v, n: p * v / (n * R))

        result = mesh.resolve({"n": 1, "T": 273, "V": 0.0224})
        assert abs(result["P"] - 101325) < 1000  # Approximately 1 atm


class TestRelationMeshChaining:
    """Test chained derivations requiring multiple steps."""

    def test_multi_step_derivation(self):
        """Test derivation requiring multiple intermediate steps."""
        mesh = RelationMesh()

        # a -> b
        mesh.add_rule(("a",), "b", lambda a: a * 2)
        # b -> c
        mesh.add_rule(("b",), "c", lambda b: b + 3)
        # c -> d
        mesh.add_rule(("c",), "d", lambda c: c**2)
        # d -> e
        mesh.add_rule(("d",), "e", lambda d: d / 2)

        result = mesh.resolve({"a": 5})
        assert result["b"] == 10
        assert result["c"] == 13
        assert result["d"] == 169
        assert result["e"] == 84.5

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph."""
        mesh = RelationMesh()

        #     a
        #    / \
        #   b   c
        #    \ /
        #     d

        mesh.add_rule(("a",), "b", lambda a: a * 2)
        mesh.add_rule(("a",), "c", lambda a: a * 3)
        mesh.add_rule(("b", "c"), "d", lambda b, c: b + c)

        result = mesh.resolve({"a": 5})
        assert result["b"] == 10
        assert result["c"] == 15
        assert result["d"] == 25

    def test_complex_dependency_graph(self):
        """Test complex dependency with multiple paths."""
        mesh = RelationMesh()

        mesh.add_rule(("x",), "a", lambda x: x + 1)
        mesh.add_rule(("x",), "b", lambda x: x * 2)
        mesh.add_rule(("a", "b"), "c", lambda a, b: a + b)
        mesh.add_rule(("b",), "d", lambda b: b**2)
        mesh.add_rule(("c", "d"), "e", lambda c, d: c * d)

        result = mesh.resolve({"x": 3})
        assert result["a"] == 4
        assert result["b"] == 6
        assert result["c"] == 10
        assert result["d"] == 36
        assert result["e"] == 360


class TestRelationMeshDeriveOnly:
    """Test derive_only functionality."""

    def test_derive_only_subset(self):
        """Test deriving only a subset of computed values."""
        mesh = RelationMesh()

        mesh.add_rule(("a",), "b", lambda a: a * 2)
        mesh.add_rule(("b",), "c", lambda b: b + 3)
        mesh.add_rule(("c",), "d", lambda c: c**2)

        result = mesh.derive_only({"a": 5}, {"c", "d"})
        assert result == {"c": 13, "d": 169}
        assert "a" not in result
        assert "b" not in result

    def test_derive_only_with_intermediate(self):
        """Test that intermediate values are computed but not returned."""
        mesh = RelationMesh()

        mesh.add_rule(("x",), "intermediate1", lambda x: x * 2)
        mesh.add_rule(("intermediate1",), "intermediate2", lambda i: i + 5)
        mesh.add_rule(("intermediate2",), "final", lambda i: i**2)

        result = mesh.derive_only({"x": 3}, {"final"})
        assert result == {"final": 121}  # ((3*2)+5)^2 = 11^2 = 121


class TestRelationMeshMultipleRules:
    """Test behavior with multiple rules for same output."""

    def test_multiple_rules_same_output(self):
        """Test that first applicable rule wins when multiple target same output."""
        mesh = RelationMesh()

        # Two different ways to compute 'result'
        mesh.add_rule(("a",), "result", lambda a: a * 10)
        mesh.add_rule(("b",), "result", lambda b: b + 100)

        # If we provide 'a', first rule should be used
        result1 = mesh.resolve({"a": 5})
        assert result1["result"] == 50

        # If we provide 'b', second rule should be used
        result2 = mesh.resolve({"b": 5})
        assert result2["result"] == 105

        # If we provide both, the first applicable rule should win
        result3 = mesh.resolve({"a": 5, "b": 10})
        assert result3["result"] == 50  # First rule applied

    def test_fallback_rules(self):
        """Test fallback behavior when preferred inputs are missing."""
        mesh = RelationMesh()

        # Preferred: compute from both x and y
        mesh.add_rule(("x", "y"), "result", lambda x, y: x * y)
        # Fallback: compute from just x
        mesh.add_rule(("x",), "result", lambda x: x**2)

        # With both inputs, use first rule
        result1 = mesh.resolve({"x": 3, "y": 4})
        assert result1["result"] == 12

        # With only x, use fallback
        result2 = mesh.resolve({"x": 5})
        assert result2["result"] == 25


class TestRelationMeshEdgeCases:
    """Test edge cases and error conditions."""

    def test_circular_dependency_no_infinite_loop(self):
        """Test that circular dependencies don't cause infinite loops."""
        mesh = RelationMesh()

        # Create a potential circular dependency
        mesh.add_rule(("a",), "b", lambda a: a * 2)
        mesh.add_rule(("b",), "c", lambda b: b + 1)
        # This would create a circle if 'a' wasn't already known
        mesh.add_rule(("c",), "a", lambda c: c - 1)

        # Should resolve without infinite loop
        result = mesh.resolve({"a": 5})
        assert result["a"] == 5
        assert result["b"] == 10
        assert result["c"] == 11

    def test_missing_dependencies(self):
        """Test behavior when dependencies cannot be satisfied."""
        mesh = RelationMesh()

        mesh.add_rule(("a", "b"), "c", lambda a, b: a + b)
        mesh.add_rule(("c",), "d", lambda c: c * 2)

        # Only provide 'a', not 'b'
        result = mesh.resolve({"a": 5})
        assert result == {"a": 5}  # c and d cannot be computed

    def test_empty_initial_state(self):
        """Test with empty initial state."""
        mesh = RelationMesh()

        mesh.add_rule(("a",), "b", lambda a: a * 2)

        result = mesh.resolve({})
        assert result == {}


class TestRelationMeshRealWorldScenarios:
    """Test real-world physics scenarios."""

    def test_projectile_motion(self):
        """Test projectile motion calculations."""
        mesh = RelationMesh()

        g = 9.8  # gravity

        # Horizontal velocity components
        mesh.add_rule(("v0", "angle"), "vx", lambda v0, angle: v0 * math.cos(angle))
        mesh.add_rule(("v0", "angle"), "vy", lambda v0, angle: v0 * math.sin(angle))

        # Time of flight: t = 2*vy/g
        mesh.add_rule(("vy",), "time_of_flight", lambda vy: 2 * vy / g)

        # Range: R = vx * t
        mesh.add_rule(("vx", "time_of_flight"), "range", lambda vx, t: vx * t)

        # Maximum height: h = vy²/(2g)
        mesh.add_rule(("vy",), "max_height", lambda vy: vy**2 / (2 * g))

        # Launch at 45 degrees with initial velocity 20 m/s
        result = mesh.resolve({"v0": 20, "angle": math.pi / 4})

        assert abs(result["vx"] - 20 * math.cos(math.pi / 4)) < 1e-10
        assert abs(result["vy"] - 20 * math.sin(math.pi / 4)) < 1e-10
        assert (
            abs(result["max_height"] - (20 * math.sin(math.pi / 4)) ** 2 / (2 * g))
            < 1e-8
        )

    def test_lens_equation(self):
        """Test thin lens equation: 1/f = 1/u + 1/v."""
        mesh = RelationMesh()

        # Focal length from object and image distances
        mesh.add_rule(("u", "v"), "f", lambda u, v: 1 / (1 / u + 1 / v))

        # Image distance from object distance and focal length
        mesh.add_rule(("u", "f"), "v", lambda u, f: 1 / (1 / f - 1 / u))

        # Object distance from image distance and focal length
        mesh.add_rule(("v", "f"), "u", lambda v, f: 1 / (1 / f - 1 / v))

        # Magnification
        mesh.add_rule(("v", "u"), "magnification", lambda v, u: -v / u)

        result = mesh.resolve({"u": 20, "f": 10})
        assert abs(result["v"] - 20) < 1e-10
        assert abs(result["magnification"] - (-1)) < 1e-10


class TestResolveFromArraysBasic:
    """Test basic functionality of resolve_from_arrays."""

    def test_single_column_single_row(self):
        """Test with a single parameter and single data point."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)

        initial = np.array([[5]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (1, 2)
        assert result[0, 0] == 5
        assert "x" in params
        assert "y" in params

    def test_single_rule_multiple_rows(self):
        """Test vectorized computation with multiple data points."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)

        # 5 rows, 1 column
        initial = np.array([[1], [2], [3], [4], [5]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (5, 2)
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3, 4, 5])
        assert "x" in params
        assert "y" in params

    def test_multiple_columns_input(self):
        """Test with multiple input parameters."""
        mesh = RelationMesh()
        mesh.add_rule(("a", "b"), "c", lambda a, b: a + b)

        # 3 rows, 2 columns
        initial = np.array([[1, 2], [3, 4], [5, 6]])
        result, params = mesh.resolve_from_arrays(initial, ("a", "b"))

        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[:, 0], [1, 3, 5])  # column a
        np.testing.assert_array_equal(result[:, 1], [2, 4, 6])  # column b
        assert "a" in params
        assert "b" in params
        assert "c" in params

    def test_param_order_preserved(self):
        """Test that the original parameter order is preserved in output."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)
        mesh.add_rule(("y",), "z", lambda y: y + 1)

        initial = np.array([[10], [20], [30]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        # Original parameter order should be maintained
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[:, 0], [10, 20, 30])
        assert params[0] == "x"  # x should be first since it was in param_order


class TestResolveFromArraysMathematical:
    """Test mathematical operations with arrays."""

    def test_pythagorean_theorem_vectorized(self):
        """Test Pythagorean theorem with multiple triangles."""
        mesh = RelationMesh()
        mesh.add_rule(("a", "b"), "c", lambda a, b: np.sqrt(a**2 + b**2))

        # Multiple right triangles
        initial = np.array(
            [
                [3, 4],  # 3-4-5 triangle
                [5, 12],  # 5-12-13 triangle
                [8, 15],  # 8-15-17 triangle
                [7, 24],  # 7-24-25 triangle
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("a", "b"))

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result[:, 0], [3, 5, 8, 7])
        np.testing.assert_array_equal(result[:, 1], [4, 12, 15, 24])
        assert "c" in params

    def test_circle_calculations_vectorized(self):
        """Test circle area and circumference for multiple radii."""
        mesh = RelationMesh()
        mesh.add_rule(("radius",), "area", lambda r: np.pi * r**2)
        mesh.add_rule(("radius",), "circumference", lambda r: 2 * np.pi * r)
        mesh.add_rule(("radius",), "diameter", lambda r: 2 * r)

        initial = np.array([[1], [2], [3], [5], [10]])
        result, params = mesh.resolve_from_arrays(initial, ("radius",))

        assert result.shape == (5, 4)
        np.testing.assert_array_equal(result[:, 3], [1, 2, 3, 5, 10])
        assert "area" in params
        assert "circumference" in params
        assert "diameter" in params

    def test_quadratic_formula_vectorized(self):
        """Test quadratic formula with multiple equations."""
        mesh = RelationMesh()

        # Discriminant
        mesh.add_rule(("a", "b", "c"), "discriminant", lambda a, b, c: b**2 - 4 * a * c)

        # Roots
        mesh.add_rule(
            ("a", "b", "discriminant"),
            ("x1", "x2"),
            lambda a, b, d: ((-b + np.sqrt(d)) / (2 * a), (-b - np.sqrt(d)) / (2 * a)),
        )

        # Multiple quadratic equations
        initial = np.array(
            [
                [1, -5, 6],  # x² - 5x + 6 = 0 (roots: 2, 3)
                [1, -7, 12],  # x² - 7x + 12 = 0 (roots: 3, 4)
                [1, -3, 2],  # x² - 3x + 2 = 0 (roots: 1, 2)
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("a", "b", "c"))

        assert result.shape == (3, 6)
        assert "discriminant" in params
        assert "x1" in params
        assert "x2" in params

    def test_coordinate_transformation_vectorized(self):
        """Test Cartesian to Polar conversion for multiple points."""
        mesh = RelationMesh()

        mesh.add_rule(
            ("x", "y"),
            ("r", "theta"),
            lambda x, y: (np.sqrt(x**2 + y**2), np.arctan2(y, x)),
        )

        # Multiple points
        initial = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [3, 4],
                [-1, 1],
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("x", "y"))

        assert result.shape == (5, 4)
        assert "r" in params
        assert "theta" in params


class TestResolveFromArraysPhysics:
    """Test physics calculations with arrays."""

    def test_kinematics_multiple_objects(self):
        """Test kinematic equations for multiple moving objects."""
        mesh = RelationMesh()

        # v = u + at
        mesh.add_rule(("u", "a", "t"), "v", lambda u, a, t: u + a * t)

        # s = ut + 0.5at²
        mesh.add_rule(("u", "a", "t"), "s", lambda u, a, t: u * t + 0.5 * a * t**2)

        # Multiple objects with different initial conditions
        initial = np.array(
            [
                [0, 10, 3],  # u=0, a=10, t=3
                [5, 5, 2],  # u=5, a=5, t=2
                [10, -2, 5],  # u=10, a=-2, t=5 (deceleration)
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("u", "a", "t"))

        assert result.shape == (3, 5)
        assert "v" in params
        assert "s" in params

    def test_energy_calculations_vectorized(self):
        """Test energy calculations for multiple objects."""
        mesh = RelationMesh()

        mesh.add_rule(
            ("mass", "velocity"), "kinetic_energy", lambda m, v: 0.5 * m * v**2
        )
        mesh.add_rule(
            ("mass", "g", "height"), "potential_energy", lambda m, g, h: m * g * h
        )
        mesh.add_rule(
            ("kinetic_energy", "potential_energy"),
            "total_energy",
            lambda ke, pe: ke + pe,
        )

        # Multiple objects
        initial = np.array(
            [
                [1, 10, 10, 5],  # m=1, v=10, g=10, h=5
                [2, 5, 10, 10],  # m=2, v=5, g=10, h=10
                [5, 4, 9.8, 2],  # m=5, v=4, g=9.8, h=2
            ]
        )
        result, params = mesh.resolve_from_arrays(
            initial, ("mass", "velocity", "g", "height")
        )

        assert result.shape == (3, 7)
        assert "kinetic_energy" in params
        assert "potential_energy" in params
        assert "total_energy" in params

    def test_ohms_law_multiple_circuits(self):
        """Test Ohm's law for multiple circuits."""
        mesh = RelationMesh()

        mesh.add_rule(("voltage", "resistance"), "current", lambda v, r: v / r)
        mesh.add_rule(("voltage", "current"), "power", lambda v, i: v * i)

        # Multiple circuits
        initial = np.array(
            [
                [12, 4],  # V=12, R=4
                [9, 3],  # V=9, R=3
                [5, 10],  # V=5, R=10
                [24, 6],  # V=24, R=6
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("voltage", "resistance"))

        assert result.shape == (4, 4)
        assert "current" in params
        assert "power" in params

    def test_projectile_motion_multiple_launches(self):
        """Test projectile motion for multiple launches."""
        mesh = RelationMesh()

        g = 9.8

        mesh.add_rule(("v0", "angle"), "vx", lambda v0, angle: v0 * np.cos(angle))
        mesh.add_rule(("v0", "angle"), "vy", lambda v0, angle: v0 * np.sin(angle))
        mesh.add_rule(("vy",), "time_of_flight", lambda vy: 2 * vy / g)
        mesh.add_rule(("vx", "time_of_flight"), "range", lambda vx, t: vx * t)
        mesh.add_rule(("vy",), "max_height", lambda vy: vy**2 / (2 * g))

        # Multiple launches at different angles and speeds
        initial = np.array(
            [
                [20, np.pi / 4],  # 45 degrees
                [30, np.pi / 6],  # 30 degrees
                [25, np.pi / 3],  # 60 degrees
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("v0", "angle"))

        assert result.shape == (3, 7)
        assert "vx" in params
        assert "vy" in params
        assert "range" in params
        assert "max_height" in params


class TestResolveFromArraysChaining:
    """Test chained derivations with arrays."""

    def test_multi_step_chain_vectorized(self):
        """Test multi-step derivation with vectorized operations."""
        mesh = RelationMesh()

        mesh.add_rule(("a",), "b", lambda a: a * 2)
        mesh.add_rule(("b",), "c", lambda b: b + 3)
        mesh.add_rule(("c",), "d", lambda c: c**2)

        initial = np.array([[1], [2], [3], [4], [5]])
        result, params = mesh.resolve_from_arrays(initial, ("a",))

        assert result.shape == (5, 4)
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3, 4, 5])
        assert "b" in params
        assert "c" in params
        assert "d" in params

    def test_diamond_dependency_vectorized(self):
        """Test diamond-shaped dependency with arrays."""
        mesh = RelationMesh()

        mesh.add_rule(("a",), "b", lambda a: a * 2)
        mesh.add_rule(("a",), "c", lambda a: a * 3)
        mesh.add_rule(("b", "c"), "d", lambda b, c: b + c)

        initial = np.array([[5], [10], [15]])
        result, params = mesh.resolve_from_arrays(initial, ("a",))

        assert result.shape == (3, 4)
        assert "b" in params
        assert "c" in params
        assert "d" in params


class TestResolveFromArraysEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_array(self):
        """Test with empty array (0 rows)."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)

        initial = np.array([]).reshape(0, 1)
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (0, 2)
        assert "x" in params
        assert "y" in params

    def test_single_row_many_columns(self):
        """Test with single row but many columns."""
        mesh = RelationMesh()
        mesh.add_rule(("a", "b", "c"), "d", lambda a, b, c: a + b + c)

        initial = np.array([[1, 2, 3]])
        result, params = mesh.resolve_from_arrays(initial, ("a", "b", "c"))

        assert result.shape == (1, 4)
        assert "d" in params

    def test_large_dataset(self):
        """Test with a large dataset."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x**2)
        mesh.add_rule(("y",), "z", lambda y: np.sqrt(y))

        # 1000 rows
        initial = np.random.rand(1000, 1) * 100
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (1000, 3)
        assert "y" in params
        assert "z" in params

    def test_floating_point_precision(self):
        """Test floating point operations maintain precision."""
        mesh = RelationMesh()
        mesh.add_rule(("a",), "b", lambda a: a + 0.1)
        mesh.add_rule(("b",), "c", lambda b: b + 0.1)

        initial = np.array([[0.1], [0.2], [0.3]])
        result, params = mesh.resolve_from_arrays(initial, ("a",))

        assert result.shape == (3, 3)
        assert "b" in params
        assert "c" in params

    def test_negative_values(self):
        """Test handling of negative values."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x**2)
        mesh.add_rule(("x",), "abs_x", lambda x: np.abs(x))

        initial = np.array([[-5], [-3], [0], [3], [5]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[:, 1], [-5, -3, 0, 3, 5])
        assert "y" in params
        assert "abs_x" in params

    def test_tuple_outputs_vectorized(self):
        """Test tuple outputs with array operations."""
        mesh = RelationMesh()

        mesh.add_rule(("x",), ("double", "triple"), lambda x: (x * 2, x * 3))

        initial = np.array([[1], [2], [3], [4]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (4, 3)
        assert "double" in params
        assert "triple" in params


class TestResolveFromArraysRealWorld:
    """Test real-world scenarios with arrays."""

    def test_financial_calculations_batch(self):
        """Test compound interest for multiple investments."""
        mesh = RelationMesh()

        mesh.add_rule(
            ("principal", "rate", "time"),
            "future_value",
            lambda p, r, t: p * (1 + r) ** t,
        )
        mesh.add_rule(
            ("future_value", "principal"), "interest_earned", lambda fv, p: fv - p
        )

        # Multiple investments
        initial = np.array(
            [
                [1000, 0.05, 10],
                [5000, 0.07, 5],
                [10000, 0.04, 20],
                [2500, 0.06, 15],
            ]
        )
        result, params = mesh.resolve_from_arrays(
            initial, ("principal", "rate", "time")
        )

        assert result.shape == (4, 5)
        assert "future_value" in params
        assert "interest_earned" in params

    def test_temperature_conversion_batch(self):
        """Test temperature conversions for multiple values."""
        mesh = RelationMesh()

        # Celsius to Fahrenheit
        mesh.add_rule(("celsius",), "fahrenheit", lambda c: c * 9 / 5 + 32)

        # Celsius to Kelvin
        mesh.add_rule(("celsius",), "kelvin", lambda c: c + 273.15)

        # Multiple temperatures
        initial = np.array([[0], [25], [100], [-40], [37]])
        result, params = mesh.resolve_from_arrays(initial, ("celsius",))

        assert result.shape == (5, 3)
        assert "fahrenheit" in params
        assert "kelvin" in params

    def test_statistical_calculations(self):
        """Test statistical transformations."""
        mesh = RelationMesh()

        # Z-score calculation
        mesh.add_rule(
            ("value", "mean", "std"), "z_score", lambda x, mu, sigma: (x - mu) / sigma
        )

        # Squared deviation
        mesh.add_rule(
            ("value", "mean"), "squared_deviation", lambda x, mu: (x - mu) ** 2
        )

        # Multiple data points with same mean and std
        mean_val = 100
        std_val = 15
        initial = np.array(
            [
                [85, mean_val, std_val],
                [100, mean_val, std_val],
                [115, mean_val, std_val],
                [70, mean_val, std_val],
                [130, mean_val, std_val],
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("value", "mean", "std"))

        assert result.shape == (5, 5)
        assert "z_score" in params
        assert "squared_deviation" in params

    def test_unit_conversions_batch(self):
        """Test various unit conversions in batch."""
        mesh = RelationMesh()

        # Distance conversions
        mesh.add_rule(("meters",), "kilometers", lambda m: m / 1000)
        mesh.add_rule(("meters",), "miles", lambda m: m / 1609.34)
        mesh.add_rule(("meters",), "feet", lambda m: m * 3.28084)

        # Multiple distances
        initial = np.array([[100], [1000], [5000], [10000]])
        result, params = mesh.resolve_from_arrays(initial, ("meters",))

        assert result.shape == (4, 4)
        assert "kilometers" in params
        assert "miles" in params
        assert "feet" in params

    def test_geometry_batch_processing(self):
        """Test geometric calculations for multiple shapes."""
        mesh = RelationMesh()

        # Rectangle
        mesh.add_rule(("length", "width"), "area", lambda l, w: l * w)
        mesh.add_rule(("length", "width"), "perimeter", lambda l, w: 2 * (l + w))
        mesh.add_rule(
            ("area",), "diagonal", lambda a: np.sqrt(a) * np.sqrt(2)
        )  # for squares

        # Multiple rectangles
        initial = np.array(
            [
                [5, 3],
                [10, 8],
                [7, 7],  # square
                [12, 5],
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("length", "width"))

        assert result.shape == (4, 5)
        assert "area" in params
        assert "perimeter" in params


class TestResolveFromArraysNumerical:
    """Test numerical accuracy and performance."""

    def test_numerical_stability(self):
        """Test numerical stability with very small and very large numbers."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "x_squared", lambda x: x**2)

        # Mix of very small and very large
        initial = np.array([[1e-100], [1e-50], [1], [1e50], [1e100]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (5, 2)
        assert "x_squared" in params
        # Check that computation completed without errors

    def test_preserve_array_dtype(self):
        """Test that array dtype is preserved where possible."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)

        # Integer input
        initial = np.array([[1], [2], [3]], dtype=np.int32)
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (3, 2)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x * 2)

        initial = np.array([[1], [np.nan], [3]])
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (3, 2)
        assert np.isnan(result[1, 0])  # NaN should be preserved

    def test_inf_handling(self):
        """Test handling of infinity values."""
        mesh = RelationMesh()
        mesh.add_rule(("x",), "y", lambda x: x / 2)

        initial = np.array(
            [
                [np.inf],
                [-np.inf],
                [10],
            ]
        )
        result, params = mesh.resolve_from_arrays(initial, ("x",))

        assert result.shape == (3, 2)
        assert np.isinf(result[0, 1])
        assert np.isinf(result[1, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
