import unittest

from backend.app.services.ml_service import (
    compute_fragility_analysis,
    compute_fragility_scores,
)


class TestFragilityScoring(unittest.TestCase):

    def test_empty_graph(self):
        result = compute_fragility_scores([], [])
        self.assertEqual(result, {})

    def test_single_isolated_node(self):
        nodes = [{"id": "A", "type": "function"}]

        result = compute_fragility_scores(nodes, [])

        self.assertIn("A", result)
        self.assertGreaterEqual(result["A"], 0)
        self.assertLessEqual(result["A"], 100)

    def test_high_fan_in(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
            {"id": "C", "type": "function"},
            {"id": "D", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "D"},
            {"source": "B", "target": "D"},
            {"source": "C", "target": "D"},
        ]

        analysis = compute_fragility_analysis(nodes, edges)

        self.assertGreater(
            analysis["D"]["fan_in"],
            analysis["A"]["fan_in"],
        )

    def test_high_fan_out(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
            {"id": "C", "type": "function"},
            {"id": "D", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "A", "target": "D"},
        ]

        analysis = compute_fragility_analysis(nodes, edges)

        self.assertGreater(
            analysis["A"]["fan_out"],
            analysis["B"]["fan_out"],
        )

    def test_duplicate_edges_do_not_inflate_fan_out(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
        ]

        duplicate_edges = [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "B"},
            {"source": "A", "target": "B"},
        ]

        single_edge = [
            {"source": "A", "target": "B"},
        ]

        duplicate_result = compute_fragility_analysis(
            nodes,
            duplicate_edges,
        )

        single_result = compute_fragility_analysis(
            nodes,
            single_edge,
        )

        self.assertEqual(
            duplicate_result["A"]["fan_out"],
            single_result["A"]["fan_out"],
        )

        self.assertEqual(
            duplicate_result["B"]["fan_in"],
            single_result["B"]["fan_in"],
        )

    def test_risk_propagation(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
            {"id": "C", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]

        analysis = compute_fragility_analysis(
            nodes,
            edges,
            iterations=3,
            damping=0.85,
        )

        self.assertIn(
            "propagated_risk",
            analysis["A"],
        )

    def test_scores_are_bounded(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
            {"id": "C", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},
        ]

        result = compute_fragility_scores(nodes, edges)

        for score in result.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_invalid_iterations(self):
        nodes = [{"id": "A", "type": "function"}]

        with self.assertRaises(ValueError):
            compute_fragility_scores(
                nodes,
                [],
                iterations=0,
            )

    def test_invalid_damping(self):
        nodes = [{"id": "A", "type": "function"}]

        with self.assertRaises(ValueError):
            compute_fragility_scores(
                nodes,
                [],
                damping=1.5,
            )

    def test_explainability(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "B"},
        ]

        analysis = compute_fragility_analysis(
            nodes,
            edges,
        )

        self.assertIn("reasons", analysis["A"])
        self.assertIsInstance(
            analysis["A"]["reasons"],
            list,
        )

    def test_invalid_edge_is_ignored(self):
        nodes = [
            {"id": "A", "type": "function"},
            {"id": "B", "type": "function"},
        ]

        edges = [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "DOES_NOT_EXIST"},
        ]

        result = compute_fragility_scores(nodes, edges)

        self.assertIn("A", result)
        self.assertIn("B", result)


if __name__ == "__main__":
    unittest.main()