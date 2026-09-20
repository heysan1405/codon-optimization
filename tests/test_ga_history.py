import unittest

from optimizers.genetic_algorithm import optimize_ga


class TestGAHistory(unittest.TestCase):

    def test_history_is_returned(self):

        result = optimize_ga(
            "ATGGCCGCG",
            organism="ecoli",
            population_size=10,
            generations=5,
            seed=42,
            return_history=True,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "sequence",
            result,
        )

        self.assertIn(
            "history",
            result,
        )

    def test_history_length(self):

        generations = 5

        result = optimize_ga(
            "ATGGCCGCG",
            organism="ecoli",
            population_size=10,
            generations=generations,
            seed=42,
            return_history=True,
        )

        history = result["history"]

        self.assertEqual(
            len(history),
            generations + 1,
        )

    def test_generation_numbers(self):

        result = optimize_ga(
            "ATGGCCGCG",
            organism="ecoli",
            population_size=10,
            generations=5,
            seed=42,
            return_history=True,
        )

        generations = [
            item["generation"]
            for item in result["history"]
        ]

        self.assertEqual(
            generations,
            [0, 1, 2, 3, 4, 5],
        )

    def test_history_fields(self):

        result = optimize_ga(
            "ATGGCCGCG",
            organism="ecoli",
            population_size=10,
            generations=3,
            seed=42,
            return_history=True,
        )

        first = result["history"][0]

        expected_fields = {
            "generation",
            "best_fitness",
            "mean_fitness",
            "best_cai",
            "best_gc",
            "edit_rate",
        }

        self.assertEqual(
            set(first.keys()),
            expected_fields,
        )

    def test_normal_mode_still_returns_string(self):

        result = optimize_ga(
            "ATGGCCGCG",
            organism="ecoli",
            population_size=10,
            generations=3,
            seed=42,
        )

        self.assertIsInstance(
            result,
            str,
        )


if __name__ == "__main__":
    unittest.main()