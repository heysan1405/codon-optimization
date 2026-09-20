import unittest

from constraints.gc_target import (
    validate_target_gc,
    gc_deviation,
    gc_target_score,
    within_gc_tolerance,
)


class TestGCTarget(unittest.TestCase):

    def test_validate_target(self):

        self.assertEqual(
            validate_target_gc(0.60),
            0.60,
        )

    def test_invalid_target(self):

        with self.assertRaises(ValueError):
            validate_target_gc(1.5)

    def test_gc_deviation(self):

        # ATGC = 50% GC
        self.assertAlmostEqual(
            gc_deviation(
                "ATGC",
                0.60,
            ),
            0.10,
        )

    def test_exact_target(self):

        # ATGC = 50% GC
        self.assertAlmostEqual(
            gc_deviation(
                "ATGC",
                0.50,
            ),
            0.0,
        )

        self.assertAlmostEqual(
            gc_target_score(
                "ATGC",
                0.50,
            ),
            1.0,
        )

    def test_target_score(self):

        # ATGC = 50%, target = 60%
        self.assertAlmostEqual(
            gc_target_score(
                "ATGC",
                0.60,
            ),
            0.90,
        )

    def test_within_tolerance(self):

        self.assertTrue(
            within_gc_tolerance(
                "ATGC",
                0.55,
                0.05,
            )
        )

    def test_outside_tolerance(self):

        self.assertFalse(
            within_gc_tolerance(
                "ATGC",
                0.70,
                0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()