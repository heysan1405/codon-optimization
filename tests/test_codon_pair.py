import unittest

from core.codon_pair import (
    get_codon_pairs,
    codon_pair_frequency,
    codon_pair_score,
    codon_pair_scores,
    validate_pair_table,
)


class TestCodonPair(unittest.TestCase):

    def setUp(self):

        self.pair_table = {
            ("ATG", "GCC"): 0.20,
            ("GCC", "GCG"): 0.40,
            ("GCG", "GCA"): 0.30,
        }

    def test_get_codon_pairs(self):

        sequence = "ATGGCCGCGGCA"

        expected = [
            ("ATG", "GCC"),
            ("GCC", "GCG"),
            ("GCG", "GCA"),
        ]

        self.assertEqual(
            get_codon_pairs(sequence),
            expected,
        )

    def test_single_codon_sequence(self):

        self.assertEqual(
            get_codon_pairs("ATG"),
            [],
        )

    def test_pair_frequency(self):

        self.assertAlmostEqual(
            codon_pair_frequency(
                "ATG",
                "GCC",
                self.pair_table,
            ),
            0.20,
        )

    def test_missing_pair(self):

        self.assertEqual(
            codon_pair_frequency(
                "AAA",
                "CCC",
                self.pair_table,
            ),
            0.0,
        )

    def test_pair_score(self):

        sequence = "ATGGCCGCGGCA"

        expected = (
            0.20 + 0.40 + 0.30
        ) / 3

        self.assertAlmostEqual(
            codon_pair_score(
                sequence,
                self.pair_table,
            ),
            expected,
        )

    def test_pair_scores(self):

        sequence = "ATGGCCGCGGCA"

        results = codon_pair_scores(
            sequence,
            self.pair_table,
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertEqual(
            results[0]["codon1"],
            "ATG",
        )

        self.assertEqual(
            results[0]["codon2"],
            "GCC",
        )

        self.assertAlmostEqual(
            results[0]["score"],
            0.20,
        )

    def test_invalid_pair_table(self):

        with self.assertRaises(ValueError):

            validate_pair_table(
                {
                    ("ATG",): 0.5
                }
            )

    def test_empty_pair_table(self):

        self.assertEqual(
            codon_pair_score(
                "ATGGCC",
                {},
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()