import unittest

from core.similarity import (
    hamming_distance,
    sequence_identity,
    changed_positions,
    codon_changes,
    total_codons,
    edit_rate,
)


class TestSimilarity(unittest.TestCase):

    def test_hamming_distance(self):

        self.assertEqual(
            hamming_distance(
                "ATGGCC",
                "ATGACC",
            ),
            1,
        )

    def test_identical_sequences(self):

        self.assertEqual(
            hamming_distance(
                "ATGGCC",
                "ATGGCC",
            ),
            0,
        )

        self.assertEqual(
            sequence_identity(
                "ATGGCC",
                "ATGGCC",
            ),
            1.0,
        )

    def test_sequence_identity(self):

        self.assertAlmostEqual(
            sequence_identity(
                "ATGGCC",
                "ATGACC",
            ),
            5 / 6,
        )

    def test_changed_positions(self):

        self.assertEqual(
            changed_positions(
                "ATGGCC",
                "ATGACC",
            ),
            [3],
        )

    def test_total_codons(self):

        self.assertEqual(
            total_codons("ATGGCCGCG"),
            3,
        )

    def test_codon_changes(self):

        self.assertEqual(
            codon_changes(
                "ATGGCCGCG",
                "ATGGCGGCA",
            ),
            2,
        )

    def test_edit_rate(self):

        self.assertAlmostEqual(
            edit_rate(
                "ATGGCCGCG",
                "ATGGCGGCA",
            ),
            2 / 3,
        )

    def test_invalid_hamming_length(self):

        with self.assertRaises(ValueError):

            hamming_distance(
                "ATG",
                "ATGGCC",
            )

    def test_invalid_codon_length(self):

        with self.assertRaises(ValueError):

            codon_changes(
                "ATGGC",
                "ATGGC",
            )


if __name__ == "__main__":
    unittest.main()