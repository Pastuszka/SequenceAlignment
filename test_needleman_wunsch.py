import unittest
import numpy as np
from needleman_wunsch import NeedlemanWunsch


class TestNeedlemanWunsch(unittest.TestCase):

    test_config = {'GAP_PENALTY': -4,
                   'SAME_AWARD': 5,
                   'DIFFERENCE_PENALTY': -3,
                   'MAX_SEQ_LENGTH': 1000}

    def test_align(self):
        seq_a = 'CATAC'
        seq_b = 'ATCGAC'
        correct_alignment = np.array([[0, -4, -8, -12, -16, -20, -24],
                                      [-4, -3, -7, -3, -7, -11, -15],
                                      [-8, 1, -3, -7, -6, -2, -6],
                                      [-12, -3, 6, 2, -2, -6, -5],
                                      [-16, -7, 2, 3, -1, 3, -1],
                                      [-20, -11, -2, -1, 0, -1, 8]])
        alignment_matrix, arrow_matrix = NeedlemanWunsch.__align__(seq_a, seq_b)
        self.assertEqual(alignment_matrix, correct_alignment)


if __name__ == '__main__':
    unittest.main()
