import unittest
import numpy as np
from needleman_wunsch import NeedlemanWunsch


class TestNeedlemanWunsch(unittest.TestCase):

    test_config = {'GAP_PENALTY': -1,
                   'SAME_AWARD': 1,
                   'DIFFERENCE_PENALTY': 0,
                   'MAX_SEQ_LENGTH': 1000}

    def test_align(self):
        seq_a = 'CGA'
        seq_b = 'CACGA'
        solver = NeedlemanWunsch()
        correct_alignment = np.array([[0, -1, -2, -3, -4, -5],
                                      [-1, 1, 0, -1, -2, -3],
                                      [-2, 0, 1, 0, 0, -1],
                                      [-3, -1, 1, 1, 0, 1]],
                                     dtype=float)
        alignment_matrix, arrow_matrix = solver.__align__(seq_a, seq_b)
        self.assertTrue(np.array_equal(alignment_matrix, correct_alignment))


if __name__ == '__main__':
    unittest.main()
