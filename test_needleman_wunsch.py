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

    def test_arrows(self):
        seq_a = 'CGA'
        seq_b = 'CACGA'
        solver = NeedlemanWunsch()
        # [Left, Up, Diagonal]
        correct_arrows = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                    [1, 0, 0], [1, 0, 0]],
                                   [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1],
                                    [1, 0, 0], [1, 0, 0]],
                                   [[0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                                    [0, 0, 1], [1, 0, 0]],
                                   [[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
                                    [1, 0, 1], [0, 0, 1]]],
                                  dtype=bool)
        alignment_matrix, arrow_matrix = solver.__align__(seq_a, seq_b)
        self.assertTrue(np.array_equal(arrow_matrix, correct_arrows))

    def test_alignments(self):
        seq_a = 'CGA'
        seq_b = 'CACGA'
        solver = NeedlemanWunsch()
        alignment = solver.align(seq_a, seq_b)
        solutions = alignment.solutions

        sol_a1 = '--CGA'
        sol_a2 = 'C--GA'
        sol_b = 'CACGA'

        first_correct = \
            solutions[0][0] == sol_a1 and solutions[1][0] == sol_a2 or \
            solutions[0][0] == sol_a2 and solutions[1][0] == sol_a1
        second_correct = solutions[0][1] == sol_b and solutions[1][1] == sol_b

        self.assertTrue(first_correct and second_correct)

    def test_score(self):
        seq_a = 'CGA'
        seq_b = 'CACGA'
        solver = NeedlemanWunsch()
        alignment = solver.align(seq_a, seq_b)

        self.assertTrue(alignment.score == 1)


if __name__ == '__main__':
    unittest.main()
