import numpy as np


class Alignment:

    def __init__(self):
        pass

    def __repr__(self):
        pass


class NeedlemanWunsch:
    GAP_PENALTY = -1
    SAME_AWARD = 1
    DIFFERENCE_PENALTY = 0
    MAX_SEQ_LENGTH = 1000

    def __align__(self, seq_a, seq_b):

        len_a = len(seq_a)
        len_b = len(seq_b)
        alignment_matrix = np.empty((len_a + 1, len_b + 1))
        arrow_matrix = np.zeros((len_a + 1, len_b + 1, 3), dtype=bool)

        alignment_matrix[0, 0] = 0

        for i in range(1, len_a + 1):
            alignment_matrix[i, 0] = alignment_matrix[i - 1, 0]\
                                     + self.GAP_PENALTY
            arrow_matrix[i, 0, 1] = True

        for j in range(1, len_b + 1):
            alignment_matrix[0, j] = alignment_matrix[0, j - 1]\
                                     + self.GAP_PENALTY
            arrow_matrix[0, j, 0] = True

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):

                right_score = alignment_matrix[i, j - 1] + self.GAP_PENALTY
                down_score = alignment_matrix[i - 1, j] + self.GAP_PENALTY
                if seq_a[i - 1] == seq_b[j - 1]:
                    diag_score = alignment_matrix[i - 1, j - 1]\
                                 + self.SAME_AWARD
                else:
                    diag_score = alignment_matrix[i - 1, j - 1] \
                                 + self.DIFFERENCE_PENALTY

                best_score = np.max((right_score, down_score, diag_score))

                arrow_matrix[i, j, 0] = right_score == best_score
                arrow_matrix[i, j, 1] = down_score == best_score
                arrow_matrix[i, j, 2] = diag_score == best_score

                alignment_matrix[i, j] = best_score

        return alignment_matrix, arrow_matrix
