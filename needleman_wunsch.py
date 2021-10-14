import numpy as np


class NeedlemanWunsch:
    GAP_PENALTY = -1
    SAME_AWARD = 1
    DIFFERENCE_PENALTY = 0
    MAX_SEQ_LENGTH = 1000

    def __align__(self, seq_a, seq_b):

        len_a = len(seq_a)
        len_b = len(seq_b)
        alignment_array = np.empty((len_a + 1, len_b + 1))

        alignment_array[0, 0] = 0

        for i in range(1, len_a + 1):
            alignment_array[i, 0] = alignment_array[i - 1, 0] + self.GAP_PENALTY

        for j in range(1, len_b + 1):
            alignment_array[0, j] = alignment_array[0, j - 1] + self.GAP_PENALTY

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):

                right_score = alignment_array[i, j - 1] + self.GAP_PENALTY
                down_score = alignment_array[i - 1, j] + self.GAP_PENALTY
                if seq_a[i - 1] == seq_b[j - 1]:
                    diag_score = alignment_array[i - 1, j - 1] + self.SAME_AWARD
                else:
                    diag_score = alignment_array[i - 1, j - 1] \
                                 + self.DIFFERENCE_PENALTY

                best_score = np.max((right_score, down_score, diag_score))

                alignment_array[i, j] = best_score

        return alignment_array, None
