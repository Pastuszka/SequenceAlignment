import numpy as np


class Alignment:

    def __init__(self, solutions, score):
        self.solutions = solutions
        self.score = score

    def __str__(self):
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

    def __build_solutions__(self, alignment_matrix, arrow_matrix, seq_a, seq_b):

        class SolutionBuilder:
            def __init__(self, seq_a, seq_b, x, y):
                self.seq_a = seq_a
                self.seq_b = seq_b
                self.x = x
                self.y = y

        n, m = alignment_matrix.shape

        solution = SolutionBuilder('', '', m - 1, n - 1)
        solution_stack = [solution]
        final_solutions = []

        while len(solution_stack) > 0:
            solution = solution_stack.pop()

            if solution.x == 0 and solution.y == 0:
                final_solution = (solution.seq_a[::-1], solution.seq_b[::-1])
                final_solutions.append(final_solution)
                continue

            if arrow_matrix[solution.y, solution.x, 0]:
                new_solution = SolutionBuilder(solution.seq_a + '-',
                                               solution.seq_b
                                               + seq_b[solution.x - 1],
                                               solution.x - 1,
                                               solution.y)
                solution_stack.append(new_solution)
            if arrow_matrix[solution.y, solution.x, 1]:
                new_solution = SolutionBuilder(solution.seq_a
                                               + seq_a[solution.y - 1],
                                               solution.seq_b + '-',
                                               solution.x,
                                               solution.y - 1)
                solution_stack.append(new_solution)
            if arrow_matrix[solution.y, solution.x, 2]:
                new_solution = SolutionBuilder(solution.seq_a
                                               + seq_a[solution.y - 1],
                                               solution.seq_b
                                               + seq_b[solution.x - 1],
                                               solution.x - 1,
                                               solution.y - 1)
                solution_stack.append(new_solution)

        return Alignment(final_solutions, alignment_matrix[n - 1, m - 1])

    def align(self, seq_a, seq_b):
        alignment_matrix, arrow_matrix = self.__align__(seq_a, seq_b)
        return self.__build_solutions__(alignment_matrix, arrow_matrix,
                                        seq_a, seq_b)
