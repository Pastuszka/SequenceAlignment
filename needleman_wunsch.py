import numpy as np
import sys
import getopt


class Alignment:

    def __init__(self, solutions: list[tuple[str, str]], score: float):
        self.solutions = solutions
        self.score = score

    def __str__(self):
        output = ""
        output = output + f'SCORE = {self.score}\n'

        for solution in self.solutions:
            output = output + '\n'
            output = output + solution[0] + '\n'
            output = output + solution[1] + '\n'

        return output


def read_fasta_sequence(path: str) -> str:
    sequence = ''
    with open(path) as f:
        for line in f:
            if line[0] == '>':
                continue
            else:
                sequence = sequence + line.strip()
    return sequence


class NeedlemanWunsch:
    GAP_PENALTY = -1
    SAME_AWARD = 1
    DIFFERENCE_PENALTY = 0
    MAX_SEQ_LENGTH = 500

    def __align__(self, seq_a: str, seq_b: str) -> (np.ndarray, np.ndarray):

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

    def __build_solutions__(self, alignment_matrix: np.ndarray,
                            arrow_matrix: np.ndarray,
                            seq_a: str,
                            seq_b: str) -> Alignment:

        class SolutionBuilder:
            def __init__(self, seq_a: str, seq_b: str, x: int, y: int) -> None:
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

    def align(self, seq_a: str, seq_b: str) -> Alignment:

        if max(len(seq_a), len(seq_b)) > self.MAX_SEQ_LENGTH:
            print('Error: exceeded max sequence length')
            sys.exit(0)

        alignment_matrix, arrow_matrix = self.__align__(seq_a, seq_b)
        return self.__build_solutions__(alignment_matrix, arrow_matrix,
                                        seq_a, seq_b)

    def load_config(self, path: str) -> None:
        try:
            config = {}
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    config_line = line.split('=')
                    argument = config_line[0].strip().upper()
                    config[argument] = config_line[1].strip()

            self.GAP_PENALTY = float(config['GAP_PENALTY'])
            self.SAME_AWARD = float(config['SAME_AWARD'])
            self.DIFFERENCE_PENALTY = float(config['DIFFERENCE_PENALTY'])
            self.MAX_SEQ_LENGTH = float(config['MAX_SEQ_LENGTH'])

        except OSError:
            print('Error: file ' + path + ' cannot be read')
            sys.exit(0)
        except KeyError:
            print('Error: config file does not contain required arguments')
            sys.exit(0)
        except ValueError:
            print('Error: config file contains incorrect values')
            sys.exit(0)


def main(argv: list[str]) -> None:
    file_a = ''
    file_b = ''
    config = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv, "a:b:c:o:", ["seqa=", "seqb=",
                                                      "conf=", "ofile="])
    except getopt.GetoptError:
        print('needleman_wunsch.py -a <sequence1> -o <sequenceb> -c <config>'
              ' -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-a", "--seqa"):
            file_a = arg
        elif opt in ("-b", "--seqb"):
            file_b = arg
        elif opt in ("-c", "--conf"):
            config = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg

    if not file_a or not file_b:
        print('Error: missing input file')
        sys.exit(1)

    seq_a = read_fasta_sequence(file_a)
    seq_b = read_fasta_sequence(file_b)

    solver = NeedlemanWunsch()
    if config:
        solver.load_config(config)
    else:
        print("No config provided, using defaults:")
        print(f"GAP_PENALTY = {solver.GAP_PENALTY}")
        print(f"SAME_AWARD = {solver.SAME_AWARD}")
        print(f"DIFFERENCE_PENALTY = {solver.DIFFERENCE_PENALTY}")
        print(f"MAX_SEQ_LENGTH = {solver.MAX_SEQ_LENGTH}")

    alignment = solver.align(seq_a, seq_b)
    if output_file:
        with open(output_file, 'w') as f:
            print(alignment, file=f)
    else:
        print(alignment)


if __name__ == "__main__":
    main(sys.argv[1:])
