import random
import numpy as np


class Sudoku:
    def __init__(self, sudoku=None):
        if sudoku is None:
            self.board = [['.' for _ in range(9)] for _ in range(9)]
        elif isinstance(sudoku, Sudoku):
            self.board = sudoku.board
        elif isinstance(sudoku, str):
            # sudoku from file
            if sudoku.endswith('.txt'):
                with open(sudoku, 'r') as f:
                    # if sudoku is encoded only in one line (no \n)
                    lines = f.readlines()
                    if len(lines) == 1 and len(lines[0]) == 81:
                        f.seek(0)
                        self.board = [[c for c in f.read(9)] for _ in range(9)]
                    elif len(lines) == 9 and len(lines[0]) == 10:
                        f.seek(0)
                        self.board = [[c for c in line.strip()] for line in f.readlines()]
                    elif len(lines) == 9 and len(lines[0]) == 12:
                        # skip ' '
                        f.seek(0)
                        self.board = [[c for c in line.strip() if c != ' '] for line in f.readlines()]
                    else:
                        raise ValueError('Invalid sudoku file')
            # sudoku from string
            else:
                if len(sudoku) == 81:
                    self.board = [[c for c in sudoku[i:i + 9]] for i in range(0, 81, 9)]
                else:
                    raise ValueError('Invalid sudoku string')
        else:
            raise TypeError('Invalid sudoku type')

    def __str__(self):
        res = '\n'
        for i in range(9):
            if i % 3 == 0 and i != 0:
                res += '---------------------\n'
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    res += '| '
                res += self.board[i][j] + ' '
            res += '\n'

        return res

    def is_correct(self):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        subgrid = [set() for _ in range(9)]

        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                if num == '.':
                    continue

                subgrid_index = (i // 3) * 3 + j // 3
                if num in rows[i] or num in cols[j] or num in subgrid[subgrid_index]:
                    return False

                rows[i].add(num)
                cols[j].add(num)
                subgrid[subgrid_index].add(num)

        return True

    def is_solved(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == '.':
                    return False
        return True

    def to_string(self):
        res = ''
        for i in range(9):
            for j in range(9):
                res += self.board[i][j]
        return res

    # constraint satisfaction solver
    def constraint_propagation_solver(self, MRV=True, LCV=True):
        def update_domains(domain_matrix, row_pos, col_pos, value):
            previous_domains = []
            # update row
            for col in range(9):
                if value in domain_matrix[row_pos][col]:
                    previous_domains.append(((row_pos, col), value))
                    domain_matrix[row_pos][col].discard(value)

            # update column
            for row in range(9):
                if value in domain_matrix[row][col_pos]:
                    previous_domains.append(((row, col_pos), value))
                    domain_matrix[row][col_pos].discard(value)

            # update subgrid
            subgrid_i = (row_pos // 3) * 3
            subgrid_j = (col_pos // 3) * 3
            for row in range(subgrid_i, subgrid_i + 3):
                for col in range(subgrid_j, subgrid_j + 3):
                    if value in domain_matrix[row][col]:
                        previous_domains.append(((row, col), value))
                        domain_matrix[row][col].discard(value)

            return previous_domains

        def init_domains():
            # init domains
            domain_matrix = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]

            for _i in range(9):
                for _j in range(9):
                    if self.board[_i][_j] != '.':
                        update_domains(domain_matrix, _i, _j, int(self.board[_i][_j]))

            return domain_matrix

        def get_num_of_removed_values(domain_matrix, row_pos, col_pos, value):
            num_of_removed_values = 0
            # update row
            for col in range(9):
                if value in domain_matrix[row_pos][col]:
                    num_of_removed_values += 1

            # update column
            for row in range(9):
                if value in domain_matrix[row][col_pos]:
                    num_of_removed_values += 1

            # update subgrid
            subgrid_i = (row_pos // 3) * 3
            subgrid_j = (col_pos // 3) * 3
            for row in range(subgrid_i, subgrid_i + 3):
                for col in range(subgrid_j, subgrid_j + 3):
                    if value in domain_matrix[row][col]:
                        num_of_removed_values += 1

            return num_of_removed_values

        # LCV heuristic
        def get_values_ordered_by_least_constraining_value(domain_matrix, row_pos, col_pos):
            values = list(domain_matrix[row_pos][col_pos])
            values.sort(key=lambda x: get_num_of_removed_values(domain_matrix, row_pos, col_pos, x))
            return values

        # MRV heuristic: get cell with minimum domain
        def get_min_domain_cell(domain_matrix):
            min_domain = 10
            min_domain_index = (-1, -1)
            for _i in range(9):
                for _j in range(9):
                    if self.board[_i][_j] == '.' and len(domain_matrix[_i][_j]) < min_domain:
                        min_domain = len(domain_matrix[_i][_j])
                        min_domain_index = (_i, _j)

                        if min_domain == 1:
                            return min_domain_index

            return min_domain_index

        def get_nex_cell(row_pos, col_pos):
            # get next cell that is not fixed
            if (row_pos, col_pos) == (8, 8):
                return -1, -1
            else:
                next_pos = (row_pos, col_pos + 1) if col_pos < 8 else (row_pos + 1, 0)
                while self.board[next_pos[0]][next_pos[1]] != '.':  # skip fixed cells
                    if next_pos == (8, 8):
                        return -1, -1

                    next_pos = (next_pos[0], next_pos[1] + 1) if next_pos[1] < 8 else (next_pos[0] + 1, 0)

                return next_pos

        domains = init_domains()
        values_tried = [[set() for _ in range(9)] for _ in range(9)]
        visited_positions = []
        if MRV:
            min_domain_pos = get_min_domain_cell(domains)
        else:
            if self.board[0][0] == '.':
                min_domain_pos = (0, 0)
            else:
                min_domain_pos = get_nex_cell(0, 0)

        backtracking_count = 0

        while min_domain_pos != (-1, -1):
            i, j = min_domain_pos
            if len(domains[i][j] - values_tried[i][j]) == 0:
                # backtrack
                self.board[i][j] = '.'
                values_tried[i][j] = set()
                min_domain_pos, prev_domains = visited_positions.pop()
                for pos, prev_domain in prev_domains:
                    domains[pos[0]][pos[1]].add(prev_domain)

                backtracking_count += 1
            else:
                if LCV:
                    sorted_domain = get_values_ordered_by_least_constraining_value(domains, i, j)
                else:
                    sorted_domain = list(domains[i][j])
                new_value = next(iter(set(sorted_domain) - values_tried[i][j]))
                values_tried[i][j].add(new_value)
                self.board[i][j] = str(new_value)
                prev_domains = update_domains(domains, i, j, new_value)
                visited_positions.append(((i, j), prev_domains))
                if MRV:
                    min_domain_pos = get_min_domain_cell(domains)
                else:
                    # go to next cell
                    min_domain_pos = get_nex_cell(i, j)

        return backtracking_count

    # genetic algorithm solver
    def genetic_solver(self, population_size, mutation_probability, max_restart=50, num_repetitions_before_restart=100, mutation_times=1):
        FIXED_CELLS = [(i, j) for i in range(9) for j in range(9) if self.board[i][j] != '.']

        def sudoku_fitness(_sudoku_instance: [int]):
            # sum conflicts in each row and column => lower is better
            score = 0
            for _i in range(9):
                row_vals = set()
                col_vals = set()
                for _j in range(9):
                    row_vals.add(_sudoku_instance[_i * 9 + _j])
                    col_vals.add(_sudoku_instance[_j * 9 + _i])

                score += 9 - len(row_vals)
                score += 9 - len(col_vals)

            return score

        def sudoku_pu_crossover(_sudoku_instance1: [int], _sudoku_instance2: [int]):
            # crossover two sudoku instances
            # select a random subgrid and cut sudoku instances at the left and right of the subgrid. No subgrid (0, 0)
            subgrid_i = random.randint(0, 2)
            if subgrid_i == 0:
                subgrid_j = random.randint(1, 2)
            else:
                subgrid_j = random.randint(0, 2)

            new_sudoku_instance = [0 for _ in range(81)]
            parent1 = True
            for _i in range(3):
                for _j in range(3):
                    if _i == subgrid_i and _j == subgrid_j:
                        parent1 = False

                    subgrid = [(_i * 3 + _k, _j * 3 + _l) for _k in range(3) for _l in range(3)]
                    for cell in subgrid:
                        if parent1:
                            new_sudoku_instance[cell[0] * 9 + cell[1]] = _sudoku_instance1[cell[0] * 9 + cell[1]]
                        else:
                            new_sudoku_instance[cell[0] * 9 + cell[1]] = _sudoku_instance2[cell[0] * 9 + cell[1]]

            return new_sudoku_instance

        def sudoku_pu_mutation(_sudoku_instance: [int], times=1):
            # mutate sudoku instance
            # swap two cells in the same subgrid randomly
            # select random subgrid and two random cells
            for _ in range(times):
                subgrid_i = random.randint(0, 2)
                subgrid_j = random.randint(0, 2)

                # choose two random cells from the subgrid that are not fixed
                subgrid = list((set([(subgrid_i * 3 + _k, subgrid_j * 3 + _l) for _k in range(3) for _l in range(3)]) - set(FIXED_CELLS)))
                cell1 = random.choice(subgrid)
                subgrid.remove(cell1)
                cell2 = random.choice(subgrid)

                # swap values
                _sudoku_instance[cell1[0] * 9 + cell1[1]], _sudoku_instance[cell2[0] * 9 + cell2[1]] = \
                    _sudoku_instance[cell2[0] * 9 + cell2[1]], _sudoku_instance[cell1[0] * 9 + cell1[1]]

        def sudoku_mutation_v2(_sudoku_instance: [int]):
            # swap two rows and two cols randomly
            # swap two rows
            pass

        def sudoku_generate_initial_population(_population_size):
            # generate initial population
            # no same values in subgrid
            _population = []
            for _ in range(_population_size):
                _sudoku_instance = [int(self.board[_i][_j]) if (_i, _j) in FIXED_CELLS else 0 for _i in range(9) for _j in
                                    range(9)]
                # for each subgrid generate random values without duplicates without moving fixed cells
                for _i in range(3):
                    for _j in range(3):
                        subgrid = [(_i * 3 + _k, _j * 3 + _l) for _k in range(3) for _l in range(3)]
                        subgrid_values = set([_sudoku_instance[_k * 9 + _l] for _k, _l in subgrid])
                        for _k, _l in subgrid:
                            if (_k, _l) not in FIXED_CELLS:
                                _sudoku_instance[_k * 9 + _l] = random.choice(list(set(range(1, 10)) - subgrid_values))
                                subgrid_values.add(_sudoku_instance[_k * 9 + _l])

                _population.append(_sudoku_instance)

            return _population

        def check_all_subgrid(_sudoku_instance: [int]):
            # check if all subgrid are correct
            for _i in range(3):
                for _j in range(3):
                    subgrid = [(_i * 3 + _k, _j * 3 + _l) for _k in range(3) for _l in range(3)]
                    subgrid_values = set([_sudoku_instance[_k * 9 + _l] for _k, _l in subgrid])
                    if len(subgrid_values) != 9:
                        return False

            return True

        population = sudoku_generate_initial_population(population_size)
        generation_count = 0
        mutation_count = 0
        restart_count = 0
        repetition_count = 1
        previous_min_fitness = None

        while restart_count < max_restart:
            if repetition_count == num_repetitions_before_restart:
                population = sudoku_generate_initial_population(population_size)
                generation_count = 0
                mutation_count = 0
                restart_count += 1
                repetition_count = 1
                previous_min_fitness = None

            population_fitness = [sudoku_fitness(sudoku_instance) for sudoku_instance in population]
            new_population = []

            generation_count += 1
            min_fitness = min(population_fitness)
            max_fitness = max(population_fitness)

            if previous_min_fitness == min_fitness:
                repetition_count += 1
            else:
                repetition_count = 0

            print(f'Generation: {generation_count}, min fitness: {min_fitness}, max fitness: {max_fitness}, mutation count: {mutation_count}, restart count: {restart_count}, repetition count: {repetition_count}')

            """
            # check subgrid
            for sudoku_instance in population:
                if not check_all_subgrid(sudoku_instance):
                    print('Wrong subgrid')
                    return
            """

            # check if we have a solution
            if 0 in population_fitness:
                break

            for _ in range(population_size):
                population_index = range(len(population))
                # transform fitness to reverse probability up to 1
                inverse_fitness = [1 / fitness for fitness in population_fitness]
                population_fitness_prob = inverse_fitness / np.sum(inverse_fitness)
                # select x and y (parents) with x != y using weighted probability
                parents = np.random.choice(population_index, size=2, replace=False, p=population_fitness_prob)

                # crossover
                child = sudoku_pu_crossover(population[parents[0]], population[parents[1]])
                # mutate for a small probability
                if random.random() < mutation_probability:
                    mutation_count += 1
                    sudoku_pu_mutation(child, mutation_times)

                new_population.append(child)

            population = new_population
            previous_min_fitness = min_fitness

        # get solution or best sudoku instance
        solution = min(population, key=lambda x: sudoku_fitness(x))
        for i in range(9):
            for j in range(9):
                self.board[i][j] = str(solution[i * 9 + j])

        return generation_count, restart_count
