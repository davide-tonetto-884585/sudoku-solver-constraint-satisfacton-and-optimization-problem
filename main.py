from sudoku import Sudoku
import pandas as pd
import random
import time


def test_from_csv():
    file_path = 'tests/sudoku-3m.csv'
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
        # df = df[df['difficulty'] > 4]
        random_row = random.randint(1, df.index.size)
        row = df.iloc[random_row]

        sudoku = Sudoku(row['puzzle'])
        print(sudoku)
        print('Constraint propagation and backtracking solver')
        start_time = time.time()
        res = sudoku.constraint_propagation_solver()
        end_time = time.time()
        print(sudoku)

        print(f'Solved: {sudoku.is_solved()}')
        print(f'Solved in: {round(end_time - start_time, 4)} seconds')
        print(f'Correct: {sudoku.is_correct()}')
        print(f'Num. of clues: {row["clues"]}')
        print(f'Difficulty: {row["difficulty"]}')
        print(f'Backtracks: {res}')

        print('\n\n----------------------------------------')
        print('Genetic solver')
        sudoku = Sudoku(row['puzzle'])
        start_time = time.time()
        sudoku.genetic_solver(5000, 0.1, 3, 200, 10)
        end_time = time.time()
        print(sudoku)

        print(f'Solved: {sudoku.is_solved()}')
        print(f'Solved in: {round(end_time - start_time, 4)} seconds')
        print(f'Correct: {sudoku.is_correct()}')
        print(f'Num. of clues: {row["clues"]}')
        print(f'Difficulty: {row["difficulty"]}')


def test_from_string(file_path):
    sudoku = Sudoku(file_path)
    print(sudoku)
    print('Constraint propagation and backtracking solver')
    start_time = time.time()
    res = sudoku.constraint_propagation_solver()
    end_time = time.time()
    print(sudoku)

    print(f'Solved: {sudoku.is_solved()}')
    print(f'Solved in: {round(end_time - start_time, 4)} seconds')
    print(f'Correct: {sudoku.is_correct()}')
    print(f'Backtracks: {res}')

    print('\n\n----------------------------------------')
    print('Genetic solver')
    sudoku = Sudoku(file_path)
    start_time = time.time()
    sudoku.genetic_solver(2000, 0.1, 3, 200, 10)
    end_time = time.time()
    print(sudoku)

    print(f'Solved: {sudoku.is_solved()}')
    print(f'Solved in: {round(end_time - start_time, 4)} seconds')
    print(f'Correct: {sudoku.is_correct()}')


def experiments():
    file_path = 'tests/sudoku-3m.csv'
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
        difficulties = [0, 1, 2, 3, 4, 5]
        times = {}
        for difficulty in difficulties:
            new_df = df[df['difficulty'] == difficulty]
            time_list = []
            count = 0
            for i in range(100):
                random_row = random.randint(1, new_df.index.size)
                # row = new_df.iloc[random_row]
                row = new_df.iloc[count]
                sudoku = Sudoku(row['puzzle'])
                start_time = time.time()
                backtracks_count = sudoku.constraint_propagation_solver(False, True)
                end_time = time.time()

                if not sudoku.is_solved() or not sudoku.is_correct():
                    Exception('Sudoku not solved or not correct')

                time_list.append((end_time - start_time, backtracks_count))

                count += 1

            times[difficulty] = time_list

        return times


def experiments_GA():
    file_path = 'tests/sudoku-3m.csv'
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
        difficulties = [0, 1, 2, 3, 4, 5]
        times = {}
        for difficulty in difficulties:
            new_df = df[df['difficulty'] == difficulty]
            time_list = []
            count = 0
            solved_count = 0
            for i in range(10):
                random_row = random.randint(1, new_df.index.size)
                # row = new_df.iloc[random_row]
                row = new_df.iloc[count]
                sudoku = Sudoku(row['puzzle'])
                start_time = time.time()
                metrics = sudoku.genetic_solver(2000, 0.1, 5, 200, 10)
                end_time = time.time()

                if sudoku.is_solved() and sudoku.is_correct():
                    solved_count += 1

                time_list.append((end_time - start_time, metrics, solved_count))

                count += 1

            times[difficulty] = time_list

        return times


def main():
    # import sudoku from file
    file_path = 'tests/sudoku.txt'
    test_from_string(file_path)

    # test_from_csv()

    # requirest csv file with sudokus https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings/versions/2?resource=download
    results = experiments()
    for key, value in results.items():
        mean = sum([x[0] for x in value]) / len(value)
        backtracks = sum([x[1] for x in value]) / len(value)
        print(f'Difficulty: {key}, Mean time: {round(mean, 4)}, Mean backtracks: {round(backtracks, 4)}')

    result_GA = experiments_GA()
    for key, value in result_GA.items():
        mean = sum([x[0] for x in value]) / len(value)
        solved = sum([x[2] for x in value]) / len(value)
        print(f'Difficulty: {key}, Mean time: {round(mean, 4)}, Mean solved: {round(solved, 4)}')


if __name__ == '__main__':
    main()
