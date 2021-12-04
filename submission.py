import logic
import random
from AbstractPlayers import *
import time


# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


WEIGHT_MATRIX_UP = [
    [ 0,  1,  2, 4],
    [-1,  0,  1, 2],
    [-2, -1,  0, 1],
    [-4, -2, -1, 0]
]
WEIGHT_MATRIX_DOWN = [
    [0, -1, -2, -4],
    [1,  0, -1, -2],
    [2,  1,  0, -1],
    [4,  2,  1,  0]
]
WEIGHT_MATRIX_LEFT = [
    [4,  2,  1,  0],
    [2,  1,  0, -1],
    [1,  0, -1, -2],
    [0, -1, -2, -4]
]
WEIGHT_MATRIX_RIGHT = [
    [-4, -2, -1, 0],
    [-2, -1,  0, 1],
    [-1,  0,  1, 2],
    [ 0,  1,  2, 4]
]


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    # TODO: erase the following line and implement this function.
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                empty = self.calculate_empty(new_board, time_limit)
                close = self.calculate_close(new_board, time_limit)
                optional_moves_score[move] = score + empty - close

        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed
    def calculate_empty(self, board, time_limit) -> int:
        empty = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] is 0:
                    empty = empty + 1
        return empty

    def calculate_close(self, board, time_limit) -> int:
        close = 0
        for i in range(4):
            for j in range(4):
                if i-1 >= 0:
                    close += abs(board[i][j] - board[i-1][j])
                if i+1 <= 3:
                    close += abs(board[i][j] - board[i+1][j])
                if j-1 >= 0:
                    close += abs(board[i][j] - board[i][j-1])
                if j+1 <= 3:
                    close += abs(board[i][j] - board[i][j+1])
        return close


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.next_move = Move.UP
        self.minimax_move = Move.UP

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        depth = 1
        value = self.minimax(board, 2, 1, depth)
        self.next_move = self.minimax_move
        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            depth += 1
            iteration_start = time.time()
            value = self.minimax(board, 2, 1, depth)
            self.next_move = self.minimax_move
            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.next_move

    # TODO: add here helper functions in class, if needed
    def calculate_utility(self, board) -> float:
        non_empty_squares = 0
        sum_of_non_empty_squares = 0

        for i in range(4):
            for j in range(4):
                if board[i][j] is not 0:
                    non_empty_squares += 1
                    sum_of_non_empty_squares += board[i][j]

        if non_empty_squares is 0:
            return -1

        return sum_of_non_empty_squares / non_empty_squares

    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def minimax(self, board, score, turn, depth) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.calculate_utility(board)

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.minimax(new_board, new_score, 0, depth - 1)
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            self.minimax_move = max_move
            return optional_moves_utility[max_move]
        else:
            min_value = -1
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = 2
                        temp_value = self.minimax(new_board, score + 2, 1, depth - 1)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                        new_board[i][j] = 0
                        new_board[i][j] = 4
                        temp_value = self.minimax(new_board, score + 4, 1, depth - 1)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                        new_board[i][j] = 0
            return min_value


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed
        self.i = 0
        self.j = 0
        self.minimax_i = 0
        self.minimax_j = 0
        self.depth = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.

        start_time = time.time()
        self.depth = 1
        for i in range(4):
            for j in range(4):
                if board[i][j] is 0:
                    self.i = self.minimax_i = i
                    self.j = self.minimax_j = j
        value = self.minimax(board, 2, 0, self.depth)
        if board[self.minimax_i][self.minimax_j] is 0:
            self.i = self.minimax_i
            self.j = self.minimax_j

        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            iteration_start = time.time()
            self.depth += 1
            value = self.minimax(board, 2, 0, self.depth)
            if board[self.minimax_i][self.minimax_j] is 0:
                self.i = self.minimax_i
                self.j = self.minimax_j

            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.i, self.j

    # TODO: add here helper functions in class, if needed
    def calculate_utility(self, board) -> float:
        non_empty_squares = 0
        sum_of_non_empty_squares = 0

        for i in range(4):
            for j in range(4):
                if board[i][j] is not 0:
                    non_empty_squares += 1
                    sum_of_non_empty_squares += board[i][j]

        if non_empty_squares is 0:
            return -1

        return sum_of_non_empty_squares / non_empty_squares

    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def minimax(self, board, score, turn, depth) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.calculate_utility(board)

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.minimax(new_board, new_score, 0, depth - 1)
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            return optional_moves_utility[max_move]
        else:
            min_value = -1
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = 2
                        temp_value = self.minimax(new_board, score + 2, 1, depth - 1)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                            if self.depth is depth:
                                self.minimax_i = i
                                self.minimax_j = j
                        new_board[i][j] = 0

            return min_value


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.next_move = Move.UP
        self.alpha_beta_move = Move.UP

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        depth = 1
        value = self.alpha_beta(board, 2, 1, depth, float('-inf'), float('inf'))
        self.next_move = self.alpha_beta_move
        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            depth += 1
            iteration_start = time.time()
            value = self.alpha_beta(board, 2, 1, depth, float('-inf'), float('inf'))
            self.next_move = self.alpha_beta_move
            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.next_move

    # TODO: add here helper functions in class, if needed
    def calculate_utility(self, board) -> float:
        non_empty_squares = 0
        sum_of_non_empty_squares = 0

        for i in range(4):
            for j in range(4):
                if board[i][j] is not 0:
                    non_empty_squares += 1
                    sum_of_non_empty_squares += board[i][j]

        if non_empty_squares is 0:
            return -1

        return sum_of_non_empty_squares / non_empty_squares

    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def alpha_beta(self, board, score, turn, depth, alpha, beta) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.calculate_utility(board)

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.alpha_beta(new_board, new_score, 0, depth - 1, alpha, beta)
                    max_move = max(optional_moves_utility, key=optional_moves_utility.get)
                    self.alpha_beta_move = move
                    alpha = max(alpha, optional_moves_utility[max_move])
                    if optional_moves_utility[max_move] >= beta:
                        return float('inf')
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            self.alpha_beta_move = max_move
            return optional_moves_utility[max_move]
        else:
            current_min = float('inf')
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = 2
                        temp = self.alpha_beta(new_board, score + 2, 1, depth - 1, alpha, beta)
                        new_board[i][j] = 0
                        current_min = min(current_min, temp)
                        beta = min(beta, current_min)
                        if current_min <= alpha:
                            return float('-inf')
                        new_board[i][j] = 4
                        temp = self.alpha_beta(new_board, score + 4, 1, depth - 1, alpha, beta)
                        new_board[i][j] = 0
                        current_min = min(current_min, temp)
                        beta = min(beta, current_min)
                        if current_min <= alpha:
                            return float('-inf')
            return current_min


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.next_move = Move.UP
        self.alpha_beta_move = Move.UP

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        depth = 1
        value = self.expectimax(board, 2, 1, depth, 2)
        self.next_move = self.alpha_beta_move
        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            depth += 1
            iteration_start = time.time()
            value = self.expectimax(board, 2, 1, depth, 2)
            self.next_move = self.alpha_beta_move
            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.next_move

    # TODO: add here helper functions in class, if needed
    def calculate_utility(self, board) -> float:
        non_empty_squares = 0
        sum_of_non_empty_squares = 0

        for i in range(4):
            for j in range(4):
                if board[i][j] is not 0:
                    non_empty_squares += 1
                    sum_of_non_empty_squares += board[i][j]

        if non_empty_squares is 0:
            return -1

        return sum_of_non_empty_squares / non_empty_squares

    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def expectimax(self, board, score, turn, depth, value) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.calculate_utility(board)

        if turn is 3:
            value_2 = 0.9 * self.expectimax(board, score, 0, depth - 1, 2)
            value_4 = 0.1 * self.expectimax(board, score, 0, depth - 1, 4)
            return value_2 + value_4

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.expectimax(new_board, new_score, 0, depth - 1, 2)
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            self.alpha_beta_move = max_move
            return optional_moves_utility[max_move]
        else:
            min_value = -1
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = value
                        temp_value = self.expectimax(new_board, score + value, 1, depth - 1, value)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                        new_board[i][j] = 0
            return min_value


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed
        self.i = 0
        self.j = 0
        self.minimax_i = 0
        self.minimax_j = 0
        self.depth = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        self.depth = 1
        for i in range(4):
            for j in range(4):
                if board[i][j] is 0:
                    self.i = self.minimax_i = i
                    self.j = self.minimax_j = j
        temp = self.expectimax(board, 2, 0, self.depth, value)
        if board[self.minimax_i][self.minimax_j] is 0:
            self.i = self.minimax_i
            self.j = self.minimax_j
        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            iteration_start = time.time()
            self.depth += 1
            temp = self.expectimax(board, 2, 0, self.depth, value)
            if board[self.minimax_i][self.minimax_j] is 0:
                self.i = self.minimax_i
                self.j = self.minimax_j
            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.i, self.j

    # TODO: add here helper functions in class, if needed
    def calculate_utility(self, board) -> float:
        non_empty_squares = 0
        sum_of_non_empty_squares = 0

        for i in range(4):
            for j in range(4):
                if board[i][j] is not 0:
                    non_empty_squares += 1
                    sum_of_non_empty_squares += board[i][j]

        if non_empty_squares is 0:
            return -1

        return sum_of_non_empty_squares/non_empty_squares

    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def expectimax(self, board, score, turn, depth, value) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.calculate_utility(board)

        if turn is 3:
            value_2 = 0.9 * self.expectimax(board, score, 0, depth - 1, 2)
            value_4 = 0.1 * self.expectimax(board, score, 0, depth - 1, 4)
            return value_2 + value_4

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.expectimax(new_board, new_score, 0, depth - 1, 2)
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            return optional_moves_utility[max_move]
        else:
            min_value = -1
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = value
                        temp_value = self.expectimax(new_board, score + value, 1, depth - 1, value)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                            if self.depth is depth:
                                self.minimax_i = i
                                self.minimax_j = j
                        new_board[i][j] = 0
            return min_value


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.next_move = Move.UP
        self.alpha_beta_move = Move.UP

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        start_time = time.time()
        depth = 1
        value = self.expectimax(board, 2, 1, depth, 2)
        self.next_move = self.alpha_beta_move
        iteration_time = time.time() - start_time
        upper_bound = 16 * 4 * iteration_time
        accumulated_time = time.time() - start_time

        while accumulated_time + upper_bound < time_limit - 0.1:
            depth += 1
            iteration_start = time.time()
            value = self.expectimax(board, 2, 1, depth, 2)
            self.next_move = self.alpha_beta_move
            iteration_time = time.time() - iteration_start
            upper_bound = 16 * 4 * iteration_time
            accumulated_time = time.time() - start_time

        return self.next_move

    # TODO: add here helper functions in class, if needed
    def is_final(self, board) -> bool:
        count = 0
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                count += 1

        if count is 0:
            return True
        else:
            return False

    def corners(self, board):
        up = 0
        down = 0
        right = 0
        left = 0

        for i in range(4):
            for j in range(4):
                up += board[i][j] * WEIGHT_MATRIX_UP[i][j]
                down += board[i][j] * WEIGHT_MATRIX_DOWN[i][j]
                right += board[i][j] * WEIGHT_MATRIX_RIGHT[i][j]
                left += board[i][j] * WEIGHT_MATRIX_LEFT[i][j]

        return max(up, down, right, left)

    def utility(self, board):
        weight_matrix = self.corners(board)
        return weight_matrix

    def expectimax(self, board, score, turn, depth, value) -> float:
        if self.is_final(board):
            return score

        if depth is 0:
            return self.utility(board)

        if turn is 3:
            value_2 = 0.9 * self.expectimax(board, score, 0, depth - 1, 2)
            value_4 = 0.1 * self.expectimax(board, score, 0, depth - 1, 4)
            return value_2 + value_4

        if turn is 1:
            optional_moves_utility = {}
            for move in Move:
                new_board, done, new_score = commands[move](board)
                if done:
                    optional_moves_utility[move] = self.expectimax(new_board, new_score, 0, depth - 1, 2)
            max_move = max(optional_moves_utility, key=optional_moves_utility.get)
            self.alpha_beta_move = max_move
            return optional_moves_utility[max_move]
        else:
            min_value = -1
            new_board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

            for i in range(4):
                for j in range(4):
                    new_board[i][j] = board[i][j]

            for i in range(4):
                for j in range(4):
                    if new_board[i][j] is 0:
                        new_board[i][j] = value
                        temp_value = self.expectimax(new_board, score + value, 1, depth - 1, value)
                        if min_value is -1 or min_value > temp_value:
                            min_value = temp_value
                        new_board[i][j] = 0
            return min_value

