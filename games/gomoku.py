from abc import ABCMeta, abstractmethod

import numpy as np
import copy
import random
import time


# #######################

# #######################

o = 1
x = -1
empty = 0

n_in_a_row = 5
o_win = n_in_a_row
x_win = -n_in_a_row

start_player = 1
board_size = 9

def coordinates_set(width, height):
    """
    根据宽和高生成一个坐标元组集合。
    Get a set of coordinate tuples with width and height.
    :param width: 宽度。 width.
    :param height: 高度。 height.
    :return: <set (x_axis, y_axis)>
    """
    s = set()
    for i in range(width):
        for j in range(height):
            s.add((i, j))
    return s


def get_data_augmentation(array: np.ndarray, operation=lambda a: a):
    """
    获取数据扩充。
    Get data augmentation.
    :param array: 要扩充的数据。 the data to augment.
    :param operation: 数据扩充后要执行的操作。 What to do after data augmentation.
    :return: 已扩充的数据。 Augmented data.
    """

    if array.shape == ():
        return np.zeros(8) + array

    return [operation(array),
            operation(np.rot90(array, 1)),
            operation(np.rot90(array, 2)),
            operation(np.rot90(array, 3)),
            operation(np.fliplr(array)),
            operation(np.rot90(np.fliplr(array), 1)),
            operation(np.rot90(np.fliplr(array), 2)),
            operation(np.rot90(np.fliplr(array), 3))]


class Board:

    def __init__(self):
        self.board = np.zeros((board_size, board_size))
        self.available_actions = coordinates_set(board_size, board_size)
        self.current_player = start_player

    def reset(self):
        self.board = np.zeros((board_size, board_size))
        self.available_actions = coordinates_set(board_size, board_size)
        self.current_player = start_player

    def step(self, action):
        i = action[0]
        j = action[1]
        if (i, j) not in self.available_actions:
            return False
        self.board[i, j] = self.current_player
        self.available_actions.remove((i, j))
        self.current_player = -self.current_player
        return True

    def result(self):
        """
        分析当前局面是否有玩家胜利，或者平局，或者未结束。
        Analyze whether the current situation has a player victory, or a draw, or is not over.
        :return: <tuple (is_over, winner)>
        """
        for piece in coordinates_set(board_size, board_size) - self.available_actions:
            i = piece[0]
            j = piece[1]

            # 横向扫描。 Horizontal scan.
            if j in range(board_size - n_in_a_row + 1):
                s = sum([self.board[i, j + v] for v in range(n_in_a_row)])
                if s == o_win or s == x_win:
                    return True, s / n_in_a_row

            # 纵向扫描。 Vertical scan.
            if i in range(board_size - n_in_a_row + 1):
                s = sum([self.board[i + v, j] for v in range(n_in_a_row)])
                if s == o_win or s == x_win:
                    return True, s / n_in_a_row

            # 斜向右下扫描。 Scan diagonally right.
            if i in range(board_size - n_in_a_row + 1) and j in range(board_size - n_in_a_row + 1):
                s = sum([self.board[i + v, j + v] for v in range(n_in_a_row)])
                if s == o_win or s == x_win:
                    return True, s / n_in_a_row

            # 斜向左下扫描。 Scan diagonally left.
            if i not in range(n_in_a_row - 1) and j in range(board_size - n_in_a_row + 1):
                s = sum([self.board[i - v, j + v] for v in range(n_in_a_row)])
                if s == o_win or s == x_win:
                    return True, s / n_in_a_row

        # 没地儿下了。 Nowhere to move.
        if len(self.available_actions) == 0:
            return True, empty

        return False, empty


class MonteCarloTreeSearch(metaclass=ABCMeta):

    def __init__(self):
        self.root = None

    @abstractmethod
    def run(self, board: Board, times):
        """
        蒙特卡洛树，开始搜索...
        Monte Carlo Tree, start searching...
        :param board:
        :param times: 运行次数。run times.
        :return: 最佳的执行动作。 the best action.
        """


class TreeNode(object):

    def __init__(self, prior_prob, parent=None):
        self.parent = parent
        self.children = {}  # key=action, value=TreeNode
        self.reward = 0  # 该节点的奖励。 Total simulation reward of the node.
        self.visited_times = 0  # 该节点访问的次数。 Total number of visits of the node.
        self.prior_prob = prior_prob  # 父节点选到此节点的概率。 prior probability of the move.

    def is_root(self):
        """
        节点是否根节点。
        Whether the node is the root node.
        :return: <Bool>
        """
        return self.parent is None

    def expand(self, action, probability):
        """
        扩展节点。
        Expand node.
        :param action: 选择的扩展动作。 Selected extended action.
        :param probability: 父节点选到此动作的概率。 prior probability of the move.
        :return: <TreeNode> 扩展出的节点
        """
        # 如果已经扩展过了（一般不可能）。 If it has been extended (generally impossible).
        if action in self.children:
            return self.children[action]

        child_node = TreeNode(prior_prob=probability,
                              parent=self)
        self.children[action] = child_node

        return child_node

    def UCT_function(self, c=5.0):
        greedy = c * self.prior_prob * np.sqrt(self.parent.visited_times) / (1 + self.visited_times)
        if self.visited_times == 0:
            return greedy
        return self.reward / self.visited_times + greedy

    def choose_best_child(self, c=5.0):
        """
        依据 UCT 函数，选择一个最佳的子节点。
        According to the UCT function, select an optimal child node.
        :param c: 贪婪值。 greedy value.
        :return: <(action(x_axis, y_axis), TreeNode)> 最佳的子节点。 An optimal child node.
        """
        return max(self.children.items(), key=lambda child_node: child_node[1].UCT_function(c))

    def backpropagate(self, value):
        """
        反向传输，将结果返回父节点。
        Backpropagate, passing the result to the parent node.
        :param value: 反向传输的值。 The value to be backpropagated.
        """
        self.visited_times += 1
        self.reward += value

        if not self.is_root():
            self.parent.backpropagate(-value)


class Player(metaclass=ABCMeta):

    @abstractmethod
    def take_action(self, board: Board):
        """
        """


class AI_MCTS(MonteCarloTreeSearch, Player):

    def __init__(self, name="AI_MCTS", search_times=2000, greedy_value=5.0):
        super().__init__()
        self.name = name

        self.search_times = search_times  # 树搜索次数。 The search times of tree.
        self.greedy_value = greedy_value  # 贪婪值。 The greedy value.

    def __str__(self):
        return "----- 纯蒙特卡洛树搜索的 AI -----\n" \
               "----- AI with pure MCTS -----\n" \
               "search times: {}\n" \
               "greedy value: {}\n".format(self.search_times, self.greedy_value)

    def reset(self):
        self.root = TreeNode(prior_prob=1.0)

    def take_action(self, board: Board):
        """
        下一步 AI 玩家执行动作。
        The AI player take action next step.
        :param board: 当前棋盘。 Current board.
        :return: <tuple (i, j)> 采取行动时，落子的坐标。 Coordinate of the action.
        """
        self.reset()
        self.run(board, self.search_times)
        action, _ = self.root.choose_best_child(0)
        board.step(action)

        return action

    def output_analysis(self):
        analysis = ""
        # 建立一个 16 * 16 的二维数组。 Build a 16 * 16 array.
        print_array = [["" for _ in range(board_size + 1)] for _ in range(board_size + 2)]

        print_array[0] = ["胜率(%)"] + [str(i + 1) for i in range(board_size)]
        print_array[1] = ["-----"] + ["-----" for i in range(board_size)]
        for row in range(board_size):
            print_array[row + 2][0] = str(row + 1)

        # 填充内容。 Fill Content.
        for i in range(board_size):
            for j in range(board_size):
                if (i, j) in self.root.children:
                    visited_times = float(self.root.children[(i, j)].visited_times)
                    reward = float(self.root.children[(i, j)].reward)
                    print_array[i + 2][j + 1] = "{0:.1f}".format(reward / visited_times * 100)

        # 输出。 Print.
        for i in range(board_size + 2):
            for j in range(board_size + 1):
                analysis += f" | {print_array[i][j]}"
            analysis += " |\n"
        
        return analysis

    def run(self, start_board: Board, times):
        """
        蒙特卡洛树，开始搜索...
        Monte Carlo Tree, start searching...
        :param start_board: 开始搜索的棋盘。 The start state of the board.
        :param times: 运行次数。run times.
        :param running_output_function: 输出 running 的函数。 running output function.
        :param is_stop: 询问是否停止。 Ask whether to stop.
        """
        for i in range(times):
            board = copy.deepcopy(start_board)

            # 扩展节点。
            node = self.traverse(self.root, board)
            node_player = board.current_player

            winner = self.rollout(board)

            value = 0
            if winner == node_player:
                value = 1
            elif winner == -node_player:
                value = -1

            node.backpropagate(-value)
        print("\r                      ", end="\r")

    def traverse(self, node: TreeNode, board: Board):
        """
        扩展子节点。
        Expand node.
        :param node: 当前节点。 Current node.
        :param board: 棋盘。 The board.
        :return: <TreeNode> 扩展出的节点。 Expanded nodes.
        """
        while True:
            if len(node.children) == 0:
                break
            action, node = node.choose_best_child(c=self.greedy_value)
            board.step(action)

        is_over, _ = board.result()
        if is_over:
            return node

        # 扩展所有子节点。 Expand all child node.
        actions = board.available_actions
        probs = np.ones(len(actions)) / len(actions)

        for action, prob in zip(actions, probs):
            _ = node.expand(action, prob)

        return node

    def rollout(self, board: Board):
        """
        模拟。
        Simulation.
        :param board: 棋盘。 The board.
        :return: winner<int> 获胜者。 winner.
        """
        while True:
            is_over, winner = board.result()
            if is_over:
                break
            # 决策下一步。 Decision making next step.
            self.rollout_policy(board)
        return winner

    def rollout_policy(self, board: Board):
        """
        决策函数，在这里随机决策。
        Decision function, random decision here.
        :param board: 棋盘。 The board.
        """

        # 随机执行动作。 Randomly execute actions.
        action = random.choice(list(board.available_actions))

        # 执行。 Action.
        board.step(action)
