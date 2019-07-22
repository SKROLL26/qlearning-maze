import numpy as np
import pandas as pd
from IPython.display import clear_output, display_html
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

import skimage
from skimage.color import colorconv
from skimage.measure import block_reduce


class Agent:

    def __init__(self, maze_path, maze_cell_size):
        self.maze, self._start, self._finish = self._load_maze(
            maze_path, maze_cell_size)
        self.actionSpace = np.array([
            0,  # move up
            1,  # move down
            2,  # move right
            3  # move left
        ])
        self.q_table = np.zeros((self.maze.size, self.actionSpace.size))

    """
    Load maze from file
    """

    def _load_maze(self, path, cell_size):
        maze = skimage.io.imread(path, as_gray=True)
        maze = block_reduce(maze, cell_size, np.max)
        start = tuple(np.argwhere(maze*255 == 64)[0])
        finish = tuple(np.argwhere(maze*255 == 128)[0])
        maze[start] = 1.0
        maze[finish] = 1.0
        return maze, start, finish

    """
    Convert coordinate of 2D matrix to 1D array index
    """

    def _coordinate_to_number(self, coordinate, n_cols):
        return coordinate[0]*n_cols+coordinate[1]

    """
    Train agent
    """

    def learn(self, epochs=100, steps_per_epoch=300, epsilon=0.1,  lr=0.1, gamma=0.7):
        # Reduce chance of the taking random move every epoch
        epsilon_decay = (epsilon/epochs) * 1.5

        for _ in tqdm(range(epochs), "Epoch: "):
            self._reset()

            for _ in range(steps_per_epoch):
                actions = self.actionSpace

                # Get valid actions that actor can perform from current position
                if self.maze[self._pos] == 1.0:
                    upper_cell = (self._pos[0] - 1, self._pos[1])
                    lower_cell = (self._pos[0] + 1, self._pos[1])
                    right_cell = (self._pos[0], self._pos[1] + 1)
                    left_cell = (self._pos[0], self._pos[1] - 1)

                    if self.maze[upper_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[0]))
                    if self.maze[lower_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[1]))
                    if self.maze[right_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[2]))
                    if self.maze[left_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[3]))

                # Select action to perform
                if np.random.uniform() < epsilon:
                    action = np.random.choice(actions)
                else:
                    state_n = self._coordinate_to_number(
                        self._pos, self.maze.shape[1])
                    action = actions[np.argmax(
                        np.take(self.q_table[state_n], actions))]

                if self._step(action, lr, gamma):
                    break

            epsilon -= epsilon_decay

    """
    Show visualization of how agent has learned
    """

    def visualize_result(self, n=1):
        for _ in range(n):
            self._reset()
            done = False
            while not done:
                self.render(0, 0)
                clear_output(wait=True)
                actions = self.actionSpace

                if self.maze[self._pos] == 1.0:
                    upper_cell = (self._pos[0] - 1, self._pos[1])
                    lower_cell = (self._pos[0] + 1, self._pos[1])
                    right_cell = (self._pos[0], self._pos[1] + 1)
                    left_cell = (self._pos[0], self._pos[1] - 1)

                    if self.maze[upper_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[0]))
                    if self.maze[lower_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[1]))
                    if self.maze[right_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[2]))
                    if self.maze[left_cell] == 0.0:
                        actions = np.delete(actions, np.argwhere(
                            actions == self.actionSpace[3]))

                state_n = self._coordinate_to_number(
                    self._pos, self.maze.shape[1])
                action = actions[np.argmax(
                    np.take(self.q_table[state_n], actions))]

                if action == 0:
                    new_pos = (self._pos[0]-1, self._pos[1])
                if action == 1:
                    new_pos = (self._pos[0]+1, self._pos[1])
                if action == 2:
                    new_pos = (self._pos[0], self._pos[1]+1)
                if action == 3:
                    new_pos = (self._pos[0], self._pos[1]-1)

                if new_pos == self._finish:
                    done = True

                self._pos = new_pos

    """
    Reset agent
    """

    def _reset(self):
        self._pos = self._start
        self._pos_prev = None
        self._steps = []

    """
    Perform action and update Q-Table
    """

    def _step(self, action, lr, gamma):
        reward = 100
        penalty = -10

        if action == 0:
            new_pos = (self._pos[0]-1, self._pos[1])
        if action == 1:
            new_pos = (self._pos[0]+1, self._pos[1])
        if action == 2:
            new_pos = (self._pos[0], self._pos[1]+1)
        if action == 3:
            new_pos = (self._pos[0], self._pos[1]-1)

        state_curr = self._coordinate_to_number(self._pos, self.maze.shape[1])
        state_new = self._coordinate_to_number(new_pos, self.maze.shape[1])

        if new_pos == self._pos_prev:
            self.q_table[(state_curr, action)] = (1-lr) * self.q_table[(state_curr, action)] + \
                lr * (penalty + gamma * np.max(self.q_table[state_new]))
        else:
            self.q_table[(state_curr, action)] = (1-lr) * self.q_table[(state_curr, action)] + \
                lr * (gamma * np.max(self.q_table[state_new]))

        self._pos_prev = self._pos
        self._pos = new_pos

        self._steps.append((state_curr, action))

        done = self._pos == self._finish

        if done:
            for i, step in enumerate(self._steps[:-1]):
                self.q_table[step] = (1-lr) * self.q_table[step] + \
                    lr * (reward/len(self._steps) + gamma *
                          np.max(self.q_table[self._steps[i+1][0]]))
                self.q_table[self._steps[-1]] = (1-lr) * self.q_table[self._steps[-1]] + \
                    lr * (reward/len(self._steps))

        return done

    """
    Convert Q-Table to Pandas DataFrame
    """

    def qtable_to_pandas(self):
        df = pd.DataFrame({
            'up': self.q_table[:, 0],
            'down': self.q_table[:, 1],
            'right': self.q_table[:, 2],
            'left': self.q_table[:, 3],
        })
        df.index.name = 'state'
        return df

    """
    Visualize maze and agent
    """

    def render(self, epoch_n, step_n):
        maze = self.maze.copy()
        maze[self._pos] = 0.2
        plt.figure(figsize=(8, 8))
        plt.title("Position: {}".format(self._pos))
        plt.imshow(maze, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
