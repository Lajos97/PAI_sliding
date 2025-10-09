"""Solving the sliding puzzle (or 8-puzzle) with local search. You are going to
implement the heuristics seen at the lecture, hill climbing, and tabu search.
The states are row-major flattened versions of the puzzle.

The strategy I recommend is to implement the simplest heuristic (# of misplaced
tiles) and the simpler search algorithm (hill climbing) first, check that they
work on easier puzzles, and continue with the rest of the heuristics and tabu
search.

You only need to modify the code in the "YOUR CODE HERE" sections. """

import random
from functools import partial
from collections import deque

from typing import Callable, Generator, Optional, Any

import FreeSimpleGUI as sg #type: ignore

from framework.gui import BoardGUI
from framework.board import Board

BLANK_IMAGE_PATH = 'tiles/chess_blank_scaled.png'
sg.set_options(scaling=4)

"""The state is a tuple with 9 integers. For convenience we just define it as a
tuple of integers."""
State = tuple[int, ...]

goal: State = (1, 2, 3, 8, 0, 4, 7, 6, 5)


class SlidingBoard(Board):
    def __init__(self, start: State):
        self.m = 3
        self.n = 3
        self.create_board()
        self.update_from_state(start)

    def update_from_state(self, state: State) -> None:
        """Updates the board from the state of the puzzle."""
        for i, field in enumerate(state):
            self.board[i // self.n][i % self.n] = field

    def _default_state_for_coordinates(self, i: int, j: int) -> int:
        return 0


class SlidingProblem:
    """The search problem for the sliding puzzle."""

    def __init__(self, start_permutations: int = 10):
        self.goal : State = goal
        self.nil : State = (0,) * 9
        self.possible_slides = (
            (1, 3),         # from the upper left corner, you can move to right (+1) or down (+3)
            (-1, 1, 3),     # from the upper middle tile, you can move to left (-1), right (+1) or down (+3)
            (-1, 3),        # ...
            (-3, 1, 3),
            (-1, 1, -3, 3),
            (-1, -3, 3),
            (1, -3),
            (-1, 1, -3),
            (-1, -3),
        )
        self.start : State = self.generate_start_state(start_permutations)

    def start_state(self) -> State:
        return self.start

    def next_states(self, state: State) -> set[State]:
        ns = set()
        empty_ind = state.index(0)
        slides = self.possible_slides[empty_ind]
        for s in slides:
            ns.add(self.switch(state, empty_ind, empty_ind + s))
        return ns

    def is_goal_state(self, state: State) -> bool:
        return state == self.goal

    def generate_start_state(self, num_permutations: int) -> State:
        start = self.goal
        for _ in range(num_permutations):
            empty_ind = start.index(0)
            slides = self.possible_slides[empty_ind]
            start = self.switch(start, empty_ind, empty_ind + random.choice(slides))
        return start

    def switch(self, current: State, first: int, second: int) -> State:
        new = list(current)
        new[first], new[second] = new[second], new[first]
        return tuple(new)

HeuristicFunction = Callable[[State], int]
Algorithm = Callable[[SlidingProblem, HeuristicFunction], Generator]

# YOUR CODE HERE

# search


def hill_climbing(
    problem: SlidingProblem, f: HeuristicFunction
) -> Generator[State, None, None]:
    current = problem.start_state()
    parent: Optional[State] = None
    while True:
        yield current
        if problem.is_goal_state(current):
            break
        neighbors = problem.next_states(current)
        if parent is not None and parent in neighbors and len(neighbors) > 1:
            neighbors.remove(parent)
        if not neighbors:
            break
        best_next = min(neighbors, key=f)
        if f(best_next) < f(current):
            parent, current = current, best_next
        else:
            break


def tabu_search(
    problem: SlidingProblem,
    f: HeuristicFunction,
    tabu_len: int = 10,
    long_time: int = 1000,
) -> Generator[State, None, None]:
    current = problem.start_state()
    best = current
    best_val = f(best)
    tabu_queue: deque[State] = deque(maxlen=tabu_len)
    tabu_set: set[State] = set()
    steps_since_improvement = 0
    while True:
        yield current
        if problem.is_goal_state(current):
            break
        neighbors = problem.next_states(current)
        if not neighbors:
            break
        cand = []
        for s in neighbors:
            hv = f(s)
            if (s not in tabu_set) or (hv < best_val):
                cand.append((hv, s))
        if not cand:
            cand = [(f(s), s) for s in neighbors]
        cand.sort(key=lambda t: t[0])
        next_state = cand[0][1]
        tabu_queue.append(current)
        tabu_set = set(tabu_queue)
        current = next_state
        cur_val = f(current)
        if cur_val < best_val:
            best, best_val = current, cur_val
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            if steps_since_improvement >= long_time:
                break

# heuristics


def misplaced(state: State) -> int:
    return sum(1 for i, v in enumerate(state) if v != 0 and v != goal[i])


def manhattan(state: State) -> int:
    goal_pos: dict[int, tuple[int, int]] = {val: (i // 3, i % 3) for i, val in enumerate(goal)}
    total = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        r, c = i // 3, i % 3
        rg, cg = goal_pos[v]
        total += abs(r - rg) + abs(c - cg)
    return total

# END OF YOUR CODE

start_permutations = 10

sliding_draw_dict = {
    i: (f"{i}", ("black", "lightgrey"), BLANK_IMAGE_PATH) for i in range(1, 9)
}
sliding_draw_dict.update({0: (" ", ("black", "white"), BLANK_IMAGE_PATH)})

sliding_problem = SlidingProblem(start_permutations)
board = SlidingBoard(sliding_problem.start)
board_gui = BoardGUI(board, sliding_draw_dict)

algorithms : dict[str, Algorithm] = {"Hill climbing": hill_climbing, "Tabu search": tabu_search}

heuristics : dict[str, HeuristicFunction] = {"Misplaced": misplaced, "Manhattan": manhattan,}

layout = [
    [
        sg.Column(board_gui.board_layout),
        sg.Frame("Log", [[sg.Output(size=(30, 10), key="log")]]),
    ],
    [
        sg.Frame(
            "Algorithm settings",
            [
                [
                    sg.T("Algorithm: "),
                    sg.Combo(
                        [algo for algo in algorithms], key="algorithm", readonly=True, default_value="Hill climbing"
                    ),
                    sg.T("Tabu length:"),
                    sg.Spin(
                        values=list(range(1000)),
                        initial_value=10,
                        key="tabu_len",
                        size=(5, 1),
                    ),
                ],
                [
                    sg.T("Heuristics: "),
                    sg.Combo(
                        [heur for heur in heuristics], key="heuristics", readonly=True, default_value="Misplaced"
                    ),
                ],
                [sg.Button("Change", key="Change_algo")],
            ],
        ),
        sg.Frame(
            "Problem settings",
            [
                [
                    sg.T("Starting permutations: "),
                    sg.Spin(
                        values=list(range(1, 100)),
                        initial_value=start_permutations,
                        key="start_permutations",
                        size=(5, 1),
                    ),
                ],
                [sg.Button("Change", key="Change_problem")],
            ],
        ),
    ],
    [sg.T("Steps: "), sg.T("0", key="steps", size=(7, 1), justification="right")],
    [sg.Button("Restart"), sg.Button("Step"), sg.Button("Go!"), sg.Button("Exit")],
]

window = sg.Window(
    "Sliding puzzle problem", layout, default_button_element_size=(10, 1), location=(0,0), finalize=True
)

starting = True
go = False
steps = 0

while True:  # Event Loop
    event, values = window.Read(0)
    window.Element("tabu_len").Update(disabled=values["algorithm"] != "Tabu search")
    window.Element("Go!").Update(text="Stop!" if go else "Go!")
    if event is None or event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Change_algo" or event == "Change_problem" or starting:
        if event == "Change_problem":
            start_permutations = int(values["start_permutations"])
            sliding_problem = SlidingProblem(start_permutations)
        algorithm : Any = algorithms[values["algorithm"]]
        heuristic = heuristics[values["heuristics"]]
        if algorithm is tabu_search:
            tabu_len = int(values["tabu_len"])
            algorithm = partial(algorithm, tabu_len=tabu_len)
        algorithm = partial(algorithm, f=heuristic)
        path = algorithm(sliding_problem)
        steps = 0
        window.Element("log").Update("")
        starting = False
        stepping = True
    if event == "Restart":
        path = algorithm(sliding_problem)
        steps = 0
        window.Element("log").Update("")
        stepping = True
    if event == "Step" or go or stepping:
        try:
            state = next(path)
        except StopIteration:
            pass
        else:
            print(f"{state}: {heuristic(state)}")
            window.Element("steps").Update(f"{steps}")
            steps += 1
        board.update_from_state(state)
        board_gui.update()
        stepping = False
    if event == "Go!":
        go = not go

window.Close()
