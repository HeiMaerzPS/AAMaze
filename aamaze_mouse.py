"""Maze solving utilities - Teaching project for Python learning."""

import logging
from collections import deque, defaultdict
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, field
import time, os, json
from datetime import datetime

import numpy as np

__version__ = '20250926_0917'

MAX_STEPS = 5000
JSON_OUT = True
LOG_LEVEL = logging.INFO #DEBUG

# logger = logging.getLogger(__name__)

# Type aliases for clarity
Position = Tuple[int, int]
Direction = int  # 0=N, 1=E, 2=S, 3=W


def main():
    """Demo the maze and my_mouse functionality."""
    try:
        # Create maze and my_mouse
        maze, start, goal = get_default_maze()
        my_aamaze = AAMaze(maze=maze, start=start, goal=goal)

        # Create and test my_mouse
        my_mouse = AAMouse(aamaze=my_aamaze, max_steps=MAX_STEPS)
        aardvark = my_mouse.move(direction='ahead')
        print(render_with_mouse(maze=my_aamaze, mouse=my_mouse))
        print()

        if True:
            while my_mouse.steps < my_mouse.max_steps:
                if my_mouse.sense_at_goal():
                    print(f"Goal reached in {my_mouse.steps} steps!")
                    print()
                    print(render_with_mouse(maze=my_aamaze, mouse=my_mouse))
                    print()
                    break

                # Get walls
                aardvark = my_mouse.sense_walls()
                # Get unvisited neighbors from current position
                unvisited_relative = my_mouse.sense_unvisited_directions()

                if unvisited_relative:
                    # Choose the best direction toward goal from unvisited neighbors
                    greedy_dir = my_mouse.sense_goal_direction()  # my_mouse.sense_greedy_goal_direction()

                    # If greedy direction is available and unvisited, use it
                    if greedy_dir and greedy_dir in unvisited_relative:
                        chosen_dir = greedy_dir
                    else:
                        # Otherwise, pick the first unvisited direction as fallback
                        chosen_dir = unvisited_relative[0]

                    # Move in that direction
                    if my_mouse.move(direction=chosen_dir):
                        # Successfully moved, add to our path stack
                        # print(render_with_agent(maze=my_aamaze, agent=my_mouse))
                        print(f"step={my_mouse.steps}, last step='{chosen_dir}'")
                        # print()
                    else:
                        print(f"Failed to move {chosen_dir} - unexpected collision")
                        break
                else:
                    my_mouse.backtrack()
                    print(f"step={my_mouse.steps}, last step='backtrack'")
                    # print(render_with_agent(maze=my_aamaze, agent=my_mouse))
                    # print()

            # aardvark = my_aamaze.prepare_render(mouse=my_mouse)
            # if JSON_OUT:
            #     fname_out = os.path.join(os.getcwd(), 'lcl_data', 'data', f"{datetime.now().strftime('%m%d_%H%M')}_maze.json")
            #     if write_json(fname_out=fname_out, data=aardvark):
            #         print(f"... wrote '{fname_out}'")

        # logger.debug("Demo complete!")
        return True

    except Exception as e:
        print(f"Error in main: {e}")
        # logger.exception(f"Error in main: {e}")
        return False


def get_default_maze() -> Tuple[np.ndarray, Position, Position]:
    """Return (grid, start, goal). Grid: 1=wall, 0=free."""
    if True:
        maze = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
            [1,1,1,0,1,0,1,1,1,0,1,0,1,0,1],
            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
            [1,0,1,1,1,1,0,1,1,1,0,1,1,0,1], # less solid wall
            # [1,1,1,1,1,1,1,1,1,1,0,1,1,0,1], # more solid wall
            [1,0,1,0,0,0,0,0,0,1,0,0,0,0,1],
            [1,0,1,0,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,0,1,1,1,0,1,0,1],
            [1,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
            [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1],
            [1,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ], dtype=int)
        start = (1, 1)
        goal = (7, 5)

    else:
        maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)
        start = (1, 1)
        goal = (19, 19)

    return maze, start, goal


# def read_json_maze(*, fname_in):
#     with open(fname_in, 'r') as json_file:
#         lcl_maze = json.load(json_file)
#     lcl_data, lcl_dtype, lcl_shape = lcl_maze['background']['data'], lcl_maze['background']['dtype'], lcl_maze['background']['shape']
#     lcl_maze['background']['data'] = np.array(lcl_data, dtype=lcl_dtype).reshape(lcl_shape)
#     return lcl_maze


def write_json(*, fname_out, data):
    try:
        with open(fname_out, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=4)
        return True
    except Exception as e:
        exc_msg = f"{type(e).__name__}: {e}"
        print(exc_msg)
        return False


class AAMaze:
    """Maze environment with walls and free spaces."""

    def __init__(
        self, *,
        maze: np.ndarray,
        start: Position,
        goal: Position,
        # logger: Optional[logging.Logger] = None,
    ):
        self.maze = np.asarray(maze, dtype=int)
        self.rows, self.cols = self.maze.shape
        self.start = start
        self.goal = goal
        # self.logger = logger or logging.getLogger(f"{__name__}.AAMaze")

        # Validate positions
        for pos, name in [(start, "start"), (goal, "goal")]:
            if not self.is_free(pos=pos):
                raise ValueError(f"{name} {pos} must be on a free cell")

        # and log
        # self.logger.debug(f"{self.rows}x{self.cols} AAMaze instantiated, goal@{self.goal}, start@{self.start}")

    def is_free(self, *, pos: Position) -> bool:
        """Check if position is within bounds and not a wall."""
        try:
            r, c = pos
            lcl_free = (0 <= r < self.rows and 0 <= c < self.cols and self.maze[r, c] == 0)
            return lcl_free
        except (TypeError, IndexError):
            return False

    def neighbours(self, *, pos: Position) -> Dict[str, Optional[Position]]:
        """Get neighbouring positions in each cardinal direction."""
        r, c = pos
        candidates = {"N": (r-1, c), "E": (r, c+1), "S": (r+1, c), "W": (r, c-1),}
        lcl_neighbours = {d: p if self.is_free(pos=p) else None for d, p in candidates.items()}
        return lcl_neighbours

    def prepare_render(self, *, mouse):
        rows, cols = self.maze.shape

        # colors
        COLORS = {
            "wall_gray": "#6E6E6E",
            "start_green": "#009E73",
            "goal_red": "#D55E00",
            "mouse_yellow": "#F0E442",
            "trail_dark": "#0072B2",
            "trail_light": "#A6D5F7",
            "stud_gray": "#B0B0B0",
        }

        # sizes (data units, i.e., fractions of a cell)
        R = {
            "start_goal": 0.30,
            "mouse": 0.30,
            "trail_dark": 0.12,
            "trail_light": 0.12,
            "stud": 0.06,
        }

        # derive sets
        start = tuple(self.start)
        goal = tuple(self.goal)
        mouse_pos = tuple(mouse.pos)

        trail_dark_pts = [p for p in mouse.trail if p not in (mouse_pos, start, goal)]
        visited_backtrack = [p for p in mouse.visited if p not in set(mouse.trail) and p not in (start, goal, mouse_pos)]

        # optional “studs”: faint gray dots at free cell centers
        studs = []
        for r in range(rows):
            for c in range(cols):
                if self.maze[r, c] == 0:
                    studs.append((r, c))

        return {
            "extent": {"rows": rows, "cols": cols},
            # TODO: revert to {"array": self.maze}, this is just to serialise for Jupyter
            # "background": {"__ndarray__": True, "shape": self.maze.shape, "dtype": str(self.maze.dtype), "data": self.maze.tolist()}, #{"array": self.maze},  # 0 free, 1 wall (for imshow if you prefer)
            "background": {"array": self.maze},  # 0 free, 1 wall (for imshow if you prefer)
            "walls": None,  # keep None if using imshow; or supply segments if you switch later
            "studs": {"points": studs, "radius": R["stud"], "color": COLORS["stud_gray"], "alpha": 0.2},

            "start": {"rc": start, "radius": R["start_goal"], "facecolor": COLORS["start_green"], "edgecolor": "black"},
            "goal": {"rc": goal, "radius": R["start_goal"], "facecolor": COLORS["goal_red"], "edgecolor": "black"},

            "mouse": {"rc": mouse_pos, "radius": R["mouse"], "facecolor": COLORS["mouse_yellow"], "edgecolor": "black",
                      "label": {"text": "AA", "color": "black", "weight": "bold"}},

            "trail_dark": {"points": trail_dark_pts, "radius": R["trail_dark"], "facecolor": COLORS["trail_dark"],
                           "alpha": 0.6},
            "trail_light": {"points": visited_backtrack, "radius": R["trail_light"], "facecolor": COLORS["trail_light"],
                            "alpha": 0.2},

            "styles": {
                "grid": {"alpha": 0.3},
                "walls": {"color": COLORS["wall_gray"], "linewidth": 2},
                "title": "Maze",
            }
        }

    def render_ascii(self, *, path: Optional[List[Position]] = None, visited: Optional[Set[Position]] = None) -> str:
        """Render maze as ASCII art."""
        path_set = set(path) if path else set()
        visited_set = set(visited) if visited else set()

        rows = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.start:
                    char = "S"
                elif pos == self.goal:
                    char = "G"
                elif pos in path_set:
                    char = "*"
                elif pos in visited_set:
                    char = "."
                elif self.maze[r, c] == 1:
                    char = "#"
                else:
                    char = " "
                row_chars.append(char)
            rows.append("".join(row_chars))
        return "\n".join(rows)


@dataclass
class AgentState:
    """Immutable state snapshot for loop detection."""
    pos: Position
    direction: Direction
    openings: Tuple[bool, bool, bool, bool]  # N, E, S, W

    def __hash__(self):
        return hash((self.pos, self.direction, self.openings))


class AAMouse:
    """Maze navigation agent with sensors and movement capabilities."""

    # Class constant for direction vectors
    DIRECTIONS = {
        'N': (-1, 0),
        'E': (0, 1),
        'S': (1, 0),
        'W': (0, -1)
    }
    DIR_ORDER = ['N', 'E', 'S', 'W']  # For indexed access

    # TODO: remove direction parameter, init mouse to the first free direction
    def __init__(
            self, *,
            aamaze: AAMaze,
            max_steps: int = MAX_STEPS,
            # logger: Optional[logging.Logger] = None,
            loop_window: int = 50,
            loop_repeat_threshold: int = 3,
    ):
        self.maze = aamaze
        self.pos = aamaze.start
        self.goal = aamaze.goal
        self.max_steps = max_steps
        # Determine initial facing by checking first open passage: E > S > W > N
        pref_abs = ['E', 'S', 'W', 'N']
        chosen_abs = None
        r, c = self.pos
        for d in pref_abs:
            dr, dc = AAMouse.DIRECTIONS[d]
            if self.maze.is_free(pos=(r + dr, c + dc)):
                chosen_abs = d
                break
        self.direction = (AAMouse.DIR_ORDER.index(chosen_abs) if chosen_abs is not None else 0)

        # self.logger = logger or aamaze.logger or logging.getLogger(f"{__name__}.AAMouse")

        # Movement tracking
        self.steps = 0
        self.collisions = 0
        self.visited: Set[Position] = {self.pos}
        self.trail: List[Position] = [self.pos]
        self.visit_counts = defaultdict(int)
        self.visit_counts[self.pos] = 1

        # Loop detection
        self.loop_window = loop_window
        self.loop_threshold = loop_repeat_threshold
        self.state_history = deque(maxlen=loop_window)
        self.state_history.append(self._get_state())

        # Stuck detection
        self.last_pos = None
        self.stuck_counter = 0

        # and log
        # self.logger.debug(f"AAMouse instantiated, pos@{self.pos} facing {AAMouse.DIR_ORDER[self.direction]}")

    def _get_state(self) -> AgentState:
        """Get current state signature for loop detection."""
        nb = self.maze.neighbours(pos=self.pos)
        openings = tuple(nb[d] is not None for d in self.DIR_ORDER)
        return AgentState(self.pos, self.direction, openings)

    def _update_stuck_counter(self):
        """Update counter for stuck detection."""
        if self.pos == self.last_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_pos = self.pos

    def _get_relative_direction(self, *, relative: str) -> str:
        """Convert relative direction to cardinal direction based on current facing."""
        if relative == 'left':
            new_direction_index = (self.direction - 1) % 4
        elif relative == 'right':
            new_direction_index = (self.direction + 1) % 4
        elif relative == 'ahead':
            new_direction_index = self.direction
        else:
            raise ValueError(f"Invalid relative direction: {relative}. Must be 'left', 'right', or 'ahead'")

        return self.DIR_ORDER[new_direction_index]

    def _move_one_step(self) -> bool:
        """Internal method to move one step in the current facing direction."""
        dr, dc = self.DIRECTIONS[self.DIR_ORDER[self.direction]]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)

        if not self.maze.is_free(pos=next_pos):
            self.collisions += 1
            # self.logger.debug(f"Blocked at {next_pos}")
            return False

        # Execute move
        self.pos = next_pos
        self.steps += 1
        self.visited.add(self.pos)
        self.trail.append(self.pos)
        self.visit_counts[self.pos] += 1

        # Update state tracking
        self.state_history.append(self._get_state())
        self._update_stuck_counter()

        return True

    def move(self, *, direction: str):
        if 'ahead' == direction:
            """Move one step ahead. Returns True if successful, False if blocked."""
            return self._move_one_step()
        elif 'left' == direction:
            """Turn left and move one step. Returns True if successful, False if blocked."""
            self.turn(direction='left')
            return self._move_one_step()
        elif 'right' ==  direction:
            """Turn right and move one step. Returns True if successful, False if blocked."""
            self.turn(direction='right')
            return self._move_one_step()
        elif 'back' == direction:
            """Turn around and move one step. Returns True if successful, False if blocked."""
            self.turn(direction='around')
            return self._move_one_step()
        elif 'backtrack' == direction:
            return self.backtrack()
        else:
            return False

    def turn(self, *, direction: str):
        """Turn the agent. Direction: 'left', 'right', or 'around'."""
        turns = {'left': -1, 'right': 1, 'around': 2}
        if direction not in turns:
            raise ValueError(f"Invalid turn direction: {direction}")

        self.direction = (self.direction + turns[direction]) % 4
        self.state_history.append(self._get_state())

    def face(self, *, direction: str):
        """Face a specific cardinal direction (N/E/S/W)."""
        if direction not in self.DIR_ORDER:
            raise ValueError(f"Invalid cardinal direction: {direction}")
        self.direction = self.DIR_ORDER.index(direction)
        self.state_history.append(self._get_state())

    def backtrack(self) -> bool:
        """
        Step back exactly one cell along the trail and normalize the trail.

        Returns
        -------
        bool
            True if a backtrack step was performed, False otherwise.
        """
        if len(self.trail) < 2:
            if not self.sense_wall_back():
                return self.move(direction='back')
            elif not self.sense_wall_ahead():
                return self.move(direction='ahead')
            elif not self.sense_wall_left():
                return self.move(direction='left')
            elif not self.sense_wall_right():
                return self.move(direction='right')
            else:
                return False

        # Previous cell on path and vector from current -> previous
        (cr, cc) = self.pos
        (pr, pc) = self.trail[-2]
        vec = (pr - cr, pc - cc)

        # Invert DIRECTIONS on the fly: (dr, dc) -> 'N'/'E'/'S'/'W'
        back_dir = {vec_: name for name, vec_ in self.DIRECTIONS.items()}[vec]

        # Turn to face the back direction and move
        self.face(direction=back_dir)
        self._move_one_step()

        # Trail normalization:
        # pre:  [..., prev, curr]
        # post: [..., prev, curr, prev]  (step appended prev)
        # fix:  [..., prev]
        if len(self.trail) >= 3:
            self.trail = self.trail[:-3] + [self.pos]

        return True

    # === Sensors ===

    def sense_at_goal(self) -> bool:
        """Check if agent is at the goal."""
        return self.pos == self.goal

    def sense_manhattan_to_goal(self) -> int:
        """Manhattan distance to goal."""
        return abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])

    def sense_wall_ahead(self) -> bool:
        """Check if there's a wall directly ahead."""
        dr, dc = self.DIRECTIONS[self.DIR_ORDER[self.direction]]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        return not self.maze.is_free(pos=next_pos)

    def sense_wall_left(self) -> bool:
        """Check if there's a wall to the left."""
        left_dir = (self.direction - 1) % 4
        dr, dc = self.DIRECTIONS[self.DIR_ORDER[left_dir]]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        return not self.maze.is_free(pos=next_pos)

    def sense_wall_right(self) -> bool:
        """Check if there's a wall to the right."""
        right_dir = (self.direction + 1) % 4
        dr, dc = self.DIRECTIONS[self.DIR_ORDER[right_dir]]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        return not self.maze.is_free(pos=next_pos)

    def sense_wall_back(self) -> bool:
        """Check if there's a wall to the back."""
        back_dir = (self.direction + 2) % 4
        dr, dc = self.DIRECTIONS[self.DIR_ORDER[back_dir]]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)
        return not self.maze.is_free(pos=next_pos)

    def sense_walls(self) -> List[str]:
        walls = []
        if self.sense_wall_ahead():
            walls.append("ahead")
        if self.sense_wall_left():
            walls.append("left")
        if self.sense_wall_right():
            walls.append("right")
        return walls

    def sense_unvisited_directions(self) -> List[str]:
        """
        Get relative directions (ahead, left, right) to unvisited neighbors.
        Does not include 'back' as it's assumed to be the last visited cell.

        Returns
        -------
        List[str]
            List containing 'ahead', 'left', and/or 'right' for unvisited cells.
        """
        unvisited_relative = []

        # Check ahead
        ahead_dir = self.DIR_ORDER[self.direction]
        dr, dc = self.DIRECTIONS[ahead_dir]
        ahead_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=ahead_pos) and ahead_pos not in self.visited:
            unvisited_relative.append('ahead')

        # Check left
        left_dir_idx = (self.direction - 1) % 4
        left_dir = self.DIR_ORDER[left_dir_idx]
        dr, dc = self.DIRECTIONS[left_dir]
        left_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=left_pos) and left_pos not in self.visited:
            unvisited_relative.append('left')

        # Check right
        right_dir_idx = (self.direction + 1) % 4
        right_dir = self.DIR_ORDER[right_dir_idx]
        dr, dc = self.DIRECTIONS[right_dir]
        right_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=right_pos) and right_pos not in self.visited:
            unvisited_relative.append('right')

        # # Check back
        # back_dir_idx = (self.direction + 2) % 4  # 180 degrees opposite
        # back_dir = self.DIR_ORDER[back_dir_idx]
        # dr, dc = self.DIRECTIONS[back_dir]
        # back_pos = (self.pos[0] + dr, self.pos[1] + dc)
        # if self.maze.is_free(pos=back_pos) and back_pos not in self.visited:
        #     unvisited_relative.append('back')

        return unvisited_relative

    def sense_visited_directions(self) -> Dict[str, int]:
        """
        Get visit counts for cells in relative directions (ahead, left, right).
        Only includes free (non-wall) cells that the mouse can potentially move to.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping relative directions ('ahead', 'left', 'right') to
            visit counts. Unvisited cells return 0, walls/barriers are excluded.
        """
        visited_counts = {}

        # Check ahead
        ahead_dir = self.DIR_ORDER[self.direction]
        dr, dc = self.DIRECTIONS[ahead_dir]
        ahead_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=ahead_pos):
            visited_counts["ahead"] = self.visit_counts[ahead_pos]

        # Check left
        left_dir_idx = (self.direction - 1) % 4
        left_dir = self.DIR_ORDER[left_dir_idx]
        dr, dc = self.DIRECTIONS[left_dir]
        left_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=left_pos):
            visited_counts["left"] = self.visit_counts[left_pos]

        # Check right
        right_dir_idx = (self.direction + 1) % 4
        right_dir = self.DIR_ORDER[right_dir_idx]
        dr, dc = self.DIRECTIONS[right_dir]
        right_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=right_pos):
            visited_counts["right"] = self.visit_counts[right_pos]

        # Check back
        back_dir_idx = (self.direction + 2) % 4  # 180 degrees opposite
        back_dir = self.DIR_ORDER[back_dir_idx]
        dr, dc = self.DIRECTIONS[back_dir]
        back_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=back_pos):
            visited_counts["backtrack"] = self.visit_counts[back_pos]

        visited_counts = {k:v for k,v in visited_counts.items() if v>0}

        return visited_counts

    def sense_compass(self) -> Dict[str, str]:
        """
        Get compass mapping of relative directions to cardinal directions.
        Only includes directions that are not blocked by walls.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping relative directions ('ahead', 'left', 'right', 'backtrack')
            to cardinal directions ('N', 'E', 'S', 'W') for unblocked directions.
        """
        compass = {}

        # Check ahead
        ahead_dir = self.DIR_ORDER[self.direction]
        dr, dc = self.DIRECTIONS[ahead_dir]
        ahead_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=ahead_pos):
            compass['ahead'] = ahead_dir

        # Check left
        left_dir_idx = (self.direction - 1) % 4
        left_dir = self.DIR_ORDER[left_dir_idx]
        dr, dc = self.DIRECTIONS[left_dir]
        left_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=left_pos):
            compass['left'] = left_dir

        # Check right
        right_dir_idx = (self.direction + 1) % 4
        right_dir = self.DIR_ORDER[right_dir_idx]
        dr, dc = self.DIRECTIONS[right_dir]
        right_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=right_pos):
            compass['right'] = right_dir

        # Check backtrack
        back_dir_idx = (self.direction + 2) % 4
        back_dir = self.DIR_ORDER[back_dir_idx]
        dr, dc = self.DIRECTIONS[back_dir]
        back_pos = (self.pos[0] + dr, self.pos[1] + dc)
        if self.maze.is_free(pos=back_pos):
            compass['backtrack'] = back_dir

        return compass

    def sense_goal_direction(self) -> Optional[str]:
        """
        Return 'ahead', 'left', 'right', or None if goal is not in straight line of sight
        from the current position in any of these relative directions.

        Returns
        -------
        Optional[str]
            'ahead', 'left', 'right' if goal is in line of sight in that direction, None otherwise.
        """
        r, c = self.pos
        gr, gc = self.maze.goal

        if (r, c) == (gr, gc):
            return None

        def _check_line_of_sight(from_pos, to_pos):
            """Check if there's clear line of sight between two positions."""
            fr, fc = from_pos
            tr, tc = to_pos

            # Must be in same row or column
            if fr != tr and fc != tc:
                return False

            # Check all cells between
            if fr == tr:  # Same row
                step = 1 if tc > fc else -1
                for col in range(fc + step, tc, step):
                    if not self.maze.is_free(pos=(fr, col)):
                        return False
            else:  # Same column
                step = 1 if tr > fr else -1
                for row in range(fr + step, tr, step):
                    if not self.maze.is_free(pos=(row, fc)):
                        return False

            # Check if destination is free
            return self.maze.is_free(pos=to_pos)

        # Check each relative direction
        for relative_dir in ['ahead', 'left', 'right']:
            # Get the cardinal direction for this relative direction
            if relative_dir == 'ahead':
                card_dir_idx = self.direction
            elif relative_dir == 'left':
                card_dir_idx = (self.direction - 1) % 4
            else:  # right
                card_dir_idx = (self.direction + 1) % 4

            card_dir = self.DIR_ORDER[card_dir_idx]

            # Check if goal is in this cardinal direction with line of sight
            if card_dir in ['N', 'S'] and c == gc:  # Same column
                if (card_dir == 'N' and gr < r) or (card_dir == 'S' and gr > r):
                    if _check_line_of_sight((r, c), (gr, gc)):
                        return relative_dir
            elif card_dir in ['E', 'W'] and r == gr:  # Same row
                if (card_dir == 'E' and gc > c) or (card_dir == 'W' and gc < c):
                    if _check_line_of_sight((r, c), (gr, gc)):
                        return relative_dir

        return None

    def sense_absolute_direction(self) -> str:
        """
        Return the absolute cardinal direction the mouse is currently facing.

        Returns
        -------
        str
            One of 'N', 'E', 'S', 'W' representing the current facing direction.
        """
        return self.DIR_ORDER[self.direction]

    def sense_loop_detected(self) -> bool:
        """Check if agent is in a loop."""
        if not self.state_history:
            return False

        current_state = self.state_history[-1]
        count = sum(1 for state in self.state_history if state == current_state)
        return count >= self.loop_threshold

    def sense_stuck(self, *, threshold: int = 30) -> bool:
        """Check if agent has been stuck for too long."""
        return self.stuck_counter >= threshold


def render_with_mouse(*, maze: AAMaze, mouse: AAMouse) -> str:
    """Render maze with agent position marked as 'A'."""
    lines = maze.render_ascii(path=mouse.trail, visited=mouse.visited).splitlines()
    r, c = mouse.pos
    row = list(lines[r])
    row[c] = "M"
    lines[r] = "".join(row)
    return "\n".join(lines)


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
    )
    main()