import heapq
from typing import List, Tuple, Optional

class MazeSolver:
    def __init__(self, maze_file: str):
        """Initialize the maze solver with a maze file."""
        self.maze = self._load_maze(maze_file)
        self.rows = len(self.maze)
        self.cols = len(self.maze[0]) if self.rows > 0 else 0
    
    def _load_maze(self, filename: str) -> List[List[int]]:
        """Load maze from file."""
        maze = []
        with open(filename, 'r') as f:
            for line in f:
                row = [int(x) for x in line.strip().split()]
                maze.append(row)
        return maze
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for A* search."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors (up, down, left, right)."""
        row, col = pos
        neighbors = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.maze[new_row][new_col] == 0:
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Returns True if there is a path from start to end.
        Uses A* and stops early once it's guaranteed we've found the shortest path.
        """
        # Input validation
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            return False
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            return False
        if self.maze[start[0]][start[1]] == 1 or self.maze[end[0]][end[1]] == 1:
            return False
        if start == end:
            return True

        counter = 0
        open_set = [(0, counter, start)]          # (f_score, tie-breaker, position)
        counter += 1

        g_score = {start: 0}
        open_set_hash = {start}

        best_goal_cost = None   # best known g-score to reach the goal

        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            current_g = g_score[current]

            # Goal check
            if current == end:
                if best_goal_cost is None:
                    best_goal_cost = current_g
                else:
                    best_goal_cost = min(best_goal_cost, current_g)

            # Correct early stopping condition for optimal path guarantee
            if best_goal_cost is not None and current_f >= best_goal_cost:
                return True  # No better path can exist → we're done

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                tentative_g = current_g + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f, counter, neighbor))
                        counter += 1
                        open_set_hash.add(neighbor)
                    # Note: in Python heapq we don't decrease-key → we just add new entry
                    # (old worse entries will be ignored when popped later)

        # If we exhaust the open set without satisfying the condition → no path
        return False


def main():
    # Load the maze
    solver = MazeSolver('p1_maze.txt')
    
    # Test cases
    test_cases = [
        ((1, 34), (15, 47)),
        ((1, 2), (3, 39)),
        ((0, 0), (3, 77)),
        ((1, 75), (8, 79)),
        ((1, 75), (39, 40))
    ]
    
    print("Maze Pathfinding Results using A* Algorithm")
    print("=" * 50)
    print(f"Maze size: {solver.rows} x {solver.cols}")
    print("=" * 50)
    
    for i, (start, end) in enumerate(test_cases, 1):
        result = solver.find_path(start, end)
        answer = "YES" if result else "NO"
        print(f"Test {i}: Start {start} → End {end}: {answer}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()