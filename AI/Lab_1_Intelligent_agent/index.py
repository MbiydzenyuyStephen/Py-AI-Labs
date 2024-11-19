  # Import required libraries
import pygame
import random
from queue import Queue, PriorityQueue
from enum import Enum

  # Define cell types using Enum for better organization
class CellType(Enum):
      EMPTY = 0      # Represents empty cells
      OBSTACLE = 1   # Represents walls/obstacles
      TARGET = 2     # Represents the goal
      AGENT = 3      # Represents the agent
      PATH = 4       # Represents the path taken

  # Define color mapping for different cell types
COLORS = {
      CellType.EMPTY: (255, 255, 255),    # White for empty cells
      CellType.OBSTACLE: (0, 0, 0),       # Black for obstacles
      CellType.TARGET: (255, 0, 0),       # Red for target
      CellType.AGENT: (0, 255, 0),        # Green for agent
      CellType.PATH: (0, 0, 255)          # Blue for path
  }

class Environment:
      def __init__(self, width, height, cell_size=30):
          # Initialize environment parameters
          self.width = width
          self.height = height
          self.cell_size = cell_size
          # Create empty grid
          self.grid = [[CellType.EMPTY for _ in range(width)] for _ in range(height)]
          self.agent_pos = None
          self.target_pos = None
        
          # Initialize Pygame window
          pygame.init()
          self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
          pygame.display.set_caption("Intelligent Agent Simulation")
        
          self.generate_environment()

      def generate_environment(self):
          # Place random obstacles (25% of grid)
          for _ in range((self.width * self.height) // 3):
              x = random.randint(0, self.width - 1)
              y = random.randint(0, self.height - 1)
              self.grid[y][x] = CellType.OBSTACLE
        
          # Place agent at top-left corner
          self.agent_pos = (0, 0)
          self.grid[0][0] = CellType.AGENT
        
          # Place target at bottom-right corner
          self.target_pos = (self.width - 1, self.height - 1)
          self.grid[self.height - 1][self.width - 1] = CellType.TARGET

class Agent:
      def __init__(self, environment):
          # Initialize agent with reference to environment
          self.env = environment
          self.pos = environment.agent_pos
          self.path = []
    
      def sense_environment(self):
          # Get valid adjacent cells (up, right, down, left)
          x, y = self.pos
          adjacent = []
          for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
              new_x, new_y = x + dx, y + dy
              if (0 <= new_x < self.env.width and 
                  0 <= new_y < self.env.height and 
                  self.env.grid[new_y][new_x] != CellType.OBSTACLE):
                  adjacent.append((new_x, new_y))
          return adjacent

      def bfs_pathfinding(self):
          # Implement Breadth-First Search algorithm
          start = self.pos
          target = self.env.target_pos
          queue = Queue()
          queue.put(start)
          visited = {start: None}
        
          # BFS main loop
          while not queue.empty():
              current = queue.get()
              if current == target:
                  break
                
              # Check all adjacent cells
              x, y = current
              for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                  next_x, next_y = x + dx, y + dy
                  next_pos = (next_x, next_y)
                
                  # Add valid moves to queue
                  if (0 <= next_x < self.env.width and 
                      0 <= next_y < self.env.height and 
                      next_pos not in visited and 
                      self.env.grid[next_y][next_x] != CellType.OBSTACLE):
                      queue.put(next_pos)
                      visited[next_pos] = current
        
          # Reconstruct path from target to start
          current = target
          path = []
          while current is not None:
              path.append(current)
              current = visited.get(current)
          self.path = list(reversed(path))

      def move(self):
          # Move agent along the calculated path
          if self.path:
              next_pos = self.path.pop(0)
              x, y = self.pos
              self.env.grid[y][x] = CellType.PATH  # Mark visited cells
              new_x, new_y = next_pos
              self.env.grid[new_y][new_x] = CellType.AGENT
              self.pos = next_pos
              return True
          return False

def main():
      # Create environment and agent
      env = Environment(30, 30)
      agent = Agent(env)
      agent.bfs_pathfinding()
      clock = pygame.time.Clock()
      running = True
    
      # Main game loop
      while running:
          # Handle events
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  running = False
        
          # Update agent position
          agent.move()
        
          # Draw environment
          env.screen.fill((128, 128, 128))  # Gray background
          for y in range(env.height):
              for x in range(env.width):
                  cell_type = env.grid[y][x]
                  color = COLORS[cell_type]
                  pygame.draw.rect(env.screen, color,
                               (x * env.cell_size, y * env.cell_size,
                                  env.cell_size - 1, env.cell_size - 1))
        
          pygame.display.flip()
          clock.tick(5)  # Control simulation speed (5 FPS)
        
          # Check if target reached
          if agent.pos == env.target_pos:
              pygame.time.wait(1000)  # Wait 1 second before closing
              running = False
    
      pygame.quit()

if __name__ == "__main__":
      main()
