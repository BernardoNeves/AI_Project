import os
import pickle
import random
import time
from collections import namedtuple
from itertools import product
from math import ceil
from queue import PriorityQueue, Queue

import neat
import numpy as np
import pygame

MapElement = namedtuple("MapElement", ["id", "color"])
pygame.init()
pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEWHEEL])
SCREEN_WIDTH, SCREEN_HEIGHT = int(1280+240), int(720 + 240)

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=0)
FONT = pygame.font.SysFont("Consolas", SCREEN_WIDTH // 50, True)
CLOCK = pygame.time.Clock()
pygame.display.set_caption("AI_PROJECT")
    
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, click_color, togglable=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = FONT.render(text, True, pygame.Color("white"))
        self.color = color
        self.hover_color = hover_color
        self.click_color = click_color
        self.clicked = False
        self.toggled = False
        self.togglable = togglable
        self.prev_clicked = False
        
    def draw(self, surface):
        self.draw_button(surface)
        self.draw_centered_text(surface, self.text)
        
        state = self.handle_toggle() if self.togglable else self.handle_click()
        self.update()
        return state

    def draw_button(self, surface):
        button_color = self.click_color if self.is_clicked() or self.toggled else self.hover_color if self.is_hovered() else self.color
        pygame.draw.rect(surface, button_color, self.rect)

    def is_hovered(self):
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def is_clicked(self):
        return self.is_hovered() and pygame.mouse.get_pressed()[0]
    
    def update(self):
        self.prev_clicked = self.is_clicked()
    
    def handle_click(self):
        if self.is_clicked():
            if not self.clicked:
                self.clicked = True
                return True
        else:
            self.clicked = False
        return False

    def handle_toggle(self):
        if self.is_clicked() and not self.prev_clicked:
            self.toggled = not self.toggled
            return True
        return False
    
    def update_text(self, text):    
        self.text = FONT.render(text, True, pygame.Color("white"))

    def draw_centered_text(self, surface, text):
        surface.blit(text, (self.rect.centerx - text.get_width() // 2, self.rect.centery - text.get_height() // 2))

 
class Camera:
    def __init__(self, x, y, scale, screen_width, screen_height, level):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.level = level
        self.lerp_factor = 0.05
        self.scale = scale
        self.scale_max = scale * 2
        self.scale_min = scale
        self.camera_view = pygame.Rect(0, 0, screen_width, screen_height).inflate(level.cell_size*2, level.cell_size*2)

    def update_scale(self, scale, x, y):
        if self.scale == min(max(scale, self.scale_min), self.scale_max):
            return
        self.scale = min(max(scale, self.scale_min), self.scale_max)
        # self.level.render_map_surface(self)
        if x < 0 or y < 0 or x >= self.level.width or y >= self.level.height:
            return
        self.focus(self.level.grid[x][y])
    
    def apply(self, entity):
        return (
            entity.x * entity.size * self.scale + entity.offset_X - self.x,
            entity.y * entity.size * self.scale + entity.offset_y - self.y,
            entity.size * self.scale,
            entity.size * self.scale,
        )

    def update(self, target):
        scaled_x = target.x * target.size * self.scale - self.screen_width // 2
        scaled_y = target.y * target.size * self.scale - self.screen_height // 2
        scaled_offset_x = SCREEN_WIDTH / self.scale / 4
        scaled_offset_y = SCREEN_HEIGHT / self.scale / 4
        
        if(abs(self.target_x - (scaled_x)) > scaled_offset_x):
            self.target_x = scaled_x + scaled_offset_x * (-1 if self.target_x < scaled_x else 1)
        if(abs(self.target_y - (scaled_y)) > scaled_offset_y):
            self.target_y = scaled_y + scaled_offset_y * (-1 if self.target_y < scaled_y else 1)
        
        if (abs(self.target_x - self.x) < target.size * 2 and abs(self.target_y - self.y) < target.size * 2):
            self.x = self.target_x
            self.y = self.target_y

        self.x += (self.target_x - self.x) * self.lerp_factor
        self.y += (self.target_y - self.y) * self.lerp_factor
        self.x = max(0,min(self.x, target.size * self.scale * len(self.level.grid) - self.screen_width))
        self.y = max(0,min(self.y, target.size * self.scale * len(self.level.grid[0]) - self.screen_height))
    
    def focus(self, target):
        self.target_x = target.x * target.size * self.scale - self.screen_width // 2
        self.target_y = target.y * target.size * self.scale - self.screen_height // 2
        self.x = self.target_x
        self.y = self.target_y


class Cell:
    def __init__(self, x, y, size, id, color, offset_X, offset_y) -> None:
        self.x = x
        self.y = y
        self.parent = None
        self.cost = float('inf')
        self.heuristic = float('inf')
        self.size = size
        self.id = id
        self.color = color
        self.offset_X = offset_X
        self.offset_y = offset_y

    def draw(self, camera, surface=SCREEN, color=None):
        pygame.draw.rect(
            surface,
            self.color if not color else color,
            camera.apply(self)
            if camera
            else (
                self.x * self.size + self.offset_X,
                self.y * self.size + self.offset_y,
                self.size,
                self.size,
            ),
        )

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class Character(Cell):
    def __init__(self, x, y, size, id, color, offset_X, offset_y) -> None:
        super().__init__(x, y, size, id, color, offset_X, offset_y)

    def move(self, x, y, level: map):
        if (x == 0 and y == 0) or self.check_collision(x, y, level):
            return False
        self.x += x
        self.y += y
        return True

    def check_collision(self, x, y, level: map):
        neighbours = level.get_neighbours(int(self.x), int(self.y))

        if (
            (self.x) + x < 0
            or (self.x) + x > (len(level.grid) - 1)
            or (self.y) + y < 0
            or (self.y) + y > (len(level.grid[0]) - 1)
        ):
            return True
        next_cell = level.grid[int(self.x) + x][int(self.y) + y]
        if next_cell in neighbours and next_cell.id == level.elements.WALL.id:
            return True
        return False
   
   
class MapElements:
    WALL = MapElement(id=0, color="#090a14")
    FLOOR = MapElement(id=1, color="#202e37")
    PLAYER = MapElement(id=2, color="#ebede9")
    GOAL = MapElement(id=3, color="#a8ca58")
    FOOD = MapElement(id=4, color="#7a4841")
    WATER = MapElement(id=5, color="#4f8fba")
    VISITED = MapElement(id=6, color="#330000")
    PATH = MapElement(id=7, color="#004400")


class Map:
    def __init__(self, x: float, y: float, width: int, height: int, cell_size: int, floor_rate: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cell_size = ceil(min((SCREEN_WIDTH/width) , (SCREEN_HEIGHT - SCREEN_HEIGHT/20 - 2*SCREEN_HEIGHT/200) / height))
        self.floors = (width * height) * floor_rate
        self.elements = MapElements
        self.goal: Cell = None
        self.map_surface = pygame.Surface((width * self.cell_size, height * self.cell_size))
        self.grid: list[list[Cell]] = []
        self.generate_random_map()
        self.visited: set[Cell] = set()
        self.path = []
        
    def generate_random_map(self) -> None:
        self.set_goal(self.goal)
        self.grid = [
            [Cell( x, y, self.cell_size, self.elements.WALL.id, self.elements.WALL.color, self.x, self.y) for y in range(0, self.height)]
            for x in range(0, self.width)
        ]
        if self.floors >= self.width * self.height:
            for row in self.grid:
                for cell in row:
                    cell.id = self.elements.FLOOR.id
                    cell.color = self.elements.FLOOR.color
            self.fill_border_with_cells()
            self.render_map_surface()
            return 
            
        
        
        position = self.get_random_cell_of_id(self.elements.WALL.id)
        position.x, position.y = self.width // 2, self.height // 2
        walker = Cell(
            position.x,
            position.y,
            self.cell_size,
            self.elements.PLAYER.id,
            self.elements.PLAYER.color,
            self.x,
            self.y,
        )
        self.grid[walker.x][walker.y].id = self.elements.FLOOR.id
        self.grid[walker.x][walker.y].color = self.elements.FLOOR.color
        floors_to_generate = self.floors - 1
        WALL_WEIGHT = 100
        FLOOR_WEIGHT = 1
        while floors_to_generate > 0:
            neighbours = list(self.get_neighbours(walker.x, walker.y))
            for neighbour in neighbours.copy():
                if not (0 <= neighbour.x < self.width and 0 <= neighbour.y < self.height):
                    neighbours.remove(neighbour)
            
            distances = [abs(cell.x - position.x) + abs(cell.y - position.y) for cell in neighbours]

            weights = [
                WALL_WEIGHT * (distance + 1)
                if cell.id == self.elements.WALL.id
                else FLOOR_WEIGHT * (distance * 10 + 1)
                for cell, distance in zip(neighbours, distances)
            ]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            x,y = walker.x, walker.y
            walker = random.choices(neighbours, weights=weights)[0]
            # if self.grid[walker.x][walker.y].id == self.elements.FLOOR.id or not (0 < walker.x < self.width - 1 and 0 < walker.y < self.height - 1):
                # walker = self.get_random_cell_of_id(self.elements.WALL.id)
            
            if not (0 < walker.x < self.width - 1 and 0 < walker.y < self.height - 1) or (walker.x != x and walker.y != y):
                walker = self.grid[x][y]
                continue
                # walker = self.get_random_cell_of_id(self.elements.WALL.id)
                

            self.grid[walker.x][walker.y].id = self.elements.FLOOR.id
            self.grid[walker.x][walker.y].color = self.elements.FLOOR.color
            floors_to_generate -= 1
        self.render_map_surface()

    def fill_border_with_cells(self) -> None:
        if not self.grid or not self.grid[0]:
            return
        rows, cols = len(self.grid), len(self.grid[0])

        for j in range(cols):
            self.grid[0][j] = Cell(
                0, j, self.cell_size, self.elements.WALL.id, self.elements.WALL.color, self.x, self.y
            )
            self.grid[rows - 1][j] = Cell(
                (rows - 1),
                j,
                self.cell_size,
                self.elements.WALL.id,
                self.elements.WALL.color,
                self.x,
                self.y,
            )

        for i in range(rows):
            self.grid[i][0] = Cell(
                i, 0, self.cell_size, self.elements.WALL.id, self.elements.WALL.color, self.x, self.y
            )
            self.grid[i][cols - 1] = Cell(
                i,
                (cols - 1),
                self.cell_size,
                self.elements.WALL.id,
                self.elements.WALL.color,
                self.x,
                self.y,
            )

    def get_neighbours(self, x: int, y: int) -> set[Cell]:
        neighbours = set()
        neighbours = {
            self.grid[x + i][y + j]
            if 0 <= x + i < len(self.grid) and 0 <= y + j < len(self.grid[0])
            else Cell(x + i, y + j, self.cell_size, self.elements.WALL.id, self.elements.WALL.color, self.x, self.y)
            for i, j in product(range(-1, 2), repeat=2)
            if (i, j) != (0, 0)
        }
        neighbours.discard(None)
        return neighbours

    def get_random_cell_of_id(self, id: int) -> Cell or None:
        empty_cells = [cell for row in self.grid for cell in row if cell.id == id]
        return random.choice(empty_cells) if empty_cells else None

    def generate_cell(self, element: MapElement, except_x:int, except_y:int, count: int = 1, max: int = None) -> None:
        if max:
            if count > max:
                count = max
            for row in self.grid:
                for cell in row:
                    if cell.id == element.id:
                        count -= 1
        if count <= 0:
            for row in self.grid:
                for cell in row:
                    if cell.id == element.id:
                        count += 1
                        self.set_cell(cell, self.elements.FLOOR)
                    if count == 0:
                        return
                        
        for _ in range(count):
            position = self.get_random_cell_of_id(self.elements.FLOOR.id)
            while position.x == except_x and position.y == except_y:
                position = self.get_random_cell_of_id(self.elements.FLOOR.id)
            if element == self.elements.GOAL:
                self.set_goal(position)
            else:
                self.set_cell(position, element)
    
    def set_goal(self, cell: Cell) -> None:
        if self.set_cell(cell, self.elements.GOAL):
            self.set_cell(self.goal, self.elements.FLOOR)
            self.goal = cell
        else:
            self.goal = None 
    
    def set_cell(self, cell: Cell, element: MapElement) -> bool:
        if not cell: 
            return False
        if cell.id == element.id or cell.id == self.elements.PLAYER.id:
            self.grid[cell.x][cell.y].id = self.elements.FLOOR.id
            self.grid[cell.x][cell.y].color = self.elements.FLOOR.color
            return False
        
        self.grid[cell.x][cell.y].id = element.id
        self.grid[cell.x][cell.y].color = element.color
        return True

    def render_map_surface(self, camera: Camera = None) -> None:
        self.map_surface.fill(pygame.Color(self.elements.WALL.color))
        [
            cell.draw(None, self.map_surface)
            for row in self.grid
            for cell in row
            if cell.id == self.elements.FLOOR.id
        ]

    def draw_path(self, camera: Camera, surface:pygame.Surface=SCREEN, path=None, visited=None, path_color=None, visited_color=None, draw_visited=False) -> None:
        if visited and draw_visited:
            visited = set(visited)
            if path:
                visited = visited - set(path)
            visited_color = visited_color if visited_color else self.elements.VISITED.color
            [cell.draw(camera, color=visited_color) for cell in visited if cell.id == self.elements.FLOOR.id]
        if path:
            path = set(path[:-1])
            path_color = path_color if path_color else self.elements.PATH.color
            [cell.draw(camera, color=path_color) for cell in path if cell.id == self.elements.FLOOR.id]

    def draw(self, camera: Camera, surface:pygame.Surface=SCREEN) -> None:
        surface.blit(
            pygame.transform.scale(self.map_surface, (int(self.map_surface.get_width() * camera.scale), int(self.map_surface.get_height() * camera.scale))),
            (-camera.x, -camera.y),
        )
        [
            cell.draw(camera, surface)
            for row in self.grid
            for cell in row
            if cell.id != self.elements.WALL.id
            and cell.id != self.elements.FLOOR.id
        ]

    def bfs_pathfinding(self, player: Cell, goal: Cell):
        open_queue = Queue()
        self.visited = set([player])
        open_queue.put(player)

        for row in self.grid:
            for cell in row:
                if cell.id != self.elements.WALL.id:
                    cell.parent = None

        while not open_queue.empty():
            current_node: Cell = open_queue.get()

            if current_node.id == goal.id:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = current_node.parent
                    if current_node == player:
                        break
                self.path = path[::-1]
                return
            neighbors = self.get_neighbours(current_node.x, current_node.y)
            for neighbor in neighbors:
                if current_node.x - neighbor.x != 0 and current_node.y - neighbor.y != 0:
                    continue
                if neighbor not in self.visited and neighbor.id != self.elements.WALL.id:
                    self.visited.add(neighbor)
                    neighbor.parent = current_node
                    open_queue.put(neighbor)

        self.path = []
        return

    def dijkstra_pathfinding(self, player: Cell, goal: Cell):
        player.cost = 0
        open_queue = PriorityQueue()
        self.visited = set([player])
        open_queue.put((player.cost, player))

        for row in self.grid:
            for cell in row:
                if cell.id != self.elements.WALL.id:
                    cell.cost = float('inf')
                    cell.parent = None

        while not open_queue.empty():
            current_node: Cell = open_queue.get()[1]

            if current_node.id == goal.id:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = current_node.parent
                    if current_node == player:
                        break
                self.path = path[::-1]
                return
            neighbors = self.get_neighbours(current_node.x, current_node.y)
            for neighbor in neighbors:
                if current_node.x - neighbor.x != 0 and current_node.y - neighbor.y != 0:
                    continue
                if neighbor.id == self.elements.WALL.id:
                    continue
                new_cost = current_node.cost + 1

                if neighbor not in self.visited or new_cost < neighbor.cost:
                    self.visited.add(neighbor)
                    neighbor.cost = new_cost
                    neighbor.parent = current_node
                    open_queue.put((new_cost, neighbor))

        self.path = []
        return
    
    def astar_pathfinding(self, player: Cell, goal: Cell):
        player.cost = 0
        player.heuristic = self.distance_to_cell(player, goal)
        open_queue = PriorityQueue()
        self.visited = set([player])
        open_queue.put((player.cost + player.heuristic, player))


        for row in self.grid:
            for cell in row:
                if cell.id != self.elements.WALL.id:
                    cell.cost = float('inf')
                    cell.heuristic = float('inf')
                    cell.parent = None

        while not open_queue.empty():
            current_node: Cell = open_queue.get()[1]

            if (current_node.x, current_node.y) == (goal.x, goal.y):
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = current_node.parent
                    if current_node == player:
                        break
                self.path = path[::-1]
                return
            neighbors = self.get_neighbours(current_node.x, current_node.y)
            for neighbor in neighbors:
                if neighbor.id == self.elements.WALL.id or (current_node.x - neighbor.x != 0 and current_node.y - neighbor.y != 0):
                    continue
                new_cost = 1
                new_heuristic = self.distance_to_cell(neighbor, goal)

                if neighbor not in self.visited or new_cost + new_heuristic < neighbor.cost + neighbor.heuristic:
                    self.visited.add(neighbor)
                    neighbor.cost = new_cost
                    neighbor.heuristic = new_heuristic
                    neighbor.parent = current_node
                    open_queue.put((neighbor.cost + neighbor.heuristic, neighbor))

        self.path = []
        return

    def find_closest_cell(self, player: Character, id: int, version:int=0) -> Cell:
        if version == 0:
            self.bfs_pathfinding(player, self.get_random_cell_of_id(id))
            if self.path:
                return self.path[-1]
        if version == 1:
            self.dijkstra_pathfinding(player, self.get_random_cell_of_id(id))
            if self.path:
                return self.path[-1]
        return min(
            (current_cell for row in self.grid for current_cell in row if current_cell.id == id),
            key=lambda current_cell: self.distance_to_cell(player, current_cell),
            default=None
        )
        
    def distance_to_cell(self, cell1: Cell, cell2: Cell) -> int:
        return abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)

    def calculate_fitness(self, genome: neat.DefaultGenome, moves: int):
        genome.fitness = moves*10 if moves > 0 else 0
        return
        
    def get_inputs(self, player: Cell, path_food: list, path_water: list, hunger: float, thirst: float):
        next_food = path_food[0] if path_food else None
        next_water = path_water[0] if path_water else None
        
        if not (hunger<1 and thirst<1 and (next_food or next_water)):
            return [thirst, hunger, 0, 0, 0, 0, 0, 0, 0, 0]
        fx, fy = (next_food.x, next_food.y) if next_food else (0, 0)
        wx, wy = (next_water.x, next_water.y) if next_water else (0, 0)
            
        if player.x > 0:
            fdl = hunger if fx == player.x - 1 and fy == player.y else 0
            wdl = thirst if wx == player.x - 1 and wy == player.y else 0

        if player.y > 0:    
            fdu = hunger if fx == player.x and fy == player.y - 1 else 0
            wdu = thirst if wx == player.x and wy == player.y - 1 else 0
            
        if player.y < self.height - 1:
            fdd = hunger if fx == player.x and fy == player.y + 1 else 0
            wdd = thirst if wx == player.x and wy == player.y + 1 else 0

        if player.x < self.width - 1:
            fdr = hunger if fx == player.x + 1 and fy == player.y else 0
            wdr = thirst if wx == player.x + 1 and wy == player.y else 0
        fdl *= 1- 10*len(path_food)/(self.width*self.height)
        fdu *= 1- 10*len(path_food)/(self.width*self.height)
        fdd *= 1- 10*len(path_food)/(self.width*self.height)
        fdr *= 1- 10*len(path_food)/(self.width*self.height)
        wdl *= 1- 10*len(path_water)/(self.width*self.height)
        wdu *= 1- 10*len(path_water)/(self.width*self.height)
        wdd *= 1- 10*len(path_water)/(self.width*self.height)
        wdr *= 1- 10*len(path_water)/(self.width*self.height)
        
            
        food_directions = [fdl,fdu,fdd,fdr]
        water_directions = [wdl,wdu,wdd,wdr]
        
        output = [hunger, thirst, *food_directions, *water_directions]
        return output
    
    def get_outputs(self, decision: int):
        x, y = 0, 0
        if decision == 0:
            x, y = -1, 0
        elif decision == 1:
            x, y = 0, -1
        elif decision == 2:
            x, y = 0, 1
        elif decision == 3:
            x, y = 1, 0
        return x, y
    
    def train(self, player: Character, genome: neat.DefaultGenome, config: neat.Config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        hunger = 0
        max_hunger = 1
        thirst = 0
        max_thirst = 1
        rate = 8
        move_rate = self.width*self.height/rate
        moves = - 1 - move_rate
        move_rate *= 4
        rate *= 1/(self.width*self.height)
        while hunger < max_hunger and thirst < max_hunger:
            moves += 1
            hunger = min(max(0, hunger), max_hunger) + rate/2
            thirst = min(max(0, thirst), max_thirst) + rate

            self.find_closest_cell(player, self.elements.FOOD.id)
            path_food = self.path.copy()
            
            self.find_closest_cell(player, self.elements.WATER.id)
            path_water = self.path.copy()
            
            net_input = self.get_inputs(player, path_food, path_water, hunger/max_hunger, thirst/max_thirst)
            net_output = net.activate(net_input)
            
            decision = net_output.index(max(net_output))
            x, y = self.get_outputs(decision)
            player.move(x, y, self)

            if self.grid[player.x][player.y].id == self.elements.FOOD.id:
                hunger = min(max(0, hunger-max_hunger), max_hunger)
                self.grid[player.x][player.y].id = self.elements.FLOOR.id
                self.grid[player.x][player.y].color = self.elements.FLOOR.color
                self.generate_cell(self.elements.FOOD, player.x, player.y, 2, 2)
                
            if self.grid[player.x][player.y].id == self.elements.WATER.id:
                thirst = min(max(0, thirst-max_thirst), max_thirst)
                self.grid[player.x][player.y].id = self.elements.FLOOR.id
                self.grid[player.x][player.y].color = self.elements.FLOOR.color
                self.generate_cell(self.elements.WATER, player.x, player.y, 4, 4)
            
        self.calculate_fitness(genome, moves/move_rate)
    
    
def eval_genomes(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config):
    level = Map(0, 0, 10, 10, 50, 1)
    level.generate_cell(level.elements.FOOD, 0, 0, 2, 2)
    level.generate_cell(level.elements.WATER, 0, 0, 4, 4)
    position = level.get_random_cell_of_id(level.elements.FLOOR.id)
    player = Character(position.x, position.y, level.cell_size, level.elements.PLAYER.id, level.elements.PLAYER.color, level.x, level.y)
    map = level.grid.copy()
    px, py = player.x, player.y
    for _, genome in genomes:
        genome.fitness = 0
        level.grid = map.copy()
        player.x, player.y = px, py
        level.train(player, genome, config)


def run_neat(config: neat.Config):
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-4")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(eval_genomes, n=100)

    with open("best_sv.pkl", "wb") as f:
        pickle.dump(winner, f)
       
       
def main(SCREEN):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_sv.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    running = True
    level = Map(0, 0, 16*20, 9*20, 50, 0.8)
    position = level.get_random_cell_of_id(level.elements.FLOOR.id)
    player = Character(position.x, position.y, level.cell_size, level.elements.PLAYER.id, level.elements.PLAYER.color, level.x, level.y)
    camera = Camera(player.x * player.size * 3 - SCREEN_WIDTH // 2, player.y * player.size * 3 - SCREEN_HEIGHT // 2, 1, SCREEN_WIDTH, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - 2*SCREEN_HEIGHT/200, level)

    execution_time = 0
    render = False

    button_backgroud = pygame.Rect(0, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - 2*SCREEN_HEIGHT/200, SCREEN_WIDTH, SCREEN_HEIGHT)

    button_bfs = Button(SCREEN_WIDTH/200*1 + SCREEN_WIDTH/9.5*0, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "BFS", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_dijkstra = Button(SCREEN_WIDTH/200*2 + SCREEN_WIDTH/9.5*1, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Dijkstra", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_astar = Button(SCREEN_WIDTH/200*3 + SCREEN_WIDTH/9.5*2, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "A*", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_test = Button(SCREEN_WIDTH/200*4 + SCREEN_WIDTH/9.5*3, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Test", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_train = Button(SCREEN_WIDTH/200*5 + SCREEN_WIDTH/9.5*4, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Train", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"))
    
    button_play = Button(SCREEN_WIDTH/200*6 + SCREEN_WIDTH/9.5*5, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Play", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_focus = Button(SCREEN_WIDTH/200*7 + SCREEN_WIDTH/9.5*6, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Focus", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"), True)
    button_generate_goal = Button(SCREEN_WIDTH/200*8 + SCREEN_WIDTH/9.5*7, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Goal", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"))
    button_generate_map = Button(SCREEN_WIDTH/200*9 + SCREEN_WIDTH/9.5*8, SCREEN_HEIGHT - SCREEN_HEIGHT/20 - SCREEN_HEIGHT/200, SCREEN_WIDTH/9.5, SCREEN_HEIGHT/20, "Map", pygame.Color("#151d28"), pygame.Color("#202e37"), pygame.Color("#090a14"))
    button_focus.toggled = True
    
    max_hunger = 1
    max_thirst = 1
    rate = 500 /(level.width*level.height)
    net = path_food = path_water = visited_food = visited_water = None
    moves = 0
    start_time = 0
    life_time = 0
    while running:
        mouse_cord = pygame.mouse.get_pos()
        mouse_position = (
            int((mouse_cord[0] + camera.x) // (level.cell_size * camera.scale)),
            int((mouse_cord[1] + camera.y) // (level.cell_size * camera.scale)),
        ) if camera.camera_view.collidepoint(mouse_cord) else (-1, -1)
        
        pathfinding = button_bfs.toggled or button_dijkstra.toggled or button_astar.toggled

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (event.button == 1 or event.button == 3) and  0 <= mouse_position[0] < level.width and 0 <= mouse_position[1] < level.height  and level.grid[mouse_position[0]][mouse_position[1]].id != level.elements.WALL.id:
                    if event.button == 1:
                        player.x, player.y = mouse_position
                    if event.button == 3 and not button_test.toggled:
                        level.set_goal(level.grid[mouse_position[0]][mouse_position[1]])
                    level.path = []
                    
            if event.type == pygame.MOUSEWHEEL:
                target = (player.x, player.y) if button_focus.toggled else mouse_position
                camera.update_scale(camera.scale + 0.1 * event.y, target[0], target[1])
                render = True 
    

        if not level.path and level.visited and not path_food and not path_water:
            level.visited = set()
        
        if pathfinding and not level.path:
            next_node = None
            if level.goal:
                execution_time = time.time()
                if button_bfs.toggled:
                    level.bfs_pathfinding(player, level.goal)
                elif button_dijkstra.toggled:
                    level.dijkstra_pathfinding(player, level.goal)
                elif button_astar.toggled:
                    level.astar_pathfinding(player, level.goal)
                execution_time = time.time() - execution_time
                
        if level.path and button_play.toggled and (button_bfs.toggled or button_dijkstra.toggled or button_astar.toggled):
            next_index = level.path.index(next_node) + 1 if next_node else 0
            next_index = min(next_index, len(level.path) - 1)
            next_node = level.path[next_index] if next_index < len(level.path) else None

            if level.distance_to_cell(player, next_node) >= 1:
                player.move(np.sign(next_node.x - player.x), np.sign(next_node.y - player.y), level)
                moves += 1
        keys = pygame.key.get_pressed()
        camera_speed = 10 * camera.scale
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            camera.x -= camera_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            camera.x += camera_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            camera.y -= camera_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            camera.y += camera_speed
        camera.x = max(0,min(camera.x,level.cell_size * camera.scale * len(level.grid) - camera.screen_width))
        camera.y = max(0,min(camera.y,level.cell_size * camera.scale * len(level.grid[0]) - camera.screen_height))

        if render:
            level.render_map_surface(camera)
            render = False
        CLOCK.tick()

        SCREEN.fill(pygame.Color(level.elements.WALL.color))
        if button_focus.toggled:
            camera.update(player)
            pass
        
        level.draw(camera)
        if visited_food and visited_water and len(visited_food) > len(visited_water):
            level.draw_path(camera, path=None, visited=visited_food, visited_color="#291c20", draw_visited=True)
            level.draw_path(camera, path=None, visited=visited_water, visited_color="#16263f", draw_visited=True)
        else:
            level.draw_path(camera, path=None, visited=visited_water, visited_color="#16263f", draw_visited=True)
            level.draw_path(camera, path=None, visited=visited_food, visited_color="#291c20", draw_visited=True)
        
        level.draw_path(camera, path=path_water, path_color="#253a5e")
        level.draw_path(camera, path=path_food, path_color="#4d2b32")
        level.draw_path(camera, path=level.path, visited=level.visited, draw_visited=True)
        player.draw(camera)
        SCREEN.fill(pygame.Color("#06070E"), button_backgroud)
        
        button_play.update_text("Pause" if button_play.toggled else "Play")
        button_play.draw(SCREEN)
        if button_focus.draw(SCREEN) and button_focus.toggled:
            camera.focus(player)
        if button_generate_map.draw(SCREEN):
            button_test.toggled = False
            level.path = []
            level.grid = []
            level.goal = None
            level.generate_random_map()
            position = level.get_random_cell_of_id(level.elements.FLOOR.id)
            player.x, player.y = position.x, position.y
            camera.focus(player)
            level.set_goal(level.goal)
        if button_generate_goal.draw(SCREEN) and not button_test.toggled:
            level.generate_cell(level.elements.GOAL, player.x, player.y)
            level.path = []
            
        if button_bfs.draw(SCREEN):
            button_bfs.toggled = button_bfs.toggled
            button_dijkstra.toggled = button_astar.toggled = False
            level.path = []
        if button_dijkstra.draw(SCREEN):
            button_dijkstra.toggled = button_dijkstra.toggled
            button_bfs.toggled = button_astar.toggled = False
            level.path = []
        if button_astar.draw(SCREEN):
            button_astar.toggled = button_astar.toggled
            button_bfs.toggled = button_dijkstra.toggled = False
            level.path = []
        
        if button_train.draw(SCREEN):
            SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=0, flags=pygame.HIDDEN)
            run_neat(config)
            SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display=0)
        
        if button_test.draw(SCREEN) or (not button_test.toggled and path_food and path_water):
            if button_test.toggled:
                level.set_goal(level.goal)
                start_time = time.time()
                moves = 0
                hunger = 0
                thirst = 0
                if os.path.exists("best_sv.pkl"):
                    with open("best_sv.pkl", "rb") as f:
                        winner = pickle.load(f)
                    fr = int(0.75*rate*20000)
                    wr = int(1*rate*20000)
                    level.generate_cell(level.elements.FOOD, player.x, player.y, fr, fr)
                    level.generate_cell(level.elements.WATER, player.x, player.y, wr, wr)
                    net = neat.nn.FeedForwardNetwork.create(winner, config)
                else:
                    button_test.toggled = False
            else:
                level.generate_cell(level.elements.FOOD, player.x, player.y, -fr, fr)
                level.generate_cell(level.elements.WATER, player.x, player.y, -wr, wr)
                path_food = path_water = visited_food = visited_water = None
            render = True
        
        if button_test.toggled and net:
            if not (button_bfs.toggled or button_dijkstra.toggled or button_astar.toggled):
                path_food = path_water = visited_food = visited_water = None
            else:
                version = 0 if button_bfs.toggled else 1 if button_dijkstra.toggled else 2 if button_astar.toggled else 0
                closest_food = level.find_closest_cell(player, level.elements.FOOD.id, version)
                if version == 2:
                    level.astar_pathfinding(player, closest_food)
                path_food = level.path
                visited_food = level.visited
                closest_water = level.find_closest_cell(player, level.elements.WATER.id, version)
                if version == 2:
                    level.astar_pathfinding(player, closest_water)
                path_water = level.path
                visited_water = level.visited
                level.path = []
                level.visited = set()
                
                if button_play.toggled:
                    hunger = min(max(0, hunger), max_hunger) + rate/2
                    thirst = min(max(0, thirst), max_thirst) + rate
                    net_input = level.get_inputs(player, path_food, path_water, hunger/max_hunger, thirst/max_thirst)
                    net_output = net.activate(net_input)
                    decision = net_output.index(max(net_output))
                    x, y = level.get_outputs(decision)
                    player.move(x, y, level)
                    life_time = time.time() - start_time
                    moves += 1
                if level.grid[player.x][player.y].id == level.elements.FOOD.id:
                    hunger = min(max(0, hunger-max_hunger), max_hunger)
                    level.grid[player.x][player.y].id = level.elements.FLOOR.id
                    level.grid[player.x][player.y].color = level.elements.FLOOR.color
                    level.generate_cell(level.elements.FOOD, player.x, player.y, fr, fr)
                    render = True
                    
                if level.grid[player.x][player.y].id == level.elements.WATER.id:
                    thirst = min(max(0, thirst-max_thirst), max_thirst)
                    level.grid[player.x][player.y].id = level.elements.FLOOR.id
                    level.grid[player.x][player.y].color = level.elements.FLOOR.color
                    level.generate_cell(level.elements.WATER, player.x, player.y, wr, wr)
                    render = True
                if not (hunger < 1 and thirst < 1):
                    print("Died of " + ("hunger" if hunger > 1 else "thirst") + f"!\n Survived for {life_time:2.3f}s and made {moves} moves")
                    level.generate_cell(level.elements.FOOD, player.x, player.y, -fr, fr)
                    level.generate_cell(level.elements.WATER, player.x, player.y, -wr, wr)
                    path_food = path_water = visited_food = visited_water = net = None
                    render = True
                
        string = f"{CLOCK.get_fps():<3.0f}"+ (f" Steps:{len(level.path):4} Time: {execution_time:3.3f}s" if level.path and (button_bfs.toggled or button_dijkstra.toggled or button_astar.toggled) else f" Steps:{moves:4} Time: {life_time:3.3f}s Hunger:{int(hunger*100):2}% Thirst:{int(thirst*100):2}%" if button_test.toggled else "") 
        SCREEN.blit(
            FONT.render(
                string,
                True,
                (255, 255, 255),
            ),
            (SCREEN_WIDTH/200, SCREEN_HEIGHT/200),
        )
        pygame.display.flip()

        if level.goal and player.x == level.goal.x and player.y == level.goal.y:
            level.path = []
            level.set_goal(level.goal) 
    pygame.quit()


if __name__ == "__main__":
    main(SCREEN)