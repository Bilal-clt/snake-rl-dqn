import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#enum used for consatnts as they are immutable(not changeable after creation)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#namedtuple means normal tuple except u can access points 
#using name instead of point[0] etc.. 
#here,Point.x  
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (220,0,0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20000

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        #320, 240 is the center of the screen
        #tail ends at position 300, 240 and 280, 240
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.obstacles = [Point(100, 100), Point(120, 100), Point(140, 100),
                          Point(200, 200), Point(220, 200), Point(240, 200),
                          Point(300, 300), Point(320, 300), Point(340, 300)]
        
        self.score = 0
        self.food = None
        self._place_food() 
        self.frame_iteration = 0

        #function name starting with _ means its private
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        initial_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        final_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    
        
        # 3. check if game over
        reward = 0
        game_over = False

        if final_dist < initial_dist:
            reward = 0.3
        else:
            reward = -0.5
        

        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -24
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    #ai need to check if a point collides or not..
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0 or pt in self.obstacles:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.obstacles:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        #[straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if (np.array_equal(action,[1,0,0])): #straight
            new_dir = clock_wise[idx]
        elif (np.array_equal(action,[0,1,0])): #right
            new_idx = (idx+1)%4
            new_dir = clock_wise[new_idx]
        else:                                  #left
            new_idx = (idx-1)%4
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

