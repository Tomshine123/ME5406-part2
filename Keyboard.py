import gym
import pygame
import uw_robot
from pygame.locals import *

env = gym.make('CustomEnv-v0')
env.reset()
pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("UnderwaterRobot - Keyboard Control")
clock = pygame.time.Clock()

def get_action_from_keyboard():
    action = 0  # Initialize action [main engine, side engine]

    keys = pygame.key.get_pressed()

    if keys[K_UP]:
        action = 2  # Main engine thrust
    if keys[K_LEFT]:
        action = 3  # Left side engine thrust
    if keys[K_RIGHT]:
        action = 1  # Right side engine thrust

    return action

running = True
returns = 0
while running:
    screen.fill((0, 0, 0))
    clock.tick(60)  # Limit the frame rate to 60 FPS

    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    action = get_action_from_keyboard()
    state, reward, done, _ = env.step(action)
    returns += reward
    env.render()

    if done:
        env.reset()
        print('Reward in this episode is %f' % returns)

pygame.quit()
env.close()
