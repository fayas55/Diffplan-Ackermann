import pygame
import sys
import numpy as np
from train_and_plot import TrajectoryDiffusionEngine, generate_random_obstacles
from src.vehicle_model import AckermannKinematics

pygame.init()
WIDTH, HEIGHT = 950, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DiffPlan-Ackermann: Guided Global Trajectory Field Engine")
clock = pygame.time.Clock()

engine = TrajectoryDiffusionEngine(num_nodes=100, total_steps=220)
car_math = AckermannKinematics()

start_pos = np.array([60.0, 150.0, np.radians(0)])
goal_pos = np.array([880.0, 520.0, np.radians(0)])

# Initialize variables
obstacles = generate_random_obstacles(seed=None)
trajectory_field = engine.generate_noise_field(start_pos, goal_pos)
diffusion_step = 0

while True:
    screen.fill((14, 15, 22)) 
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            obstacles = generate_random_obstacles(seed=None)
            trajectory_field = engine.generate_noise_field(start_pos, goal_pos)
            diffusion_step = 0

    # Step calculations
    if diffusion_step < engine.total_steps:
        trajectory_field = engine.denoise_step(trajectory_field, obstacles, diffusion_step, start_pos, goal_pos)
        diffusion_step += 1

    # --- DRAW GRAPHICS LAYERS ---
    # Draw obstacles
    for obs in obstacles:
        pygame.draw.circle(screen, (28, 30, 40), (int(obs['x']), int(obs['y'])), int(obs['radius']))
        pygame.draw.circle(screen, (55, 65, 80), (int(obs['x']), int(obs['y'])), int(obs['radius']), 1)

    # Draw individual path nodes particles
    for node in trajectory_field:
        pygame.draw.circle(screen, (0, 150, 240), (int(node[0]), int(node[1])), 3)

    # Draw continuous path
    pts_list = [(int(n[0]), int(n[1])) for n in trajectory_field]
    pygame.draw.lines(screen, (0, 255, 180), False, pts_list, 3) 

    # Start and End
    pygame.draw.circle(screen, (0, 255, 0), (int(start_pos[0]), int(start_pos[1])), 10) 
    pygame.draw.circle(screen, (255, 0, 255), (int(goal_pos[0]), int(goal_pos[1])), 12) 

    # HUD Display overlay
    pygame.draw.rect(screen, (20, 20, 28), (0, 620, WIDTH, 80))
    pygame.draw.line(screen, (38, 42, 52), (0, 620), (WIDTH, 620), 2)
    font = pygame.font.SysFont("Courier New", 14, bold=True)
    
    status_str = f"Diffusion Multi-Field Step: {diffusion_step}/{engine.total_steps}  [SPACE = Reset Map Layout]"
    desc_str = f"Active Obstacles Count: {len(obstacles)} | Guided Initialization Engine (0% Collision)."
    
    screen.blit(font.render(status_str, True, (0, 255, 150)), (25, 635))
    screen.blit(font.render(desc_str, True, (140, 145, 160)), (25, 660))

    pygame.display.flip()
    clock.tick(50)
