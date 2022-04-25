# Import and initialize the pygame library
import pygame
import torch
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1280, 880

# Set up the drawing window
screen = pygame.display.set_mode([WIDTH, HEIGHT])

# Load arena.jpg as the background image
background = pygame.image.load("arena.jpeg")

# Scale background image to fill screen, maintaining aspect ratio
background = pygame.transform.scale(background, screen.get_size())

ball_pos = (0, 0, 92)
players = [
    (-2048, -2560, 17),  # myself
    (2048, -2560, 17),  # team mate
    (-2048, 2560, 17),  # opponents
    (2048, 2560, 17),
]

boosts = [
    [0.0, -4240.0, 70.0],
    [-1792.0, -4184.0, 70.0],
    [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0],
    [3072.0, -4096.0, 73.0],
    [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0],
    [0.0, -2816.0, 70.0],
    [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0],
    [-1788.0, -2300.0, 70.0],
    [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0],
    [0.0, -1024.0, 70.0],
    [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0],
    [-1024.0, 0.0, 70.0],
    [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0],
    [-2048.0, 1036.0, 70.0],
    [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0],
    [-1788.0, 2300.0, 70.0],
    [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0],
    [3584.0, 2484.0, 70.0],
    [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0],
    [940.0, 3308.0, 70.0],
    [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0],
    [-1792.0, 4184.0, 70.0],
    [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
]


def game_pos_to_screen_pos(game_pos):
    return (game_pos[1] / 10 + WIDTH / 2, game_pos[0] / 10 + HEIGHT / 2)

def screen_pos_to_game_pos(screen_pos):
    return (5 * (screen_pos[1] - WIDTH / 2), 5 * (screen_pos[0] - HEIGHT / 2))

# Load pytorch model
model = torch.load('model_twos_23000000.pt')
model.eval()

_normal_2v2_batch = [4096, 5120 + 900, 2044] * 4 # 8192, 10240 (compensate for goal depth)
def world_pos_to_map_pos(positions):
  return (positions / _normal_2v2_batch) * [40, 50, 0] + [40, 50, 0]

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((80, 80, 80))

    # Display background image
    screen.blit(background, (0, 0))

    # Draw boosts in orange color
    for boost in boosts:
        pygame.draw.circle(screen, (255, 128, 0), game_pos_to_screen_pos(boost[:2]), 5)

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (200, 200, 200), game_pos_to_screen_pos(ball_pos[:2]), 10)
    # Self
    pygame.draw.circle(screen, (0, 0, 255), game_pos_to_screen_pos(players[0][:2]), 10)
    # Team mate
    pygame.draw.circle(screen, (0, 255, 0), game_pos_to_screen_pos(players[1][:2]), 10)
    # Opponent
    pygame.draw.circle(screen, (255, 0, 0), game_pos_to_screen_pos(players[2][:2]), 10)
    # Opponent
    pygame.draw.circle(screen, (255, 0, 0), game_pos_to_screen_pos(players[3][:2]), 10)


    # On click, move the ball
    if pygame.mouse.get_pressed()[0]:
        position = screen_pos_to_game_pos(pygame.mouse.get_pos())
        print(position)
        ball_pos = (position[0], position[1] , 92)

    model_inputs = np.array([*ball_pos, *players[1], *players[2], *players[3]]) / _normal_2v2_batch
    with torch.no_grad():
        result = model(torch.from_numpy(model_inputs).float())
        stacked_img = np.stack((result,)*3, axis=-1) * 200
        # Load pygame image from buffer
        # print(stacked_img)
        heatmap = pygame.image.frombuffer(stacked_img.astype(np.uint8), (80, 100), 'RGB')
        # Display map
        screen.blit(heatmap, (0, 0))

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
