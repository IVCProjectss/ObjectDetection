import pygame
import cv2
import time
from pygame.locals import *
from object_detection import ObjectDetection

# Initializate YOLO
detector = ObjectDetection(model_path="yolov8s.pt")

# Initializate camera
cam = ObjectDetection.initializate_cameras()

# Inicializar Pygame
pygame.init()

# Screen
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Breakout')

# Variables
bg = (234, 218, 184)
paddle_outline = (100, 100, 100)
paddle_green = (86, 174, 87)
text_col = (78, 81, 139)
block_red = (242, 85, 96)
block_green = (86, 174, 87)
block_blue = (69, 177, 232)
font = pygame.font.SysFont('Constantia', 30)
clock = pygame.time.Clock()
fps = 30
live_ball = False
cols = 6
rows = 6
game_over = 0

# Draw Text
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

# Wall class
class Wall:
    def __init__(self):
        self.width = screen_width // cols
        self.height = 50

    def create_wall(self):
        self.blocks = []
        for row in range(rows):
            block_row = []
            for col in range(cols):
                block_x = col * self.width
                block_y = row * self.height
                rect = pygame.Rect(block_x, block_y, self.width, self.height)
                strength = 3 if row < 2 else 2 if row < 4 else 1
                block_row.append([rect, strength])
            self.blocks.append(block_row)

    def draw_wall(self):
        for row in self.blocks:
            for block in row:
                block_col = block_blue if block[1] == 3 else block_red if block[1] == 2 else block_red
                pygame.draw.rect(screen, block_col, block[0])
                pygame.draw.rect(screen, bg, block[0], 2)

# Paddle class
class Paddle:
    def __init__(self, color):
        self.width = int(screen_width / cols)
        self.color = color
        self.height = 20
        self.reset()

    def move(self, object_x):
        if object_x is not None:
            self.rect.x = object_x - (self.width // 2)
        # Ensure paddle stays within screen bounds
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > screen_width:
            self.rect.right = screen_width

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, paddle_outline, self.rect, 3)

    def reset(self):
        self.x = int((screen_width / 2) - (self.width / 2))
        self.y = screen_height - (self.height * 2)
        self.rect = Rect(self.x, self.y, self.width, self.height)

# GameBall class
class GameBall:
    def __init__(self, x, y):
        self.ball_rad = 10
        self.x = x
        self.y = y
        self.rect = Rect(self.x, self.y, self.ball_rad * 2, self.ball_rad * 2)
        self.speed_x = 4
        self.speed_y = -4
        self.speed_max = 5
        self.game_over = 0

    def move(self):
        # Collision with walls (left and right)
        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed_x *= -1
        # Collision with top
        if self.rect.top < 0:
            self.speed_y *= -1
        # Collision with bottom (game over)
        if self.rect.bottom > screen_height:
            self.game_over = -1

        # Collision with paddles
        if self.rect.colliderect(paddle.rect):
            if abs(self.rect.bottom - paddle.rect.top) < 5 and self.speed_y > 0:
                self.speed_y *= -1

        # Collision with blocks (check each block in the wall)
        for row in wall.blocks:
            for block in row:
                if self.rect.colliderect(block[0]):
                    # Destroy block by reducing its strength
                    self.speed_y *= -1
                    block[1] -= 1
                    if block[1] <= 0:
                        row.remove(block)

        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        return self.game_over

    def draw(self):
        pygame.draw.circle(screen, paddle_green, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad), self.ball_rad)
        pygame.draw.circle(screen, paddle_outline, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad), self.ball_rad, 3)

    def reset(self, x, y):
        self.rect.x = x - self.ball_rad
        self.rect.y = y
        self.speed_x = 4
        self.speed_y = -4
        self.game_over = 0

# Game Objects

# Create a wall
wall = Wall()
wall.create_wall()

# Create Paddle
paddle = Paddle(paddle_green)

# Create ball
ball = GameBall(paddle.x + (paddle.width // 2), paddle.y - paddle.height)

# Main Game Loop
run = True
last_timestamp = 0

while run:
    clock.tick(fps)
    screen.fill(bg)

    # Camera Processing
    _, frame = cam.read()
    if frame is not None:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_objects = detector.detect_object_positions(frame)
        object_x = detected_objects[0] if detected_objects else None

        # Paddle Position
        paddle.move(object_x)

        # Anotation
        now_timestamp = time.time()
        frame_rate = 1 / (now_timestamp - last_timestamp)
        last_timestamp = now_timestamp

        annotated_frame = detector.annotate_frame(frame, frame_rate)
        cv2.imshow("Detecção de Objetos", annotated_frame)

    # Draw Objects
    wall.draw_wall()
    paddle.draw()
    ball.draw()

    # Ball
    if live_ball:
        game_over = ball.move()
        if game_over:
            live_ball = False
    else:
        draw_text("CLICAR PARA COMEÇAR", font, text_col, 140, screen_height // 2 + 80)
        ball.reset(paddle.x + (paddle.width // 2), paddle.y - paddle.height)

    for event in pygame.event.get():
        if event.type == QUIT:
            run = False
        if event.type == MOUSEBUTTONDOWN and not live_ball:
            live_ball = True

    pygame.display.update()

# Free Resources
cam.release()
cv2.destroyAllWindows()
pygame.quit()