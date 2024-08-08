import pygame
import math

# 초기화
pygame.init()

# 색상 정의
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# 화면 크기 설정
WIDTH, HEIGHT = 800, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('4구 당구 게임')

# 배경 이미지 로드
background = pygame.image.load('assets/background.png')
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# 공 클래스 정의
class Ball:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = 10
        self.vx = 0
        self.vy = 0

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # 벽 충돌
        if self.x - self.radius < 50 or self.x + self.radius > WIDTH - 50:
            self.vx = -self.vx
        if self.y - self.radius < 50 or self.y + self.radius > HEIGHT - 50:
            self.vy = -self.vy

        # 감속 (마찰력)
        self.vx *= 0.99
        self.vy *= 0.99

    def hit(self, power, angle):
        self.vx = power * math.cos(angle)
        self.vy = power * math.sin(angle)

# 충돌 처리 함수
def check_collision(ball1, ball2):
    dx = ball1.x - ball2.x
    dy = ball1.y - ball2.y
    distance = math.sqrt(dx**2 + dy**2)

    if distance < ball1.radius + ball2.radius:
        overlap = 0.5 * (distance - ball1.radius - ball2.radius)

        ball1.x -= overlap * (ball1.x - ball2.x) / distance
        ball1.y -= overlap * (ball1.y - ball2.y) / distance
        ball2.x += overlap * (ball1.x - ball2.x) / distance
        ball2.y += overlap * (ball1.y - ball2.y) / distance

        angle = math.atan2(dy, dx)
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        # 회전 행렬 적용
        v1x = ball1.vx * cos_a + ball1.vy * sin_a
        v1y = ball1.vy * cos_a - ball1.vx * sin_a
        v2x = ball2.vx * cos_a + ball2.vy * sin_a
        v2y = ball2.vy * cos_a - ball2.vx * sin_a

        # 속도 교환
        ball1.vx = v2x * cos_a - v1y * sin_a
        ball1.vy = v1y * cos_a + v2x * sin_a
        ball2.vx = v1x * cos_a - v2y * sin_a
        ball2.vy = v2y * cos_a + v1x * sin_a

        return True
    return False

# 공 생성
cue_ball = Ball(WIDTH // 2, HEIGHT // 4, WHITE)
red_ball1 = Ball(WIDTH // 3, HEIGHT // 2, RED)
red_ball2 = Ball(2 * WIDTH // 3, HEIGHT // 2, RED)
yellow_ball = Ball(WIDTH // 2, 3 * HEIGHT // 4, YELLOW)
balls = [cue_ball, red_ball1, red_ball2, yellow_ball]

# 메인 루프
running = True
clock = pygame.time.Clock()
score = 0
shooting = False
hit_red1 = False
hit_red2 = False
hit_yellow = False

# 추가된 변수
mouse_down_time = 0
mouse_up_time = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not shooting:
                mouse_down_time = pygame.time.get_ticks()
        elif event.type == pygame.MOUSEBUTTONUP:
            if not shooting:
                mouse_up_time = pygame.time.get_ticks()
                mouse_x, mouse_y = pygame.mouse.get_pos()
                dx = mouse_x - cue_ball.x
                dy = mouse_y - cue_ball.y
                angle = math.atan2(dy, dx)
                power = min((mouse_up_time - mouse_down_time) / 100, 20)  # 힘 조절
                cue_ball.hit(power, angle)
                shooting = True
                hit_red1 = False
                hit_red2 = False
                hit_yellow = False

    # 배경 이미지 그리기
    screen.blit(background, (0, 0))

    # 공 업데이트 및 그리기
    for ball in balls:
        ball.update()
        ball.draw(screen)

    # 공 충돌 처리
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            if check_collision(balls[i], balls[j]):
                if balls[i] == cue_ball or balls[j] == cue_ball:
                    if balls[i].color == RED or balls[j].color == RED:
                        if balls[i] == red_ball1 or balls[j] == red_ball1:
                            hit_red1 = True
                        if balls[i] == red_ball2 or balls[j] == red_ball2:
                            hit_red2 = True
                    elif balls[i].color == YELLOW or balls[j].color == YELLOW:
                        hit_yellow = True

    # 공이 멈추면 점수 계산
    if shooting and all(abs(ball.vx) < 0.1 and abs(ball.vy) < 0.1 for ball in balls):
        shooting = False
        if hit_red1 and hit_red2 and not hit_yellow:
            score += 1
        elif hit_yellow:
            score -= 1

    # 점수 표시
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
