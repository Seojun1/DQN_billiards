import gym
from gym import spaces
import numpy as np
import pygame
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 색상 정의
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

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
        if self.x - self.radius < 50 or self.x + self.radius > 800 - 50:
            self.vx = -self.vx
        if self.y - self.radius < 50 or self.y + self.radius > 500 - 50:
            self.vy = -self.vy

        # 감속 (마찰력)
        self.vx *= 0.99
        self.vy *= 0.99

    def hit(self, power, angle):
        self.vx = power * math.cos(angle)
        self.vy = power * math.sin(angle)

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

class BilliardsEnv(gym.Env):
    def __init__(self):
        super(BilliardsEnv, self).__init__()
        self.width = 800
        self.height = 500
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('4구 당구 게임')

        # 배경 이미지 로드
        self.background = pygame.image.load('assets/background.png')
        self.background = pygame.transform.scale(self.background, (self.width, self.height))

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([20, 2*math.pi]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=max(self.width, self.height), shape=(16,), dtype=np.float32)

        self.reset()
        self.clock = pygame.time.Clock()

    def reset(self):
        self.cue_ball = Ball(self.width // 2, self.height // 4, WHITE)
        self.red_ball1 = Ball(self.width // 3, self.height // 2, RED)
        self.red_ball2 = Ball(2 * self.width // 3, self.height // 2, RED)
        self.yellow_ball = Ball(self.width // 2, 3 * self.height // 4, YELLOW)
        self.balls = [self.cue_ball, self.red_ball1, self.red_ball2, self.yellow_ball]
        
        self.score = 0
        self.shooting = False
        self.hit_red1 = False
        self.hit_red2 = False
        self.hit_yellow = False
        
        state = self._get_state()
        return state

    def step(self, action):
        power, angle = action
        self.cue_ball.hit(power, angle)
        self.shooting = True

        while self.shooting:
            for ball in self.balls:
                ball.update()

            for i in range(len(self.balls)):
                for j in range(i + 1, len(self.balls)):
                    if check_collision(self.balls[i], self.balls[j]):
                        if self.balls[i] == self.cue_ball or self.balls[j] == self.cue_ball:
                            if self.balls[i].color == RED or self.balls[j].color == RED:
                                if self.balls[i] == self.red_ball1 or self.balls[j] == self.red_ball1:
                                    self.hit_red1 = True
                                if self.balls[i] == self.red_ball2 or self.balls[j] == self.red_ball2:
                                    self.hit_red2 = True
                            elif self.balls[i].color == YELLOW or self.balls[j].color == YELLOW:
                                self.hit_yellow = True

            self.shooting = not all(abs(ball.vx) < 0.1 and abs(ball.vy) < 0.1 for ball in self.balls)

        if self.hit_red1 and self.hit_red2 and not self.hit_yellow:
            self.score += 1
            reward = 1
        elif self.hit_yellow:
            self.score -= 1
            reward = -1
        else:
            reward = 0

        done = False
        state = self._get_state()
        return state, reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            self.screen.blit(self.background, (0, 0))
            for ball in self.balls:
                ball.draw(self.screen)

            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

    def _get_state(self):
        state = []
        for ball in self.balls:
            state.extend([ball.x, ball.y, ball.vx, ball.vy])
        return np.array(state)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.model_target = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(0, 1, size=self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return act_values.detach().numpy()[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model_target(next_state)[0])
            target_f = self.model(state)
            target_f[0][np.argmax(action)] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

env = BilliardsEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

for e in range(1000):
    state = env.reset()
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        env.render()  # 시각화 추가
    if e % 10 == 0:
        print(f"Episode {e}/{1000}, Score: {env.score}")

agent.save('assets/model.pth')
env.close()
