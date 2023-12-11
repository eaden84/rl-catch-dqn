# libraries
import collections
import random
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)    # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n): # rand 하게 미니배치 선정
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch: # transition = s > a > r > s'
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(3, 128)        # input(과일_x, 과일_y, 바구니_x) > layer-1
        self.fc2 = nn.Linear(128, 128)      # layer-1 > layer-2
        self.fc3 = nn.Linear(128, 3)        # layer-2 > output(hold, left, right)

    def forward(self, x):                   # x 라는 input 을 NN 에 넣는 것, x 는 100개짜리 벡터
        x = F.relu(self.fc1(x))             # activation_func = relu
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                     # 마지막 layer 에는 activation_func 없음, 출력 3개 - hold, left, right
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs) # s 를 NN 에 넣었을때 output Q(s,hold), Q(s,left), Q(s,right) 3개
        coin = random.random()
        if coin < epsilon:                  # epsilon-greedy
            return random.randint(0,2) # hold, left, right 랜덤선택
        else :
            return out.argmax().item() # hold, left, right 중 Q 높은거


def train(q, q_target, memory, optimizer):
    for i in range(10): # 왜 10번? 꼭 정해진건 아님, 경험적인 설정(에피소드 1번에 10번 학습, 학습 더 잘되게)
        s,a,r,s_prime,done_mask = memory.sample(batch_size) # mini-batch

        q_out = q(s) # DQN 의 forward() 호출됨 > (32,2)_Q값 2개의 32개
        q_a = q_out.gather(1,a) # 실제로 행한 a 에 대한 q값 > Q세타(s,a) > 예측Q값

        # DQN
        #max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # maxQ세타(s', a') > 목표Q값

        # Double DQN
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1) # Q_NN 에서의 s' 에 대한 max_q 에 해당되는 a'
        max_q_prime = q_target(s_prime).gather(1, argmax_Q) # 타겟_NN 에서 Q(s', 위_a') 를 사용

        # 목표값
        # done_mask - terminal 여부에 따라 값을 더하거나, 안하거나 (p.15)
        target = r + gamma * max_q_prime * done_mask # 목표값

        # MSE Loss
        loss = F.mse_loss(q_a, target)

        # Smooth L1 Loss
        #loss = F.smooth_l1_loss(q_a, target)

        # loss 구한 것을 백프로파게이션해서 파라미터 업데이트
        optimizer.zero_grad() # gradient 0으로 초기화
        loss.backward() # Backpropagation, gradient 구함
        optimizer.step() # NN 파라미터 업데이트 = w 업데이트?

#----------------------------
# 환경 클래스
#----------------------------
class CatchEnvironment():

  # 초기화
  def __init__(self, gridSize):
    self.gridSize = gridSize
    self.nbStates = self.gridSize * self.gridSize
    self.state = np.empty(3, dtype = np.uint8)

  # 화면정보 리턴
  def observe(self):
    canvas = self.drawState()
    canvas = np.reshape(canvas, (-1, self.nbStates))
    return canvas

  #블럭과 바를 표시하여 화면정보 리턴
  def drawState(self):
    canvas = np.zeros((self.gridSize, self.gridSize))
    # 과일 표시
    canvas[self.state[0]-1, self.state[1]-1] = 1
    # 바구니 표시
    canvas[self.gridSize-1, self.state[2] -1 - 1] = 1
    canvas[self.gridSize-1, self.state[2] -1] = 1
    canvas[self.gridSize-1, self.state[2] -1 + 1] = 1
    return canvas

  # 과일과 바구니 위치 초기화
  def reset(self):
    initialFruitColumn = random.randrange(1, self.gridSize + 1)
    initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
    self.state = np.array([1, initialFruitColumn, initialBucketPosition])
    return self.state

  # 상태 리턴
  def getState(self):
    stateInfo = self.state
    fruit_row = stateInfo[0]
    fruit_col = stateInfo[1]
    basket = stateInfo[2]
    return fruit_row, fruit_col, basket

  # 보상값 리턴
  def getReward(self):
    fruitRow, fruitColumn, basket = self.getState()
    if (fruitRow == self.gridSize - 1):  # If the fruit has reached the bottom.
      if (abs(fruitColumn - basket) <= 1): # Check if the basket caught the fruit.
        return 1
      else:
        return -1
    else:
      return 0

  # 게임오버 검사
  def isGameOver(self):
    if (self.state[0] == self.gridSize - 1):
      return True
    else:
      return False

  # 상태 업데이트
  def updateState(self, action):
    if (action == 0):
      action = -1  # 왼쪽 이동
    elif (action == 1):
      action = 0  # 대기
    else:
      action = 1  # 오른쪽 이동
    fruitRow, fruitColumn, basket = self.getState()
    newBasket = min(max(2, basket + action), self.gridSize - 1)  # 바구니 위치 변경
    fruitRow = fruitRow + 1  # 과일을 아래로 이동
    self.state = np.array([fruitRow, fruitColumn, newBasket])

  # 행동 수행 (0->왼쪽, 1->대기, 2->오른쪽)
  def act(self, action):
    self.updateState(action)
    reward = self.getReward()
    gameOver = self.isGameOver()
    return self.state, reward, gameOver


def main():
    env = CatchEnvironment(gridSize=10) # 환경 생성 
    q = Qnet() # Q_네트워크 
    q_target = Qnet() # 타겟_네트워크
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    win_count = 0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # 옵티마이저, ADAM 사용, LR=하이퍼파라미터

    for n_epi in range(3000): # 에피소드 3,000 번 수행
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        # while문 - 하나의 에피소드 하면서 transition 을 ReplayBuffer 에 입력하는 구문
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            #s_prime, r, terminated, truncated, info = env.step(a) # a 실행
            s_prime, r, gameOver = env.act(a) # a 실행
            done = gameOver
            done_mask = 0.0 if done else 1.0
            #memory.put((s,a,r/100.0,s_prime, done_mask)) # r 은 학습의 안전성을 위해 값을 줄임, 보통 리턴의 합을 10 미만으로
            memory.put((s,a,r,s_prime, done_mask)) # r 은 학습의 안전성을 위해 값을 줄임, 보통 리턴의 합을 10 미만으로
            s = s_prime

            # 승리 횟수 카운트
            if (r == 1):
               win_count = win_count + 1

            if done:
                break

        # 에피소드 끝난 후 NN 업데이트
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        # 일정 interval 마다 q 를 q_target 에 복사
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict()) # 그대로 복사 = 하드 업데이트 (업데이트 하는 방식이 여러가지 있음)
            print("n_episode :{}, win_count : {}, win_ratio : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            #n_epi, score/print_interval, memory.size(), epsilon*100))
                                                            n_epi, win_count, win_count/print_interval*100.0, memory.size(), epsilon*100))
            win_count = 0.0

    # 모델의 상태 저장
    torch.save(q_target.state_dict(), './weights/Train_Catch.pth')

if __name__ == '__main__':
    main()
