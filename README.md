# CatchGame - DQN by PyTorch
![TensorFlowPlayCatch](https://github.com/eaden84/rl-catch-dqn/blob/master/img/TestCatch_Play.gif)

파이토치를 이용해 구현한 캐치게임 DQN Agent.   
랜덤하게 내려오는 과일을 바구니로 최대한 많이 받도록 한다.  
과일을 받지 못하고 바닥에 떨어뜨리면 게임 종료.  

* State : 과일_x좌표, 과일_y좌표, 바구니_x좌표
* Action : (바구니를) Move Left, Move Right, Hold
* Reward : 과일 받으면 +1, 못받으면 -1, 그 외 0
* 알고리즘 : DDQN 


## Download model
[Train_Catch.pth](https://github.com/eaden84/rl-catch-dqn/blob/master/weights/Train_Catch.pth)

## Dependencies

[PyTorch](https://pytorch.org/)  
[NumPy](https://numpy.org/)  
[IPython](https://ipython.org/)  
[Matplotlib](https://matplotlib.org/)  


## Setup
파이썬 3.11 버전대에서 설치 및 실행 확인 하였다.

```
pip install -r requirements.txt
```


## Play and Visualization
IPython 노트북 환경에서 테스트 코드가 실행되도록 구성하였다.  
TestCatch.ipynb 파일을 열어 노트북 환경을 활성화 한 후 첫번째 코드 블록을 실행시키면 된다.  
(단, 프로젝트 폴더내 파일을 모두 유지한 상태에서 실행이 가능하다.)  


## How to train
모델을 새로 학습시키고자 할 경우 아래의 코드를 이용해 학습을 실행할 수 있다.  
학습 완료 후 ./weights 경로에 모델이 파일로 저장된다.
```
python TrainCatch.py
```
