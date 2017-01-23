from twistyRL import poketCube
import numpy as np
import random


class Game:
    def __init__(self,scram_size=25):

        self.cube = poketCube( )
        self.set = self.cube.set
        self.cube.getcube()
        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        # 스크램블 길이
        self.scram_size = scram_size
        # 최대 회전 횟수는 스크램블 사이즈의 두배로 고
        self.max_play = self.scram_size*2

    def onehot_state(self,state):
        # 큐브 상태를 one-hot상태로 변경함
        size = self.cube.size
        onehot = None
        oneline_stat = np.reshape(state,size*size*6)

        for i in range(1,7):
            # 전체 면에서 숫자 i인 면
            numstate = oneline_stat == i
            numstate = np.array(numstate,dtype=np.int)
            if i == 1:
                onehot = numstate
            else:
                onehot = np.append(onehot,numstate)
        return onehot.reshape((-1,144))

    def get_state(self):
        """
        큐브의 상태를 one-hot으로 바꿔준다
        """
        return self.onehot_state(self.cube.faces)


    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        self.cube.reset()
        self.cube.scramble(len=self.scram_size,count=1,checkface=2)
        self.total_game += 1
        self.current_reward = 0



    def proceed(self, action):
        # action: 0: 좌, 1: 유지, 2: 우
        # action - 1 을 하여, 좌표를 액션이 0 일 경우 -1 만큼, 2 일 경우 1 만큼 옮깁니다.
        act = self.cube.set[action]
        gameover,reward,count,_ =self.cube.action(act)
        if count == self.max_play:
            # 100회가 넘어가면 강제로 게임 종료
            gameover = True

        self.current_reward += reward
        if gameover:
            self.total_reward += self.current_reward

        return reward, gameover