from twistyRL import poketCube
import numpy as np
import random

class poketCube100(poketCube):
    """
    무조건 게임이 100번 진행하도록 점수 부여 방식과 게임 완료 방식을 변경함
    """
    def check( self ) :
        """
        면 상태 체크 메소드, 큐브가 변경되는 시점마다 호출하여 완셩여부와 점수를 확인한다.
        :return:
        """
        # 큐브 완셩 여부 확인
        done = [ self.cube[ x ].done for x in self.cube ]
        if False in done :
            self.done = False
        else :
            self.done = True

        # 회전 횟수
        self.count = len( self.history )


        if self.count == 0:
            # 게임 시작 전일 경우 보상은 0으로 한다
            self.reward = 0
        elif self.done == True:
            # 큐브가 완성 된 경우 무조건 100 점 부여

        elif True in done:
            # 완성된 면이 존재할 경우 완성된 면의 갯수*면의 총점수/회전횟수 만큼 보상을 줘서
            # 많이 회전할수록 점수가 떨어지게 함 즉 단순 회전을 반복하여 고득점하는 기회를 없앰
            self.reward = pow(self.size,2)
        else:
            # 큐브가 미완성일경우 점수 없음
            self.reward = 0

class Games:
    """
    여러 큐브 게임을 생성하고 게임 진행을 관리 하는 객
    """
    def __init__(self,scram_size=25,num_game=5):

        self.cube = poketCube( )
        self.set = self.cube.set
        self.cube.getcube()
        self.total_reward = []
        self.current_reward = []
        self.total_game = 0
        # 한번에 플레이할 게임 겟수
        self.num_game = num_game
        # 스크램블 길이
        self.scram_size = scram_size
        # 최대 회전 횟수는 스크램블 사이즈의 두배로 고
        self.max_play = self.scram_size*2
        self.games = self.make_game()

    def make_game(self):
        """
        원하는 갯수만큼 게임을 생성하여 리스트에 담아 반환한다
        :return:
        """
        games = []
        for i in range(self.num_game):
            game = poketCube()
            game.scramble(self.cube.scram_size)
            games.append(game)
        return games



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
        # states = [self.onehot_state(x.faces) for x in self.games]
        states = [x.faces for x in self.games]
        return states


    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        for cube in self.games:
            cube.reset()
            cube.scramble(len=self.scram_size,count=1,checkface=2)
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