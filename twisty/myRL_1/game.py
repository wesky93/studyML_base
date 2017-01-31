from twistyRL import poketCube
import numpy as np
import random

class poketCubeLimit(poketCube):
    """
    큐브가 완성되면 게임이 종료하는 것이 아닌 무조건 N번을 진행한뒤 게임을 종료하는 방식
    """
    def __init__(self):
        super(poketCube,self).__init__()
        # 아무런 행동을 하지 않는 명령어를 추가('N')
        self.set = self.set.append('N')
        # 각면이 완성될시 부여할 점수
        self.facePoint = pow(self.size,2)
        # 큐브 완성시 부여할 점수
        self.doneReward = 100

    @property
    def rotateCount(self):
        # 'N'을 제외한 회전 횟수를 계수한다
        return [False if x == 'N' else True for x in self.history].count(True)

    @property
    def reward( self ) :
        # 보상 방식 오버라이딩

        # 게임 시작 전일 경우 보상은 0으로 한다
        if self.count == 0:
            return 0
        # 큐브가 완성 되어 'N' 입력시 100 점 부여
        elif self.done == True and self.lastAct == 'N':
            return self.doneReward
        # 큐브가 완성 되었으나 행동이 N이 아닐경우 -100점 부여(감점)
        elif self.done == True and self.lastAct != 'N':
            return  -self.doneReward
        # 큐브가 완성되기전에 행동이 'N'일 경우 -100점 부여(감점)
        elif self.done != True and self.lastAct == 'N':
            return -self.doneReward
        elif self.facesDone in True:
            # 일부 완성된 면이 있을 경우 (완성된 면 갯수 * 완성된 면의 점수)만큼 점수를 부여한다.
            return self.doneCount*self.facePoint
        else:
            # 위의 사항에 해당 되지 않을 경우 0점 부여
            return 0

    def rotate( self, action ):
        # 아무런 행동을 안하는 명령어('N') 추가
        if action == 'N':
            pass
        else:
            super().rotate(action)


class Games:
    """
    여러 큐브 게임을 생성하고 게임 진행을 관리 하는 객체
    """
    def __init__(self,scram_size=25,num_game=5):


        # 전체 점수
        self.total_rewards = 0
        # 전체 플레이 횟수
        self.total_game = 0
        # 한번에 플레이할 게임 겟수
        self.num_game = num_game
        self.current_rewards = [ 0 for x in range(self.num_game) ]
        # 스크램블 길이
        self.scram_size = scram_size
        # 최대 회전 횟수는 스크램블 사이즈의 두배로 고
        self.max_play = self.scram_size*2
        self.games = self.make_game()
        self.set = self.games[0].set

    def make_game(self):
        """
        원하는 갯수만큼 게임을 생성하여 리스트에 담아 반환한다
        :return:
        """
        games = []
        for i in range(self.num_game):
            game = poketCube()
            game.scramble(self.scram_size)
            games.append(game)
        self.current_rewards = [ 0 for x in range(self.num_game) ]
        return games

    @property
    def states(self):
        # 각 큐브의 상태
        return np.array([x.faces for x in self.games])

    @property
    def rewards(self):
        # 각 큐브의 점수
        result = np.array( [x.reward for x in self.games])
        return np.reshape(result,[-1,1])

    @property
    def done(self):
        # 각 큐브의 완성 여부
        result = np.array( [ x.done for x in self.games ] )
        return result

    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        for x in range(self.num_game):
            cube = self.games[x]
            cube.reset()
            cube.scramble(len=self.scram_size,count=1,checkface=2)
            self.current_rewards[x] += self.rewards[x]
        self.total_game += 1

    def act(self,actions):
        """
        회전 명령어 인덱스 값을 회전 명령어로 바꿔준다.
        :param actions:
        :return:
        """
        return [self.set[x] for x in actions]

    def proceed(self, actions):
        """
        회전 명령어 인덱스 값을 입력받아 게임을 진행한다.
        :param action:
        :return:
        """

        acts = act(actions)
        for x in self.num_game:
            cube = self.games[x]
            action = acts[x]
            _,reward,_,_ =cube.action(action)
            # 현재 게임의 누적 보상
            self.current_rewards[x] += reward

        gameover,reward,count,_ =self.cube.action(act)
        if count == self.max_play:
            # 진행횟수가 최대 회전횟수(max_play)를 넘어가면 강제로 게임 종료
            gameover = True

        if gameover:
            self.total_reward += sum(self.current_reward)

        return self.rewards, gameover

    @property
    def avgResult(self):
        # 게임 보상 평균 값
        return self.total_rewards/(self.num_game*self.total_game)

