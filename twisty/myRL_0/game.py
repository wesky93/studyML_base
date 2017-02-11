from twistyRL import poketCube
import numpy as np
import random


class poketCubeLimit( poketCube ) :
    """
    큐브가 완성되면 게임이 종료하는 것이 아닌 무조건 N번을 진행한뒤 게임을 종료하는 방식
    """

    # todo: self.set 변수명 변경
    def __init__( self ) :
        super( ).__init__( )
        # 큐브 크기
        self.size = 2
        # 큐브에 사용 가능한 명령어
        self.set = ('F', 'F`', 'R', 'R`', 'U', 'U`', 'B', 'B`', 'L', 'L`', 'D', 'D`', 'N')
        # 동적 변수 생성
        self.activeInit( )
        self.doneReward = 100

    @property
    def rotateCount( self ) :
        # 큐브가 완성되면 history 기록을 하지 않기에 단순히 history 카운팅만 하면 됨
        return self.count
        # 'N'을 제외한 회전 횟수를 계수한다
        # return [ False if x == 'N' else True for x in self.history ].count( True )

    @property
    def reward( self ) :
        # 보상 방식 오버라이딩

        # 게임 시작 전일 경우 보상은 0으로 한다
        if self.count == 0 :
            return 0
        # 큐브가 완성 되어 'N' 입력시 100 점 부여
        elif self.done == True and self.lastAct == 'N' :
            return self.doneReward
        # 큐브가 완성 되었으나 행동이 N이 아닐경우 -100점 부여(감점)
        elif self.done == True and self.lastAct != 'N' :
            return -self.doneReward
        # 큐브가 완성되기전에 행동이 'N'일 경우 -100점 부여(감점)
        elif self.done != True and self.lastAct == 'N' :
            return -self.doneReward
        elif True in self.facesDone :
            # 일부 완성된 면이 있을 경우 (완성된 면 갯수 * 완성된 면의 점수)만큼 점수를 부여한다.
            return self.doneCount * self.facePoint
        else :
            # 위의 사항에 해당 되지 않을 경우 -1점 부여
            return -1

    def rotate( self, action ) :
        # 아무런 행동을 안하는 명령어('N') 추가
        if action == 'N' :
            pass
        else :
            super( ).rotate( action )
    @property
    def done(self):
        if self.count==0:
            return False
        else:
            super().done

    def action( self, action ) :
        """
        입력받은 act를 수행한뒤 상태를 반환해준다
        :param act: 회전 방향 기호
        :return: (완료여부,점수,회전횟수,큐브화면)
        """
        # 큐브가 완성 될경우 아무런 행동을 하지 않는다.
        if self.count!=0 and self.done:
            pass
        else:
            self.rotate( action )
            # 회전 기록
            self.history.append( action )
            # todo:180되 회전 명령어대신 90도 명령어가 2번 표기 되도록 변경

        self.point += self.reward
        return (self.done, self.reward, self.count, self.faces)

class Games :
    """
    여러 큐브 게임을 생성하고 게임 진행을 관리 하는 객체
    """

    def __init__( self, scram_size=20, max_play=None ) :

        # 전체 점수
        self.total_reward = 0
        # 전체 플레이 횟수
        self.total_game = 0
        self.current_reward = 0
        # 스크램블 길이
        self.scram_size = scram_size
        # 최대 큐브 회전 횟수를 지정하지 않을 경우 자동으로 (스크램길이*2)만큼 최대 회전 횟수로 지정한다.
        self.max_play = max_play if max_play else self.scram_size * 2
        self.game = poketCubeLimit( )

    @property
    def set( self ) :
        # 사용가능한 명령어
        return self.game.set

    @property
    def size( self ) :
        # 큐브 크기
        return self.game.size

    @property
    def states( self ) :
        # 큐브의 상태
        # 큐브의 각 면을 2*2크기로 바꿔 리스트에 담아둠
        get_state = [ np.reshape( face, (2, 2) ) for face in self.game.faces ]
        state = np.full( (1, 6, 8), -1. )

        # 2번면 입력
        state[ 0, 0 :2, 2 :4 ] = get_state[ 1 ]
        # 3번면 입력
        state[ 0, 2 :4, 0 :2 ] = get_state[ 2 ]
        # 1번면 입력
        state[ 0, 2 :4, 2 :4 ] = get_state[ 0 ]
        # 4번면 입력
        state[ 0, 2 :4, 4 :6 ] = get_state[ 3 ]
        # 6번면 입력
        state[ 0, 2 :4, 6 : ] = get_state[ 5 ]
        # 5번면 입력
        state[ 0, 4 :, 2 :4 ] = get_state[ 4 ]
        return state

    @property
    def reward( self ) :
        # 큐브의 점수
        return self.game.reward

    @property
    def done( self ) :
        # 큐브 완성여부
        return self.game.done

    @property
    def count( self ) :
        return self.game.rotateCount

    @property
    def history( self ) :
        return self.game.history

    def reset( self ) :
        """큐브 게임을 초기화 한다"""
        self.game.reset( )
        self.game.scramble( len=self.scram_size, count=1, checkface=2 )
        self.current_reward = 0
        self.total_game += 1

    def proceed( self, action ) :
        """
        회전 명령어 인덱스 값을 입력받아 게임을 진행한다.
        :param action:
        :return:
        """
        gameover = False
        act = action
        _, reward, count, _ = self.game.action( act )
        # 현재 게임의 누적 보상
        self.current_reward += reward
        if count == self.max_play :
            # 진행횟수가 최대 회전횟수(max_play)를 넘어가면 강제로 게임 종료
            gameover = True

        if gameover :
            self.total_reward += self.current_reward
            self.current_reward = 0

        return reward, gameover

    @property
    def avgResult( self ) :
        # 게임 보상 평균 값
        return self.total_rewards / self.total_game
