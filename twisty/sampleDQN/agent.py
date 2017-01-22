# -*- coding: utf-8 -*-
# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.

import tensorflow as tf
import numpy as np
import time

from twistyRL import poketCube
from game import Game
from model import DQN

tf.app.flags.DEFINE_boolean( "train", False, "학습모드. 게임을 화면에 보여주지 않습니다." )
FLAGS = tf.app.flags.FLAGS

n_action = 3
size = 144
# 총 진행할 게임 횟수
episode = 5000
batch = 1000
# 한게임당 큐브 회전 횟수 제한
max_play = 100
# 게임 진행 횟수
play_count = 0


def main( _ ) :
    game = Game( max_play )
    n_action = len( game.set )
    state = game.get_state( )
    brain = DQN( n_action, size, state )

    while 1 :
        game.reset( )
        gameover = FLAGS.train
        while not gameover :
            # DQN 모델을 이용해 실행할 액션을 결정합니다.
            action = brain.get_action( FLAGS.train )
            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            reward, gameover = game.proceed( np.argmax( action ) )
            # 점수를 받을 경우 출력
            if reward >= 8 :
                print( game.total_game, '번 게임의 ', game.cube.count, '번째 회전에서 ', reward, '점 획득!!' )
            # 위에서 결정한 액션에 따른 현재 상태를 가져옵니다.
            # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
            state = game.get_state( )

            # DQN 으로 학습을 진행합니다.
            brain.step( state, action, reward, gameover )
        if game.cube.done:
            print(game.total_game,'번째에서',game.cube.count,'회전만으로 큐브 완성!')
        if game.total_game % batch == 0:
            print( " 게임 진행횟수: {}, 평균보상: {}, , 게임 완료 여부 : {},큐브 회전 횟수: {}".format(
                    game.total_game, game.total_reward / game.total_game, game.cube.done,
                    game.cube.count ) )


if __name__ == '__main__' :
    tf.app.run( )
