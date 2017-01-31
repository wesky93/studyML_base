# -*- coding: utf-8 -*-
# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.

import tensorflow as tf
import numpy as np
import time

from twistyRL import poketCube
from game import Game
from model import cudaDQN
import time
from os import path

tf.app.flags.DEFINE_boolean( "train", False, "학습모드. 게임을 화면에 보여주지 않습니다." )
FLAGS = tf.app.flags.FLAGS

n_action = 3
size = 144
# 총 진행할 게임 횟수
episode = 5000
batch = 1000
# 한게임당 큐브 회전 횟수 제한
max_play = 100
# 스크램 길이 설정
scram_size = 5
# 게임 진행 횟수
play_count = 0

# 완성한 횟수
done_count = 0
done_percent = 0

# 배치별 성공확률 기록
batch_track = [ ]

dropout = 1

def logging( log, file ) :
    train_day = time.strftime( '%Y-%m-%d-' )
    with open( path.join( 'train_log', '{}{}'.format( train_day, file ) ), mode='a' ) as logs :
        logs.write( '\n{}'.format( log ) )


def main( _ ) :
    logname = input( "로그 파일 명을 입력하세요!" )
    game = Game( scram_size )
    n_action = len( game.set )
    state = game.get_state( )
    brain = cudaDQN( game.cube.set, game.cube.size, dropout )
    start = time.time( )
    befor_total_reward = 0

    # 완성한 횟수
    befor_done_count = 0
    done_count = 0
    done_percent = 0

    # 완성 성공한 게임의 최대 회전 횟수
    max_rotae = 0

    while 1 :
        game.reset( )
        gameover = FLAGS.train
        while not gameover :
            # DQN 모델을 이용해 실행할 액션을 결정합니다.
            action = brain.get_action( FLAGS.train )
            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            reward, gameover = game.proceed( np.argmax( action ) )
            # 점수를 받을 경우 출력
            # if reward > 0 :
            #     print( game.total_game, '번 게임의 ', game.cube.count, '번째 회전에서 ', reward, '점 획득!!' )
            # 위에서 결정한 액션에 따른 현재 상태를 가져옵니다.
            # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
            state = game.get_state( )

            # DQN 으로 학습을 진행합니다.
            brain.step( state, action, reward, gameover )

        if game.cube.done :
            # text = '{}번째에서 {} 회전만으로 큐브 완성! 총 점수는 {}점!'.format( game.total_game, game.cube.count,
            #                                                 game.cube.point )
            done_count += 1
            max_rotae = game.cube.count if max_rotae < game.cube.count else max_rotae
            # logging( text, logname )
            # print( text )

        if game.total_game % batch == 0 :
            # 각 배치 실행 시간
            end = time.time( )
            runtime = end - start

            # 각 배치별 성공 활률
            batch_done_count = done_count - befor_done_count
            befor_done_count = done_count
            # 각 배치별 평균점수
            batch_total_reward = game.total_reward - befor_total_reward
            befor_total_reward = game.total_reward

            # 배치별 평균 보상
            batch_avg_reward = batch_total_reward / batch
            # 배치별 큐브 완성 확률
            batch_done_percent = batch_done_count / batch * 100

            # 전체 평균 보상
            Avg_Allreward = game.total_reward / game.total_game
            # 전체 큐브 완성 확률
            ALL_done_percent = done_count / game.total_game * 100

            # 배치별 완성 활률 기록
            batch_track.append( batch_done_percent )

            batch_state = "=================\n게임 진행횟수: {}, 전체평균보상: {}, 배치평균보상: {} \n완료 여부 : {},큐브 회전 횟수: {}, 소요시간: {},\n완성한 게임중 최대 회전 횟수: {}회, 전체 큐브 완성 확률 {}%, 배치별 큐브 완성 확률 {}%\n현재까지 확률 추이: {}" \
                .format( game.total_game, Avg_Allreward, batch_avg_reward, game.cube.done,
                         game.cube.count, runtime, max_rotae, ALL_done_percent, batch_done_percent, batch_track )
            print( batch_state )

            logging( batch_state, logname )
            start = time.time( )


if __name__ == '__main__' :
    tf.app.run( )
