# -*- coding: utf-8 -*-
# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.

import tensorflow as tf
import numpy as np
import time

from twistyRL import poketCube
from game import Games
from model import cubeDQN
import time
from os import path
import os

tf.app.flags.DEFINE_boolean( "train", True, "학습모드. 게임을 화면에 보여주지 않습니다." )
FLAGS = tf.app.flags.FLAGS

# 배치 사이즈
batch = 200

# 한게임의 최대 회전 횟수
max_play = 50
# 테스트 배치 사이즈
test_batch_size = 100
# 스크램 길이 설정
scram_size = 5

# 실험 이름(logs 기록에 사용됨)
lab = 'lab5'

# 테스트 배치 기록
test_batch_record = {
    '학습정보' : { '스크램길이' : scram_size, '최대회전' : max_play, '배치사이즈' : batch, '테스트배치사이즈' : test_batch_size }
    }


def logging( log, file ) :
    folder = 'train_log'
    if not path.isdir( folder ) :
        os.makedirs( folder )
    train_day = time.strftime( '%Y-%m-%d-' )
    with open( path.join( folder, '{}{}'.format( train_day, file ) ), mode='a' ) as logs :
        logs.write( '\n{}'.format( log ) )


def test( scram_size, max_play, DQN, batch_size=100 ) :
    """
    학습 결과를 측정하기 위해 여러개의 시뮬레이션을 돌린뒤 평균을 반환한다.
    :param game: game 객체
    :param brain: DQN 객체
    :param batch_size: 한번 테스트에 실행할 시뮬레이션 갯수
    :return:
    """
    brain = DQN
    game = Games( scram_size, max_play )
    batch = batch_size
    train = False
    total_reward = [ ]
    total_count = [ ]
    total_done = [ ]

    for _ in range( batch_size ) :
        game.reset( )
        rewards = [ ]
        gameover = False
        act = brain.get_action( state=game.states, train=train )
        while not gameover :
            action = game.set[ act ]
            reward, gameover = game.proceed( action )
            act = brain.step( game.states, action, reward, gameover, train )
            rewards.append( reward )
        # 전체 보상 기록에 추가
        total_reward.append( sum( rewards ) )
        total_count.append( game.count )
        total_done.append( game.done )

    # 테스트 평균 보상
    avg_rewards = sum( total_reward ) / batch
    # 테스트 평균 회전 횟수
    avg_counts = sum( total_count ) / batch
    # 테스트 큐브 완성 확률
    per_done = total_count.count( True ) / batch * 100
    return avg_counts, avg_rewards, per_done


def main( _ ) :
    try :
        # logname = input( "로그 파일 명을 입력하세요!" )
        logname = lab
        game = Games( scram_size, max_play )
        brain = cubeDQN( game.set, cube_size=game.size, lab=lab )
        # 테스트 실행 횟수
        test_run_count = 0

        # 시간 측정
        start = time.time( )
        print( 'start training' )
        while 1 :
            game.reset( )
            # todo: 처음 게임을 시작할때는 getaction시 상태값을 보내게 만들고 이후엔 상태값 입력없이 진행하게 한다.
            train = True
            # 처음 진행할 경우 현재 상태를 이용하여 액션값을 가져온다
            acts = brain.get_action( state=game.states, train=train )
            # 게임 진행 여부
            end = False

            while not end :
                # DQN 모델을 이용해 실행할 액션을 결정합니다.
                action = game.set[ acts ]
                # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
                reward, gameover = game.proceed( action )

                # DQN 으로 학습을 진행한뒤 다음 액션값을 받아온다.
                acts = brain.step( game.states, acts, game.reward, train )
                # 게임이 종료되면 반복을 끝낸다.
                if gameover :
                    end = True

            if game.total_game % batch == 0 :
                # 각 배치 실행 시간 측정
                end = time.time( )
                runtime = end - start

                # 학습 결과 테스트
                test_run_count += 1
                test_count, test_reward, test_done = test( scram_size, max_play, brain, batch_size=test_batch_size )
                # 텐서보드에 테스트 결과 기록
                brain.reward_log( test_run_count * test_batch_size, test_reward, test_count, test_done )
                # 콘솔에 테스트 결과 출력
                result = { "평균회전횟수" : test_count, "평균보상" : test_reward, "완성확률" : test_done }
                # 테스트 결과 기록
                test_batch_record[ test_run_count ] = result

                batch_state = "==== 테스트 결과 ====\n" \
                              "게임 진행횟수: {}, 평균보상: {}, 큐브 완성 확률: {}\n" \
                              "평균 회전 횟수: {}, 소요시간: {}" \
                    .format( game.total_game, test_reward, test_done,
                             test_count, runtime )
                print( batch_state )
                start = time.time( )
    except Exception as e :
        raise e
    finally :
        # 프로그램 종료시 테스트 기록을 로깅 및 출력한다.
        logging( test_batch_record, logname )
        print( test_batch_record )


if __name__ == '__main__' :
    tf.app.run( )
