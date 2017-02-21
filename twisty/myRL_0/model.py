import tensorflow as tf
import numpy as np
import random
from os import path, mkdir
from collections import deque


class cubeDQN :
    def __init__( self, set, cube_size=2, lab='test', load_file=None, layer1=36, layer2=72, layer3=144, fc=1024 ) :
        """
        트위스티 큐브 Deep Q Netwarks 클래스
        :param set: 명령어 모음집
        :param cube_size: 큐브 크기
        :param dropout:
        """
        # 학습정보를 저장할 폴더가 없을시 생성
        for dir in [ 'logs', 'model' ] :
            if not path.isdir( dir ) :
                mkdir( dir )

        # 실험이름
        self.lab = lab
        # 초기화에 사용할 이전 학습자료
        self.load = load_file

        # 큐브 크기 - 포켓큐브 크기가 작아서 cnn이 제대로 안되면 루빅스큐브로 변경하기
        self.cube_size = cube_size
        # 큐브 명령어 갯수 - 기본 12 + 무반응 1개 추가
        self.set = set
        self.count_set = len( set )

        # 학습 횟수
        self.train_step = 0
        # 이전 플레이 기록
        self.replay_memory = deque( )
        # 이전 플레이 기록 갯수
        self.replay_memory_size = 50000
        # 한번에 학습할 배치 사이즈
        self.replay_batch_size = 50

        # 1차 신경망 뉴런수
        # todo: 뉴런 갯수 조절 필요
        self.num_filters1 = layer1
        # 1차 신경망 필터 사이즈
        self.size_filter1 = 4

        # 2차 신경망 뉴런수
        self.num_filters2 = layer2
        # 2차 신경망 필터 사이즈
        self.size_filter2 = 4

        self.num_filters3 = layer3
        self.size_filter3 = 2
        # 보상 감가상액 비율
        self.GAMMA = 0.99

        # 최종단계 뉴런 갯수
        self.full_neuron = fc

        # getaction 램덤 확률
        self.get_random = 1.0
        self.minimum_random = 0.001
        # 최소 학습량 - 랜덤확률을 최소 학습량 만큼 유지한다.
        self.minimum_step = 10000

        # 큐브 상태 shape
        self.state_shapeX = self.cube_size * 3
        self.state_shapeY = self.cube_size * 4
        # 큐브 상태입력
        self.state_x = tf.placeholder( tf.float32, [ None, self.state_shapeX, self.state_shapeY ], name='state' )
        self.input_x = tf.reshape( self.state_x, [ -1, self.state_shapeX, self.state_shapeY, 1 ], name='input_x' )
        # 회전 방향 입력
        self.action = tf.placeholder( tf.float32, [ None, self.count_set ], name="action" )
        self.reward_y = tf.placeholder( tf.float32, [ None ], name="reward_y" )

        # 모델 생성
        self.Q_value, self.train_opti = self.build_model( )
        # 세션생성
        # 불러올 이전 학습 자료가 없을 경우 모든 변수를 초기화 한다.
        if type( self.load ) == type( None ) :
            self.session = self.init_session( )
        # 불러올 자료가 있으면 해당 자료로 변수를 복구한다.
        else :
            self.session = self.load_model( )

        self.writer = tf.summary.FileWriter( path.join( 'logs', self.lab, ), self.session.graph )

        self.summary = tf.summary.merge_all( )

    def init_session( self ) :
        session = tf.InteractiveSession( )
        session.run( tf.global_variables_initializer( ) )
        return session

    def save_model( self ) :
        # 현재까지 학습한것을 저장한다
        save_path = path.join( 'model', '{}.ckpt'.format( self.lab ) )
        saver = tf.train.Saver( )
        # model에 랩이름을 파일명으로 저장한다.
        saver.save( self.session, save_path )
        print( "현재까지 학습한 것을 {}에 저장 했습니다".format( save_path ) )

    def load_model( self ) :
        # 이전 학습 내용이 존재할 경우 모든 변수를 초기화 하지 않고 이전 학습 내용을 불러 냅니다.
        # 불러올 파일경로
        load_path = path.join( 'model', '{}.ckpt'.format( self.load ) )

        session = tf.InteractiveSession( )
        saver = tf.train.Saver( )
        assert type( self.load ) != type( None )
        saver.restore( session, load_path )
        print( '{}의 변수로 초기화 했습니다'.format( load_path ) )
        return session

    def build_model( self ) :
        """
        학습 모델 구성
        :return: Q()텐서,train텐서
        """
        with tf.name_scope( 'input_layer' ) :
            # W_conv1 -> [ 필터크기,필터크기, 차원수,필터갯수 ]
            W_conv1 = tf.Variable(
                    tf.truncated_normal( [ self.size_filter1, self.size_filter1, 1, self.num_filters1 ] ) )
            # 1차 신경망 적용
            h_conv1 = tf.nn.conv2d( self.input_x, W_conv1, strides=[ 1, 1, 1, 1 ], padding='SAME', name='L_Input' )
            h_conv1_cutoff = tf.nn.relu( h_conv1 )
            # print( h_conv1_cutoff )
            # 6*8 -> 6*8 유지
            h_conv1_shape = (self.state_shapeX, self.state_shapeY)

        with tf.name_scope( 'hidden1_layer' ) :
            # 2차 신경망 적용
            W_conv2 = tf.Variable(
                    tf.truncated_normal(
                            [ self.size_filter2, self.size_filter2, self.num_filters1, self.num_filters2 ] ) )
            h_conv2 = tf.nn.conv2d( h_conv1_cutoff, W_conv2, strides=[ 1, 1, 1, 1 ], padding='VALID', name='L_hidden1' )
            h_conv2_cutoff = tf.nn.relu( h_conv2 )
            # 6*8 -> 3*5 로 바뀜
            h_conv2_shape = (h_conv1_shape[ 0 ] - self.size_filter2 + 1, h_conv1_shape[ 1 ] - self.size_filter2 + 1)
            # print( h_conv2_cutoff )

        with tf.name_scope( 'hidden2_layer' ) :
            # 3차 신경망 적용
            W_conv3 = tf.Variable(
                    tf.truncated_normal(
                            [ self.size_filter3, self.size_filter3, self.num_filters2, self.num_filters3 ] ) )
            h_conv3 = tf.nn.conv2d( h_conv2_cutoff, W_conv3, strides=[ 1, 1, 1, 1 ], padding='VALID', name='L_hidden2' )
            h_conv3_cutoff = tf.nn.relu( h_conv3 )
            # print(h_conv3_cutoff)
            # 3*5 -> 2*4
            h_conv3_shape = (h_conv2_shape[ 0 ] - self.size_filter3 + 1, h_conv2_shape[ 1 ] - self.size_filter3 + 1)

        with tf.name_scope( 'hidden3_layer' ) :
            # 풀 커넥티드 레이러을 위한 입력값 갯수(n*n*num_filters2)
            full_unit1 = h_conv3_shape[ 0 ] * h_conv3_shape[ 1 ] * self.num_filters3

            # n*n 행렬 'num_filters2'개를 1차원 행렬로 만든다
            h_conv3_flat = tf.reshape( h_conv3_cutoff, [ -1, full_unit1 ] )

            # 풀 커넥티드 레이어
            w2 = tf.Variable( tf.truncated_normal( [ full_unit1, self.full_neuron ] ) )
            fully_conect = tf.nn.relu( tf.matmul( h_conv3_flat, w2 ) )

        # todo: 이전의 FC레이어가 실상 히든레이어이고 Q-net 레이어가 FC레이어로 유추되나 정확한 확인이 필요
        with tf.name_scope( 'FC_layer' ) :
            # Q_value
            w0 = tf.Variable( tf.zeros( [ self.full_neuron, self.count_set ] ) )
            b0 = tf.Variable( tf.zeros( [ self.count_set ] ) )
            Q_value = tf.matmul( fully_conect, w0 ) + b0

        with tf.name_scope( 'Q_action' ) :
            # DQN 손실 함수
            Q_action = tf.reduce_sum( tf.mul( Q_value, self.action ), axis=1 )
        with tf.name_scope( 'cost_function' ) :
            cost = tf.reduce_sum( tf.square( self.reward_y - Q_action ) )
        tf.summary.scalar( 'cost', cost )
        with tf.name_scope( 'Training' ) :
            train_op = tf.train.AdamOptimizer( 1e-6 ).minimize( cost )

        return Q_value, train_op

    def train( self ) :
        """
        저장된 자료중 일부를 추출하여 학습합니다.
        :return:
        """

        self.train_step += 1
        train_batch = random.sample( self.replay_memory, self.replay_batch_size )

        state = [ data[ 0 ] for data in train_batch ]
        action = [ data[ 1 ] for data in train_batch ]
        reward = [ data[ 2 ] for data in train_batch ]
        next_state = [ data[ 3 ] for data in train_batch ]
        gameover = [ data[ 4 ] for data in train_batch ]
        # 액션값을 one-hot으로 바꾸기
        act = [ [ 1 if action[ x ] == y else 0 for y in range( self.count_set ) ] for x in
                range( self.replay_batch_size ) ]

        # todo: DQN 구현하기 - 메모리에 학습할것을 보관한뒤 랜덤으로 자료를 추출하여 학습하기
        # 학습
        Q_value = self.Q_value.eval( feed_dict={ self.state_x : next_state } )
        # y값을 구한다.
        reward_y = reward + self.GAMMA * np.max( Q_value, axis=1 )
        # y값중 게임오버된 상태가 있을 경우 y값을 현재 보상값으로 바꾼다.
        if True in gameover :
            # reward_y의 값중 게임오버된 상황의 값은 현재 보상으로 바꿔준다.
            reward_y = [ reward[ x ] if gameover[ x ] else reward_y[ x ] for x in range( self.replay_batch_size ) ]

        self.train_opti.run(
                feed_dict={ self.reward_y : reward_y, self.state_x : state, self.action : act } )

        # 텐서보드에 기록
        if self.train_step % 1000 == 0 :
            summary = self.summary.eval(
                    feed_dict={ self.reward_y : reward_y, self.state_x : state, self.action : act } )
            self.writer.add_summary( summary, self.train_step )

    # 매 스탭마다 결과를 deque에 저장해둠
    def step( self, new_state, action, reward, gameover, train=True ) :
        """
        게임의 진행 상황을 저장하고 저장된자료가 일정 크기를 넘을 경우 학습을 시작한다.
        자료는 deque 자료형에 추가되며 일정크기가 넘을경우 앞에 자료를 삭제한다.
        게임 진행상황은 매번 (이전 상태, 액션, 보상, 다음 상태, 게임 완료여부)의 형태로 저장이 된다.
        :param state:
        :param action:
        :param reward:
        :return:
        """

        # 게임 진행 상황 기록
        self.next_state = new_state
        self.input_action = action
        self.reward = reward
        self.gameover = gameover

        # 게임 진행 상황을 기록한다.
        self.replay_memory.append( (self.before_state, self.input_action, self.reward, self.next_state, self.gameover) )

        # 저장된 플레이 기록이 일정 크기를 넘을 경우 이전 자료를 삭제한다.
        if len( self.replay_memory ) > self.replay_memory_size :
            self.replay_memory.popleft( )

        # 저장된 플레이 기록이 학습하기에 충분한 양일 경우 학습을 한다.
        if train and len( self.replay_memory ) == self.replay_memory_size :
            self.train( )

        return self.get_action( self.next_state, train=train )

    def reward_log( self, playtime, avg_reward, avg_count, per_done, avg_done_count, avg_done_reward ) :
        """
        검증 자료의 평균 보상을 텐서보드에 기록함
        :param count:
        :param avg_reward:
        :return:
        """
        summary = tf.Summary( )
        summary.value.add( tag='avg_total_reward', simple_value=avg_reward )
        summary.value.add( tag='avg_total_count', simple_value=avg_count )
        summary.value.add( tag='per_done', simple_value=per_done )
        summary.value.add( tag='avg_done_count', simple_value=avg_done_count )
        summary.value.add( tag='avg_done_reward', simple_value=avg_done_reward )
        self.writer.add_summary( summary, playtime )

    def get_action( self, state=None, train=True ) :
        """
        상태를 입력받아 행동을 리턴합니다.
        입력받은 상태는 자동으로 이전 상태로 입력됩니다.
        랜덤하게 노이즈값을 줘서 오버피팅을 방지한다.
        :param train:
        :return:
        """
        # 만약 상태값이 없을 경우 이전 게임의 상태값을 기반으로 행동을 구한다.
        self.before_state = self.next_state if type( state ) == type( None ) else state

        # 무작위 상황에서 랜덤 값을 내놓는다
        if train and random.random( ) <= self.get_random :
            index = [ random.randrange( self.count_set ) ]
        else :
            # 다음 액션값을 도출할 떄는 state_x에 다음 상태를 넣어준다
            Q_value = self.Q_value.eval( feed_dict={ self.state_x : [ self.before_state ] } )
            # 게임 갯수와 상관없이 각각의 행동을 도출한다.
            index = np.argmax( Q_value, axis=1 )

        # 랜덤 최소값(minimum_random)보다 get_random값이 크면서 동시에 현재 학습량이 최소 학습량(minimum_train) 보다 클경우
        # 순차적으로 랜덤 값을 줄려 나간다.
        if self.get_random > self.minimum_random and self.train_step > self.minimum_step :
            # 랜덤 확률 차감 값을 동적으로 변경 -> (최대확률 - 최소확률)/최소 학습 횟수
            self.get_random -= (1.0 - self.minimum_random) / self.minimum_step  # 임의로 값을
        # 게임이 한개 진행하여 한개의 게임만 내보낸다.
        return index[ 0 ]
