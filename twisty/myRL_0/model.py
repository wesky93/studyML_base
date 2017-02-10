import tensorflow as tf
import numpy as np
import random


class cubeDQN :
    def __init__( self, set, cube_size=2, dropout=1 ) :
        """
        트위스티 큐브 Deep Q Netwarks 클래스
        :param set: 명령어 모음집
        :param cube_size: 큐브 크기
        :param dropout:
        """

        # 큐브 크기 - 포켓큐브 크기가 작아서 cnn이 제대로 안되면 루빅스큐브로 변경하기
        self.cube_size = cube_size
        # 큐브 명령어 갯수 - 기본 12 + 무반응 1개 추가
        self.set = set
        self.count_set = len( set )

        # 진행 횟수
        self.count_step = 0
        # 최소 자료 축적 횟수
        self.minimum_train = 1000

        # 1차 신경망 뉴런수
        # todo: 뉴런 갯수 조절 필요
        self.num_filters1 = 36
        # 1차 신경망 필터 사이즈
        self.size_filter1 = 4

        # 2차 신경망 뉴런수
        self.num_filters2 = 72
        # 2차 신경망 필터 사이즈
        self.size_filter2 = 4

        self.num_filters3 = 0
        self.size_filter3 = 2
        # 보상 감가상액 비율
        self.GAMMA = 0.99

        # 최종단계 뉴런 갯수
        self.full_neuron = 1024

        # getaction 램덤 확률
        self.get_random = 1.0
        self.minimum_random = 0.01

        # 큐브 상태 shape
        self.state_shapeX = self.cube_size*3
        self.state_shapeY = self.cube_size*4
        # 큐브 상태입력
        self.state_x = tf.placeholder( tf.float32, [ None, self.state_shapeX, self.state_shapeY ], name='state' )
        self.input_x = tf.reshape(self.state_x,[-1,self.state_shapeX,self.state_shapeY,1],name='input_x')
        # 회전 방향 입력
        self.action = tf.placeholder( tf.float32, [ None, self.count_set ], name="action" )
        self.reward_y = tf.placeholder( tf.float32, [ None ], name="reward_y" )

        # 모델 생성
        self.Q_value, self.train_opti = self.build_model( )
        # 세션
        self.session = self.init_session( )
        self.writer = tf.summary.FileWriter( 'logs', self.session.graph )

        self.summary = tf.summary.merge_all( )

    def init_session( self ) :
        session = tf.InteractiveSession( )
        session.run( tf.global_variables_initializer( ) )
        return session

    def build_model( self ) :
        # todo: FCNN레이어 층 추가하기
        # todo: state 입력 층 변경하기
        # todo: 바이어스 제거하기
        # 할일: 첫번째 필터를 4*4로 바꾸기
        W_conv1 = tf.Variable( tf.truncated_normal( [ self.size_filter1, self.size_filter1, 1, self.num_filters1 ] ) )
        # 1차 신경망 적용
        h_conv1 = tf.nn.conv2d( self.input_x, W_conv1, strides=[ 1, 1, 1, 1 ], padding='SAME' ,name='L_Input')
        h_conv1_cutoff = tf.nn.relu( h_conv1 )
        print(h_conv1_cutoff)
        # 6*8 -> 6*8 유자
        h_conv1_shape = (self.state_shapeX, self.state_shapeY)

        # 2차 신경망 적용
        W_conv2 = tf.Variable(
                tf.truncated_normal( [ self.size_filter2, self.size_filter2, self.num_filters1, self.num_filters2 ] ) )
        h_conv2 = tf.nn.conv2d( h_conv1_cutoff, W_conv2, strides=[ 1, 1, 1, 1 ] ,padding='VALID',name='L_hidden1')
        h_conv2_cutoff = tf.nn.relu( h_conv2 )
        # 6*8 -> 3*5 로 바뀜
        h_conv2_shape = (h_conv1_shape[0] - self.size_filter2 + 1, h_conv1_shape[1] - self.size_filter2 + 1)
        print(h_conv2_cutoff)

        # 3차 신경망 적용
        W_conv3 = tf.Variable(tf.truncated_normal([self.size_filter3,self.size_filter3,self.num_filters2,self.num_filters3]))
        h_conv3 = tf.nn.conv2d(h_conv2_cutoff,W_conv3,strides=[1,1,1,1],padding='VALID',name='L_hidden2')
        h_conv3_cutoff = tf.nn.relu(h_conv3)
        # 3*5 -> 2*4
        h_conv3_shape = (h_conv2_shape[ 0 ] - self.size_filter3 + 1, h_conv2_shape[ 1 ] - self.size_filter3 + 1)

        # 풀 커넥티드 레이러을 위한 입력값 갯수(n*n*num_filters2)
        full_unit1 = h_conv3_shape[0]*h_conv3_shape[1] * self.num_filters3

        # n*n 행렬 num_filters2개를 1차원 행렬로 만든다
        h_conv3_flat = tf.reshape( h_conv2_cutoff, [ -1, full_unit1 ] )

        # 풀 커넥티드 레이어
        w2 = tf.Variable( tf.truncated_normal( [ full_unit1, self.full_neuron ] ) )
        fully_conect = tf.nn.relu( tf.matmul( h_conv3_flat, w2 )  )

        # Q_value
        w0 = tf.Variable( tf.zeros( [ self.full_neuron, self.count_set ] ) )
        b0 = tf.Variable( tf.zeros( [ self.count_set ] ) )
        Q_value = tf.matmul( fully_conect, w0 ) + b0

        # DQN 손실 함수
        Q_action = tf.reduce_sum( tf.mul( Q_value, self.action ), axis=1 )
        cost = tf.reduce_mean( tf.square( self.reward_y - Q_action ) )
        tf.summary.scalar( 'cost', cost )
        train_op = tf.train.AdamOptimizer( 1e-6 ).minimize( cost )

        return Q_value, train_op

    # 매 스탭마다 학습하도록 함
    def step( self, state, action, reward, history, train=True ) :
        """
        학습 실행 - 기본적으로 다수의 큐브게임을 동시에 플레이 할 수 있다.
        그래서 state,action,reward 모두 돌리는 게임 갯수만큼 리스트에 관련사항을 담아 전달 해야 함
        :param state:
        :param action:
        :param reward:
        :return:
        """
        play_count = len( history )
        self.count_step += 1
        if play_count > 1 :
            # 이전 스텝에서의 상태를 현재 액션 전 상태로 변경
            self.before_state = self.next_state
        # 현재 액션 후 상태 할당
        self.next_state = state
        # 현재 액션 할당
        act = [ np.zeros( self.count_set ) ]
        act[ 0 ][ action ] = 1

        # 학습
        if train and play_count > 1 :
            if play_count == 1 :
                reward_y = [reward]
            else :
                Q_value = self.Q_value.eval(
                        feed_dict={ self.state_x : self.next_state } )
                reward_y = [reward + self.GAMMA * np.max( Q_value[0] )]
            trainlog = self.train_opti.run(
                    feed_dict={ self.reward_y : reward_y, self.state_x : self.before_state, self.action : act } )
            # 텐서보드에 기록
            if self.count_step % 100 == 0:
                summary = self.summary.eval(feed_dict={ self.reward_y : reward_y, self.state_x : self.before_state, self.action : act })
                self.writer.add_summary(summary,self.count_step/100)
        else :
            # 학습 모드가 아닐 경우 바로 액션값을 넘겨준다
            return self.get_action( train=False )

    def get_action( self, train=True ) :
        """
        행동을 가져옵니다. 학습모드가 아닐 경우 랜덤하게 값을 가져오지 않습니다.
        :param train:
        :return:
        """
        # 무작위 상황에서 랜덤 값을 내놓는다
        if train and random.random( ) <= self.get_random :
            index = random.randrange( self.count_set )
        else :
            # 다음 액션값을 도출할 떄는 state_x에 다음 상태를 넣어준다
            Q_value = self.Q_value.eval( feed_dict={ self.state_x : self.next_state } )
            index = np.argmax( Q_value )

        # 랜덤 최소값(minimum_random)보다 get_random값이 크면서 동시에 현재 학습량이 최소 학습량(minimum_train) 보다 클경우
        # 순차적으로 랜덤 값을 줄려 나간다.
        if self.get_random > self.minimum_random and self.count_step > self.minimum_train :
            self.get_random -= 0.99 / 1000  # 임의로 값을

        return index
