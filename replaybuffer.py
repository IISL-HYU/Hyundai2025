import numpy as np
from collections import deque
import random


class ReplayBuffer(object):
    """
    Replay Buffer 클래스
    강화학습에서 에이전트가 경험한 transition(state, action, reward, next_state, done)을 저장하고,
    무작위로 샘플링하여 학습에 사용하도록 돕는 역할을 함.
    """

    def __init__(self, buffer_size):
        """
        버퍼 초기화
        :param buffer_size: 버퍼 최대 크기 (저장 가능한 transition 최대 개수)
        """
        self.buffer_size = buffer_size
        self.buffer = deque()  # 양쪽 끝에서 빠른 추가/삭제가 가능한 덱(deque) 사용
        self.count = 0         # 현재 저장된 transition 개수

    def add_buffer(self, state, action, reward, next_state, done):
        """
        버퍼에 새로운 transition 저장
        :param state: 현재 상태
        :param action: 현재 상태에서 취한 행동
        :param reward: 행동 후 받은 보상
        :param next_state: 다음 상태
        :param done: 에피소드 종료 여부 (True/False)
        """
        transition = (state, action, reward, next_state, done)

        # 버퍼가 꽉 차지 않은 경우, 단순히 append
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            # 버퍼가 꽉 찼으면 가장 오래된 transition을 버리고 새로운 transition 추가
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        """
        버퍼에서 랜덤하게 batch_size 만큼 샘플링하여 배치 생성
        :param batch_size: 추출할 배치 크기
        :return: states, actions, rewards, next_states, dones (배치별 numpy 배열)
        """
        # 현재 버퍼에 저장된 transition 개수가 batch_size보다 작으면,
        # 있는 것만큼 랜덤 샘플링 (즉, count만큼 추출)
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # 배치에서 각각의 요소만 추출하여 배열로 변환
        states = np.asarray([i[0] for i in batch])             # 상태 배열
        actions_raw = [i[1] for i in batch]                    # 행동 리스트 (형태 다양할 수 있음)
        rewards = np.expand_dims(np.asarray([i[2] for i in batch]), axis=1)      # 보상 배열 (열 벡터)
        next_states = np.asarray([i[3] for i in batch])        # 다음 상태 배열
        dones = np.expand_dims(np.asarray([i[4] for i in batch]), axis=1)        # 에피소드 종료 여부 배열 (열 벡터)

        # actions 처리: 
        # - 만약 actions가 스칼라(단일 값)일 경우 차원 확장 (2차원 배열로 맞춤)
        # - 아니면 numpy 배열로 변환
        if np.isscalar(actions_raw[0]) or np.array(actions_raw[0]).ndim == 0:
            actions = np.expand_dims(np.asarray(actions_raw), axis=1)
        else:
            actions = np.asarray(actions_raw)

        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        """
        현재 버퍼에 저장된 transition 개수 반환
        """
        return self.count

    def clear_buffer(self):
        """
        버퍼 내용 전부 삭제하고 초기화
        """
        self.buffer = deque()
        self.count = 0
