import numpy as np


class Hmm:
    def __init__(self):
        """用于将HMM初始化"""
        self.state_number = 3  # 设置状态变量的数量
        self.observation_number = 2  # 设置观测变量的数量
        self.state_transition_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
                                                dtype=np.float64)  # 设置状态转移矩阵的初值
        self.observation_matrix = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64)  # 设置观测生成矩阵的初值
        self.initial_state_possibility_matrix = np.array([[0.2], [0.4], [0.4]], dtype=np.float64)  # 设置初始状态分布矩阵

    def learning_option(self):
        """用于重新设置HMM的初值，该函数只是为了验证学习算法的正确性而写，可直接删除"""
        self.state_number = 3
        self.observation_number = 2
        self.state_transition_matrix = np.array([[0.1, 0.6, 0.4], [0.5, 0.4, 0.1], [0.3, 0.5, 0.2]],
                                                dtype=np.float64)
        self.observation_matrix = np.array([[0.4, 0.6], [0.1, 0.9], [0.2, 0.8]], dtype=np.float64)
        self.initial_state_possibility_matrix = np.array([[0.6], [0.1], [0.3]], dtype=np.float64)

    def forward(self, observation_que, judge=True):
        """前向算法，observation_que为要用到的观测序列，judge的作用是判断是否要进一步求得序列概率"""
        forward_parameter = (self.initial_state_possibility_matrix.T *
                             self.observation_matrix[:, observation_que[0] - 1])  # 初始化前向参数
        for i in observation_que[1::]:
            """该for循环是用于递推前向参数"""
            forward_parameter = np.dot(forward_parameter, self.state_transition_matrix)  # 通过矩阵内积运算求得状态转移后的概率
            forward_parameter *= self.observation_matrix[:, i - 1]  # 乘上最后的观测概率得到每个状态的前向概率
        if judge:
            return forward_parameter
        else:
            return np.sum(forward_parameter)  # 返回序列概率

    def backward(self, observation_que, judge=True):
        """后向算法，observation_que为要用到的观测序列，judge的作用是判断是否要进一步求得序列概率"""
        backward_parameter = np.ones([self.state_number, 1], dtype=np.float64)  # 初始化后向参数
        for i in observation_que[:0:-1]:
            """该for循环是用于递推后向参数"""
            a = self.state_transition_matrix * self.observation_matrix[:, i - 1:i].T  # 求后向算法公式中的状态转移和观测的部分
            backward_parameter = np.dot(a, backward_parameter)  # 最后将上面得到的部分和后向概率求内积得到更新的后向概率
        if judge:
            return backward_parameter.T
        else:
            backward_parameter = np.dot(
                self.initial_state_possibility_matrix.T *
                self.observation_matrix[:, observation_que[0] - 1:observation_que[0]].T,
                backward_parameter)  # 求序列概率
            return backward_parameter

    def forward_backward(self, observation_que, time, state):
        observation_que1 = observation_que[0:time]
        observation_que2 = observation_que[time:]
        forward_parameter = self.forward(observation_que1)
        backward_parameter = self.backward(observation_que2)
        return ((forward_parameter[0, state - 1] * backward_parameter[0, state - 1]) /
                np.dot(forward_parameter, backward_parameter.T))

    def forward_backward_2(self, observation_que, time1, state1, time2, state2):
        observation_que1 = observation_que[0:time1]
        observation_que2 = observation_que[time2:]
        forward_parameter = self.forward(observation_que1)
        backward_parameter = self.backward(observation_que2)
        k = (forward_parameter[0, state1 - 1] * self.state_transition_matrix[state1 - 1, state2 - 1] *
             self.observation_matrix[state2 - 1, observation_que[time2 - 1]] * backward_parameter[0, state2 - 1])
        n = np.dot(self.state_transition_matrix * self.observation_matrix[:observation_que[time2 - 1]],
                   backward_parameter.T)
        summary = np.sum(n.T * forward_parameter)
        return k / summary

    def em_a(self, observation_que):
        temporary_array = np.zeros([self.state_number, self.state_number], dtype=np.float64)
        for i in range(0, self.state_number):
            for k in range(0, self.state_number):
                g = 0
                f = 0
                for n in range(0, observation_que.shape[1] - 1):
                    g += self.forward_backward_2(observation_que, n, i, n + 1, k)
                    f += self.forward_backward(observation_que, n, i)
                temporary_array[i, k] = g / f
        self.state_transition_matrix = temporary_array

    def em_b(self, observation_que):
        temporary_array = np.zeros([self.state_number, self.observation_number], dtype=np.float64)
        for i in range(0, self.state_number):
            for k in range(0, self.observation_number):
                g = 0
                f = 0
                for n in range(0, observation_que.shape[1]):
                    if observation_que[0, n] == k:
                        g += self.forward_backward(observation_que, n, i)
                    f += self.forward_backward(observation_que, n, i)
                temporary_array[i, k] = g / f
        self.observation_matrix = temporary_array

    def em_pi(self, observation_que):
        temporary_array = np.zeros([1, self.state_number])
        for i in range(0, self.state_number):
            temporary_array[0, i] = self.forward_backward(observation_que, 1, i)
        self.initial_state_possibility_matrix = temporary_array

    def approximate(self, observation_que):
        a = []
        for i in range(0, len(observation_que)):
            b = [0, 0]
            for k in range(0, self.state_number):
                d = self.forward_backward(observation_que, i, k)
                if d > b[1]:
                    b[0] = k
                    b[1] = d
            a.append(b[0])
        return a

    def viterbi(self, observation_que):
        a = np.zeros([self.state_number, len(observation_que)])
        b = self.initial_state_possibility_matrix * self.observation_matrix[:observation_que[0] - 1]
        d = []
        for t in range(0, len(observation_que)):
            e = b.reshape(self.state_number, 0) * self.state_transition_matrix
            f = e * self.observation_matrix[:, observation_que[t] - 1:observation_que[t]].T
            b = f.max(axis=0)
            a[:, t:t + 1] = e.argmax(axis=0).reshape(self.state_number, 0)
        d.insert(0, b.argmax())
        for i in range(len(observation_que, stop=-1, step=-1)):
            d.insert(0, np.where(a[:, i - 1:i] == d[0])[0])
        return d
