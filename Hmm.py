import numpy as np


class Hmm:
    def __init__(self):
        """用于将HMM初始化"""
        self.state_number = 3  # 设置状态变量的数量
        self.observation_number = 2  # 设置观测变量的数量
        self.state_transition_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
                                                dtype=np.float64)  # 设置状态转移矩阵的初值
        self.observation_matrix = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]],
                                           dtype=np.float64)  # 设置观测生成矩阵的初值
        self.initial_state_possibility_matrix = np.array([[0.2], [0.4], [0.4]],
                                                         dtype=np.float64)  # 设置初始状态分布矩阵

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
            forward_parameter = np.dot(forward_parameter,
                                       self.state_transition_matrix)  # 通过矩阵内积运算求得状态转移后的概率
            forward_parameter *= self.observation_matrix[:, i - 1]  # 乘上最后的观测概率得到每个状态的前向概率
        if judge:
            return forward_parameter
        else:
            return np.sum(forward_parameter)  # 返回序列概率

    def backward(self, observation_que, judge=True):
        """后向算法，observation_que为要用到的观测序列，judge的作用是判断是否要进一步求得序列概率"""
        backward_parameter = np.ones([self.state_number, 1], dtype=np.float64)  # 初始化后向参数
        for i in reversed(observation_que[1:len(observation_que)]):
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
        """该部分用于实现前向-后向方法求γ参数的值，observation_que为需要的观测序列，time为γ参数中设置的时刻，state为time对应的状态"""
        observation_que1 = observation_que[0:time]  # 划分用于前向的观测序列
        observation_que2 = observation_que[time - 1:]  # 划分用于后向的观测序列
        forward_parameter = self.forward(observation_que1)  # 求前向参数
        backward_parameter = self.backward(observation_que2)  # 求后向参数
        return ((forward_parameter[0, state - 1] * backward_parameter[0, state - 1]) /
                np.dot(forward_parameter, backward_parameter.T))  # 求得γ参数

    def forward_backward_2(self, observation_que, time1, state1, time2, state2):
        """该部分主要用于求解β参数，observation_que为需要的观测序列，time1为贝塔参数中设置的t时刻，state1为time1对应的状态
        time2为贝塔参数中设置的t+1时刻，state2为time2对应的状态"""
        observation_que1 = observation_que[0:time1]  # 划分用于前向的观测序列
        observation_que2 = observation_que[time2 - 1:]  # 划分用于后向的观测序列
        forward_parameter = self.forward(observation_que1)  # 求前向参数
        backward_parameter = self.backward(observation_que2)  # 求后向参数
        k = (forward_parameter[0, state1 - 1] * self.state_transition_matrix[state1 - 1, state2 - 1] *
             self.observation_matrix[state2 - 1, observation_que[time2 - 1] - 1] *
             backward_parameter[0, state2 - 1])  # 求β参数公式的上部分
        "求β参数公式的下部分"
        n = np.dot(self.state_transition_matrix *
                   self.observation_matrix[:, observation_que[time2 - 1] - 1:observation_que[time2 - 1]].T,
                   backward_parameter.reshape(3, 1))
        summary = np.sum(n.T * forward_parameter)
        return [k / summary]  # 返回β参数

    def em_a(self, observation_que):
        """该部分用于更新状态转移矩阵，observation_que为需要的观测序列"""
        temporary_array = np.zeros([self.state_number, self.state_number],
                                   dtype=np.float64)  # 创造一个零时数组存储更新后的值
        """下面的循环用于更新状态转移矩阵，i循环矩阵的行，k循环矩阵的列，n循环时间"""
        for i in range(1, self.state_number + 1):
            for k in range(1, self.state_number + 1):
                g = 0  # 设置零时变量存储更新公式的上部分
                f = 0  # 设置零时变量存储更新公式的下部分
                for n in range(1, len(observation_que)):
                    g += self.forward_backward_2(observation_que, n, i, n + 1, k)[0]  # 求上部分
                    f += self.forward_backward(observation_que, n, i)[0]  # 求下部分
                temporary_array[i - 1, k - 1] = g / f  # 将结果存入零时数组
        self.state_transition_matrix = temporary_array  # 将结果存入状态转移矩阵

    def em_b(self, observation_que):
        """该部分用于更新状态生成矩阵，observation_que为需要的观测序列"""
        temporary_array = np.zeros([self.state_number, self.observation_number],
                                   dtype=np.float64)  # 创造一个零时数组存储更新后的值
        """下面的循环用于更新状态生成矩阵，i循环矩阵的行，k循环矩阵的列，n循环时间"""
        for i in range(1, self.state_number + 1):
            for k in range(1, self.observation_number + 1):
                g = 0  # 设置零时变量存储更新公式的上部分
                f = 0  # 设置零时变量存储更新公式的下部分
                for n in range(1, len(observation_que) + 1):
                    if observation_que[n - 1] == k:  # 判断当前时刻的观测值是否为k
                        g += self.forward_backward(observation_que, n, i)[0, 0]  # 求上部分
                    f += self.forward_backward(observation_que, n, i)[0, 0]  # 求下部分
                temporary_array[i - 1, k - 1] = g / f  # 将结果存入零时数组
        self.observation_matrix = temporary_array  # 将结果存入状态生成矩阵

    def em_pi(self, observation_que):
        """该部分用于更新初始状态概率分布，observation_que为需要的观测序列"""
        temporary_array = np.zeros([1, self.state_number])  # 创造一个零时数组存储更新后的值
        """下面的循环用于更新初始状态概率分布，i用于循环状态数"""
        for i in range(1, self.state_number + 1):
            temporary_array[0, i - 1] = (
                self.forward_backward(observation_que=observation_que,
                                      time=1, state=i))  # 求更新后的初始状态概率并存入零时数组
        print(temporary_array)
        self.initial_state_possibility_matrix = temporary_array  # 将结果存入初始状态概率分布

    def approximate(self, observation_que):
        """该部分用于近似求解观测序列对应的状态系列，observation_que为需要的观测序列"""
        a = []  # 定义空列表用于存储每时刻的状态
        """下面的循环用于求解每时刻的状态，i循环时间，k循环状态"""
        for i in range(1, len(observation_que) + 1):
            b = [0, 0]  # 定义零时列表存储最大γ值及其对应的状态，前一个为状态，后一个为值
            for k in range(1, self.state_number + 1):
                d = self.forward_backward(observation_que, i, k)[0, 0]  # 求γ值
                """判断新的γ值是否更大，是则更新，否则不更新"""
                if d > b[1]:
                    b[0] = k
                    b[1] = d
            a.append(b[0])  # 将求得状态添加到a中
        return a

    def viterbi(self, observation_que):
        """该部分实现了HMM的维特比算法，observation_que为需要的观测序列"""
        a = np.zeros([self.state_number, len(observation_que)], dtype=np.int64)  # 定义零时数组存储每一步的转移节点
        b = (self.initial_state_possibility_matrix.T *
             self.observation_matrix[:, observation_que[0] - 1:observation_que[0]].T)  # 定义零时变量存储该时刻每个状态的概率最大值
        d = []  # 定义空列表存储每时刻的状态
        """下面的for用于求每一步的转移节点，t循环时间"""
        for t in range(2, len(observation_que) + 1):
            e = b.reshape(self.state_number, 1) * self.state_transition_matrix  # 求状态概率公式的前一部分
            f = e * self.observation_matrix[:, observation_que[t - 1] - 1:observation_que[t - 1]].T  # 求状态概率公式的后一部分
            b = f.max(axis=0)  # 求该时刻每个状态的概率最大值
            a[:, t - 1:t] = e.argmax(axis=0).reshape(self.state_number, 1) + 1  # 赋值该时刻每一步的转移节点
        d.insert(0, b.argmax() + 1)  # 从上面循环中获得的最终每个状态的概率最大值中选择概率最大的所对应的状态
        """下面的循环用于倒序遍历得到每时刻对应状态"""
        for i in reversed(range(2, len(observation_que) + 1)):
            d.insert(0, a[d[0] - 1, i - 1])  # 倒序遍历得到上时刻对应状态并插入d中
        return d
