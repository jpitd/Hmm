import numpy as np


class Hmm:
    def __init__(self):
        self.state_number = 3
        self.observation_number = 2
        self.state_transition_matrix = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
                                                dtype=np.float64)
        self.observation_matrix = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64)
        self.initial_state_possibility_matrix = np.array([[0.2], [0.4], [0.4]], dtype=np.float64)

    def learning_option(self):
        self.state_number = 3
        self.observation_number = 2
        self.state_transition_matrix = np.array([[0.1, 0.6, 0.4], [0.5, 0.4, 0.1], [0.3, 0.5, 0.2]],
                                                dtype=np.float64)
        self.observation_matrix = np.array([[0.4, 0.6], [0.1, 0.9], [0.2, 0.8]], dtype=np.float64)
        self.initial_state_possibility_matrix = np.array([[0.6], [0.1], [0.3]], dtype=np.float64)

    def forward(self, observation_que, judge="true"):
        forward_parameter = self.initial_state_possibility_matrix * self.observation_matrix[:, observation_que[0] - 1]
        for i in observation_que[1::]:
            forward_parameter = np.dot(forward_parameter, self.state_transition_matrix)
            forward_parameter *= self.observation_matrix[:, i - 1]
        if judge:
            return forward_parameter
        else:
            return np.sum(forward_parameter)

    def backward(self, observation_que, judge="true"):
        backward_parameter = np.ones([self.state_number, 1], dtype=np.float64)
        for i in observation_que[::-1]:
            a = self.state_transition_matrix
            a = a * observation_que[:, i - 1]
            backward_parameter = np.dot(a, backward_parameter)
        if judge:
            return backward_parameter.T
        else:
            backward_parameter = np.dot(
                self.initial_state_possibility_matrix * self.observation_matrix[:, observation_que[0]],
                backward_parameter)
            return backward_parameter.T

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
