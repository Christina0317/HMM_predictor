import numpy as np
from tqdm import tqdm


class DiscreteHMM:
    """
    Parameters
    ----------
    n_window : int
            Length of observation sequences.
    n_states: int
            Number of hidden states.
    n_iter : int, optional
            Maximum number of iterations to perform.
    em_threshold : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood is below this value.
    :param transmat_: transition matrix, array-like, shape (n_states, n_states)
    :param emitmat_: emission matrix, array-like, shape (n_states, n_symbols)
    """
    def __init__(self, n_states, n_symbols, n_iter, n_window, em_threshold):
        self.n_states = n_states    # 隐状态数量
        self.n_symbols = n_symbols  # 观测符号数量
        self.n_iter = n_iter
        self.n_window = n_window    # 序列长度
        self.em_threshold = em_threshold
        self.transmat_, self.emitmat_, self.init_prob_ = None, None, None

    def initialize_parameters(self, X):
        A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        B = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)
        pi = np.random.dirichlet(np.ones(self.n_states))
        self.transmat_, self.emitmat_, self.init_prob_ = A, B, pi

    def forward(self, X):
        """
        Forward Algorithm
        :param X: observation sequence, array-like, shape (n_window, )
        :return fwd: forward probability at time t and state j, array-like, shape (n_window, n_states)
        """
        fwd = np.zeros((len(X), self.n_states))
        fwd[0] = self.init_prob_ * self.emitmat_[:, X[0]]

        for t in range(1, self.n_window):
            for j in range(self.n_states):
                fwd[t, j] = np.sum(fwd[t - 1] * self.transmat_[:, j]) * self.transmat_[j, X[t]]  # t时刻观测到序列并处于状态j的概率
        return fwd

    def backward(self, X):
        """
        Backward Algorithm
        :param X: observation sequence, array-like, shape (n_window, )
        :return bwd: backward probability, array-like, shape (n_window, n_states)
        """
        bwd = np.zeros((len(X), self.n_states))
        bwd[-1] = 1

        for t in reversed(range(self.n_window - 1)):
            for i in range(self.n_states):
                bwd[t, i] = np.sum(bwd[t+1] * self.transmat_[i] * self.emitmat_[:, X[t+1]])
        return bwd

    def probability_calculation(self, X):
        """
        计算给定观测序列 X，模型参数 A, B, pi 的概率 P(X|A,B,pi) 即 P(O|lambda)
        :return: P(O|lambda)
        """
        fwd = self.forward(X)
        bwd = self.backward(X)
        prob = np.sum(fwd * bwd, axis=1).reshape(-1, 1)
        return prob

    def em_algorithm(self, X):
        fwd = self.forward(X)
        bwd = self.backward(X)
        previous_log_likelihood = np.inf

        for n in range(self.n_iter):
            # E step
            a = fwd * bwd
            sum_axis1 = np.sum(a, axis=1).reshape(-1, 1)
            gamma = a / sum_axis1
            # 计算 xi
            xi = np.zeros((len(X), self.n_states, self.n_states))
            for t in range(len(X)-1):
                numerator = np.outer(fwd[t, :], bwd[t + 1, :]) * self.transmat_ * np.tile(self.emitmat_[:, X[t + 1]], (self.n_states, 1))
                xi[t, :, :] = numerator / np.sum(numerator)

            # 终止条件
            current_log_likelihood = np.log(np.sum(fwd[-1]))
            if np.abs(current_log_likelihood - previous_log_likelihood) < self.em_threshold:
                # print(f"EM converged at iteration {n}.")
                break
            if np.abs(current_log_likelihood - previous_log_likelihood) > self.em_threshold and n == self.n_iter - 1:
                1
                # print(f"EM did not converge at iteration {n}.")
            previous_log_likelihood = current_log_likelihood

            # M step (update parameters)
            xi_ = xi[:len(X) - 1]
            gamma_ = gamma[:len(X) - 1]
            A = np.sum(xi_, axis=0) / np.sum(gamma_, axis=0)
            B = np.zeros_like(self.emitmat_)
            for k in range(B.shape[1]):
                mask = (X == k)
                B[:, k] = np.sum(gamma[mask, :], axis=0)
            B /= np.sum(gamma, axis=0).reshape(-1, 1)
            pi = gamma[0]
            self.transmat_, self.emitmat_, self.init_prob_ = A, B, pi
        return self.transmat_, self.emitmat_, self.init_prob_

    def viterbi_predict(self, X):
        """
        Viterbi algorithm for computing the most likely hidden state sequence.
        将预测的 hidden state 转换为 observation state
        :param X: observation sequence, array-like, shape (n_window, )
        :return: observation_prob, T+1 的观测态, array, shape (n_symbols, )
        """
        V = np.zeros((self.n_states, self.n_window))
        path = np.zeros((self.n_states, self.n_window), dtype=int)

        # 初始化
        V[:, 0] = self.init_prob_ * self.emitmat_[:, X[0]]

        for t in range(1, self.n_window):
            for j in range(self.n_states):
                seq_probs = V[:, t - 1] * self.transmat_[:, j]
                V[j, t] = np.max(seq_probs) * self.emitmat_[j, X[t]]
                path[j, t] = np.argmax(seq_probs)

        # 回溯最优路径
        best_path = np.zeros(self.n_window, dtype=int)
        best_path[-1] = np.argmax(V[:, -1])
        for t in range(self.n_window - 2, -1, -1):
            best_path[t] = path[best_path[t + 1], t + 1]

        # 预测
        last_hidden_state = best_path[-1]  # 最可能的路径的最后一个 hidden state
        next_hidden_states = self.transmat_[last_hidden_state]
        next_hidden_states_ = np.tile(next_hidden_states, (len(self.emitmat_[0]),1)).T
        observation_prob = next_hidden_states_ * self.emitmat_
        observation_prob = np.sum(observation_prob, axis=0)
        return observation_prob

    def reset_parameters(self, X):
        """重置HMM模型的参数到初始随机值"""
        self.initialize_parameters(X)
        # print("Model parameters have been reset.")


def gaussian_pdf(x, mean, var):
    """Calculate the probability density of x for a Gaussian distribution."""
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


class GaussianHMM:
    """
        Parameters
        ----------
        n_window : int
                Length of observation sequences.
        n_states: int
                Number of hidden states.
        n_iter : int, optional
                Maximum number of iterations to perform.
        em_threshold : float, optional
                Convergence threshold. EM will stop if the gain in log-likelihood is below this value.
        transmat_： transition matrix, array, shape (n_states, n_states)
        mean_: mean of Gaussian distribution, array-like, shape (n_states, )
        var_: var of Gaussian distribution, array-like, shape (n_states, )
        """

    def __init__(self, n_states, n_symbols, n_iter, n_window, em_threshold):
        self.n_states = n_states  # 隐状态数量
        self.n_symbols = n_symbols  # 观测符号数量
        self.n_iter = n_iter
        self.n_window = n_window  # 序列长度
        self.em_threshold = em_threshold
        self.transmat_, self.mean_, self.var_, self.init_prob_ = None, None, None, None

    def initialize_parameters(self, X):
        A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        mean = np.mean(X) + np.random.rand(self.n_states)
        var = np.std(X) * np.ones(self.n_states)
        pi = np.random.dirichlet(np.ones(self.n_states))
        self.transmat_, self.mean_, self.var_, self.init_prob_ = A, mean, var, pi
        return

    def forward(self, X):
        """
        Forward Algorithm
        :param X: observation sequence, array-like, shape (n_window, )
        :return fwd: forward probability at time t and state j, array-like, shape (n_window, n_states)
        """
        fwd = np.zeros((len(X), self.n_states))
        for i in range(self.n_states):
            fwd[0, i] = self.init_prob_[i] * gaussian_pdf(X[0], self.mean_[i], self.var_[i])

        for t in range(1, len(X)):
            for j in range(self.n_states):
                fwd[t, j] = np.sum(fwd[t - 1, :] * self.transmat_[:, j]) * gaussian_pdf(X[t], self.mean_[j], self.var_[j])

        return fwd

    def backward(self, X):
        """
        Backward Algorithm
        :param X: observation sequence, array-like, shape (n_window, )
        :return bwd: backward probability, array-like, shape (n_window, n_states)
        """
        bwd = np.zeros((len(X), self.n_states))
        bwd[-1] = 1

        for t in reversed(range(len(X) - 1)):
            for i in range(self.n_states):
                bwd[t, i] = np.sum(bwd[t+1] * self.transmat_[i, :] * [gaussian_pdf(X[t+1], self.mean_[j], self.var_[j]) for j in range(self.n_states)])
        return bwd

    def probability_calculation(self, X):
        """
        计算给定观测序列 X，模型参数 A, B, pi 的概率 P(X|A,B,pi) 即 P(O|lambda)
        :return: P(O|lambda)
        """
        fwd = self.forward(X)
        bwd = self.backward(X)
        prob = np.sum(fwd * bwd, axis=1).reshape(-1, 1)[-1]
        return prob

    def em_algorithm(self, X):
        fwd = self.forward(X)
        bwd = self.backward(X)
        previous_log_likelihood = np.inf
        for n in range(self.n_iter):
            # E-step: 计算 gamma 和 xi
            gamma = np.zeros((len(X), self.n_states))
            xi = np.zeros((len(X) - 1, self.n_states, self.n_states))
            for t in range(len(X)-1):
                a_1 = np.sum(fwd[t, :] * bwd[t, :])
                a_2 = np.sum([fwd[t, u] * bwd[t+1, v] * self.transmat_[u, v] * gaussian_pdf(X[t+1], self.mean_[v], self.var_[v]) for u in range(self.n_states) for v in range(self.n_states)])
                for i in range(self.n_states):
                    gamma[t, i] = fwd[t, i] * bwd[t, i] / a_1
                    for j in range(self.n_states):
                        xi[t, i, j] = fwd[t, i] * self.transmat_[i, j] * bwd[t+1, j] * gaussian_pdf(X[t+1], self.mean_[j], self.var_[j]) / a_2

            # 终止条件
            current_log_likelihood = np.log(np.sum(fwd[-1]))
            if np.abs(current_log_likelihood - previous_log_likelihood) < self.em_threshold:
                # print(f"EM converged at iteration {n}.")
                break
            if np.abs(
                    current_log_likelihood - previous_log_likelihood) > self.em_threshold and n == self.n_iter - 1:
                # print(f"EM did not converge at iteration {n}.")
                1
            previous_log_likelihood = current_log_likelihood

            # M-step: 更新参数
            for i in range(self.n_states):
                self.init_prob_[i] = gamma[0, i]
                for j in range(self.n_states):
                    self.transmat_[i, j] = np.sum(xi[:-1, i, j]) / np.sum(gamma[:-1, i])

                self.mean_[i] = np.sum(gamma[:, i] * X) / np.sum(gamma[:, i])
                self.var_[i] = np.sum(gamma[:, i] * (X - self.mean_[i]) ** 2) / np.sum(gamma[:, i])

        return self.transmat_, self.mean_, self.var_

    def comparison_predict(self, train_data, history_data, thres_prob_diff):
        """
        预测, 比较训练集与历史数据集, 将训练集的参数记为lambda, 找出历史数据集里面, P(O^*|lambda)最接近P(O|lambda)的
        :param train_data: array, shape (n_window,)
        :param history_data: array, 只希望在这些数据里面找到与训练集相似的集合, 不希望找到太久之前的
        :param thres_prob_diff: int, 如果两个概率的差异小于这个阈值, 则认为找到相似数据, 那就不再继续往前找
        :return: int, 根据训练集预测出的下一个数据
        """
        train_prob = self.probability_calculation(train_data)

        # 初始化
        min_diff = float('inf')
        prob_star = None
        t_star = None
        T = len(history_data)
        D = self.n_window

        for t in range(T-1, -2+D, -1):
            window_data = history_data[t-D+1:t+1]
            window_prob = self.probability_calculation(window_data)

            # abs diff
            prob_diff = abs(train_prob - window_prob)

            # 如果概率差异小于阈值，记录这个窗口和差异，然后停止搜索
            if prob_diff < thres_prob_diff:
                min_diff = prob_diff
                t_star = t
                prob_star = window_prob
                break

            # 如果当前窗口的差异小于之前记录的最小差异，则更新最小差异和时间索引
            if prob_diff < min_diff:
                min_diff = prob_diff
                t_star = t
                prob_star = window_prob

        o_t_star = history_data[t_star]
        if t_star == T-1:
            o_t1_star = train_data[-1]
        else:
            o_t1_star = history_data[t_star+1]
        o_t = train_data[-1]
        o_t1 = o_t + (o_t1_star - o_t_star) * np.sign(train_prob - prob_star)
        return o_t1[0]

    def viterbi_predict(self, X):
        """
        Viterbi algorithm for computing the most likely hidden state sequence.
        将预测的 hidden state 转换为 observation state
        :param X: observation sequence, array-like, shape (n_window, )
        :return: prediction, T+1 的观测态, array, shape (n_symbols, )
        """
        V = np.zeros((self.n_states, len(X)))
        path = np.zeros((self.n_states, len(X)), dtype=int)

        # 初始化
        V[:, 0] = self.init_prob_ * np.array(
            [gaussian_pdf(X[0], self.mean_[i], self.var_[i]) for i in range(self.n_states)])

        for t in range(1, len(X)):
            for j in range(self.n_states):
                seq_probs = V[:, t - 1] * self.transmat_[:, j]
                V[j, t] = np.max(seq_probs) * gaussian_pdf(X[t], self.mean_[j], self.var_[j])
                path[j, t] = np.argmax(seq_probs)

        # 回溯最优路径
        best_path = np.zeros(len(X), dtype=int)
        best_path[-1] = np.argmax(V[:, -1])
        for t in range(len(X) - 2, -1, -1):
            best_path[t] = path[best_path[t + 1], t + 1]

        # 预测
        last_hidden_state = best_path[-1]  # 最可能的路径的最后一个 hidden state
        next_hidden_states = self.transmat_[last_hidden_state]
        most_likely_state_index = np.argmax(next_hidden_states)  # 最有可能出现的隐藏状态索引
        state_mean = self.mean_[most_likely_state_index]
        state_var = self.var_[most_likely_state_index]
        prediction = np.random.normal(state_mean, state_var)
        return prediction

    def reset_parameters(self, X):
        """重置HMM模型的参数到初始随机值"""
        self.initialize_parameters(X)
        # print("Model parameters have been reset.")


class RandomModel:
    def __init__(self, n_symbols, lengths):
        self.n_symbols = n_symbols
        self.lengths = lengths

    def generate_observation_state(self):
        vectors = []
        for l in self.lengths:
            vector = np.random.dirichlet(np.ones(self.n_symbols) * 1)
            vectors.append(vector)
        return vectors

