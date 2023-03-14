import math
import numpy as np
import time
from dictionaries import id_to_word
import datetime

# helpers


def log(x):
    return math.log(x) if x > 0 else -math.inf


def str_matrix(matrix):
    s = ""
    for row in matrix:
        s += " ".join(str(x) for x in row) + "\n"
    return s


class HiddenMarkovModel:
    """
    Class implementation of Hidden Markov Models.
    """

    def __init__(self, A, O):
        """
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        """

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1.0 / self.L for _ in range(self.L)]

    def viterbi(self, x):
        """
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        """

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        #
        # probs stores log probabilities!!!
        probs = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [["" for _ in range(self.L)] for _ in range(M + 1)]

        probs[1] = [
            log(self.A_start[state] * self.O[state][x[0]]) for state in range(self.L)
        ]
        seqs[1] = [str(state) for state in range(self.L)]

        for length in range(2, M + 1):
            for state in range(self.L):
                max_prob = -math.inf
                max_seq = None
                for prev_state in range(self.L):
                    prob = (
                        probs[length - 1][prev_state]
                        + log(self.A[prev_state][state])
                        + log(self.O[state][x[length - 1]])
                    )

                    if prob > max_prob:
                        max_prob = prob
                        max_seq = seqs[length - 1][prev_state] + str(state)

                probs[length][state] = max_prob
                seqs[length][state] = max_seq

        max_seq = max(
            seqs[M],
            key=lambda seq: probs[M][int(seq[-1])] if seq is not None else -math.inf,
        )

        return max_seq

    def forward(self, x, normalize=False):
        """
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        """

        M = len(x)  # Length of sequence.
        alphas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        alphas[1] = [
            self.A_start[state] * self.O[state][x[0]] for state in range(self.L)
        ]

        for length in range(2, M + 1):
            for state in range(self.L):
                alphas[length][state] = self.O[state][x[length - 1]] * sum(
                    [
                        alphas[length - 1][prev_state] * self.A[prev_state][state]
                        for prev_state in range(self.L)
                    ]
                )
            if normalize:
                normalization_constant = sum(alphas[length])
                for state in range(self.L):
                    alphas[length][state] /= normalization_constant

        return alphas

    def backward(self, x, normalize=False):
        """
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        """

        M = len(x)  # Length of sequence.
        betas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        betas[M] = [1.0 for _ in range(self.L)]
        for length in range(M - 1, -1, -1):
            for state in range(self.L):
                betas[length][state] = sum(
                    [
                        betas[length + 1][next_state]
                        * self.A[state][next_state]
                        * self.O[next_state][x[length]]
                        for next_state in range(self.L)
                    ]
                )
            if normalize:
                normalization_constant = sum(betas[length])
                for state in range(self.L):
                    betas[length][state] /= normalization_constant

        return betas

    def supervised_learning(self, X, Y):
        """
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        """
        N = len(X)

        # Calculate each element of A using the M-step formulas.

        self.A = [[0.0 for _ in range(self.L)] for _ in range(self.L)]

        for a in range(self.L):
            for b in range(self.L):
                nominator = 0
                denominator = 0
                for i in range(N):
                    for k in range(1, len(X[i])):
                        if Y[i][k - 1] == a:
                            denominator += 1
                            if Y[i][k] == b:
                                nominator += 1
                self.A[a][b] = nominator / denominator

        # Calculate each element of O using the M-step formulas.

        self.O = [[0.0 for _ in range(self.D)] for _ in range(self.L)]

        for a in range(self.L):
            for w in range(self.D):
                nominator = 0
                denominator = 0
                for i in range(N):
                    for k in range(0, len(X[i])):
                        if Y[i][k] == a:
                            denominator += 1
                            if X[i][k] == w:
                                nominator += 1
                self.O[a][w] = nominator / denominator

    def unsupervised_learning(self, X, N_iters):
        """
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        """
        N = len(X)

        for iter in range(N_iters):
            # measure time of iteration
            start = time.time()
            print("iter:", iter, end="\r")
            new_A = [[0.0 for b in range(self.L)] for a in range(self.L)]
            new_O = [[0.0 for w in range(self.D)] for a in range(self.L)]

            for i in range(N):
                alphas = self.forward(X[i], normalize=True)
                betas = self.backward(X[i], normalize=True)

                for k in range(len(X[i])):
                    add_O = [alphas[k + 1][a] * betas[k + 1][a] for a in range(self.L)]
                    denom = sum(add_O)
                    for a in range(self.L):
                        for w in range(self.D):
                            if X[i][k] == w:
                                new_O[a][w] += add_O[a] / denom

                    if k != 0:
                        add_A = [
                            [
                                alphas[k][a]
                                * self.O[b][X[i][k]]
                                * self.A[a][b]
                                * betas[k + 1][b]
                                for b in range(self.L)
                            ]
                            for a in range(self.L)
                        ]

                        denom = 0
                        for a in range(self.L):
                            for b in range(self.L):
                                denom += add_A[a][b]

                        for a in range(self.L):
                            for b in range(self.L):
                                new_A[a][b] += add_A[a][b] / denom

            # normalizing A
            for a in range(self.L):
                denom = sum(new_A[a])
                for b in range(self.L):
                    new_A[a][b] /= denom

            # normalizing O
            for a in range(self.L):
                denom = sum(new_O[a])
                for w in range(self.D):
                    new_O[a][w] /= denom

            # updating A and O
            self.A = new_A
            self.O = new_O

            # print time of iteration
            print(f"iter: {iter} time: {round(time.time() - start, 2)}s")

    def generate_emission(self, M, end_words=[], seed=None):
        """
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        """

        # transpose self.A
        A_trans = list(zip(*self.A))
        O_trans = list(zip(*self.O))

        A_start = [0.0 for _ in range(self.L)]
        for word in end_words:
            p = O_trans[word]
            for i in range(self.L):
                A_start[i] += p[i]
        A_start = (
            [x / sum(A_start) for x in A_start]
            if sum(A_start) != 0
            else [1 / self.L for _ in range(self.L)]
        )

        # (Re-)Initialize random number generator
        rng = np.random.default_rng(seed=seed)

        emission = []
        states = []

        syllables = 0

        def get_syllable_count(word):
            syllable_c = 0
            for char in word:
                if char == "$" or char == "/":
                    syllable_c += 1
            return syllable_c

        while syllables < M:
            new_state = rng.choice(
                range(self.L),
                p=[p / sum(A_trans[states[-1]]) for p in A_trans[states[-1]]]
                if len(states) > 0
                else A_start,
            )
            states.append(new_state)

            if len(states) == 1:
                seq = rng.choice(end_words)
            else:
                seq = rng.choice(range(self.D), p=self.O[new_state])
            word = id_to_word[seq]
            syllable_c = get_syllable_count(word)

            while syllable_c + syllables > 10:
                seq = rng.choice(range(self.D), p=self.O[new_state])
                word = id_to_word[seq]
                syllable_c = get_syllable_count(word)

            syllables += syllable_c
            emission.append(seq)

        return list(reversed(emission)), list(reversed(states))

    def probability_alphas(self, x):
        """
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        """

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        """
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        """

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum(
            [betas[1][j] * self.A_start[j] * self.O[j][x[0]] for j in range(self.L)]
        )

        return prob

    def export(self, filename):
        """
        Exports the HMM to a file.
        Arguments:
            filename:   Name of the file to export to.
        """
        with open(filename, "w") as f:
            f.write(f"{self.L} {self.D}\n{str_matrix(self.A)}{str_matrix(self.O)}")


def supervised_HMM(X, Y):
    """
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    """
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters, seed=None):
    """
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
        rng:        The random number generator for reproducible result.
                    Default to RandomState(1).
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[rng.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[rng.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM


def load_HMM(filename):
    """
    Helper function to load a trained HMM from a file.
    Arguments:
        filename:   Name of the file to load from.
    """
    with open(filename, "r") as f:
        L, D = map(int, f.readline().split())
        A = [[float(x) for x in f.readline().split()] for _ in range(L)]
        O = [[float(x) for x in f.readline().split()] for _ in range(L)]

    return HiddenMarkovModel(A, O)
