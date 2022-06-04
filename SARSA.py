from game import AI_game
import game
import numpy as np

STAND_ACTION = 0

HIT_ACTION = 1


class SARSA:
    def __init__(self, q_function=None, episodes=100, learning_rate=0.01, gamma=1.0):
        self.q_function = q_function
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.convergance = 100
        self.state = None
        self.game = AI_game()
        self.number_seen = np.zeros(np.shape(q_function))
        self.t = 0
        self.start_learning = 1
        self.games_over_21 = 0
        self.games_under_21 = 0

    def update(self, state, action, reward, next_state, next_action):
        self.number_seen[state][action] += 1
        change = (
            reward
            + self.q_function[next_state][next_action]
            - self.q_function[state][action]
        )
        learning_rate = 1 / self.number_seen[state][action]
        self.q_function[state][action] = (
            self.q_function[state][action] + learning_rate * change
        )

    def get_sample(self):
        self.t += 1
        if self.state is None:
            self.state = self.game.start_game()

        state = self.state
        action = self.get_action(state)

        if action == HIT_ACTION and state < 22:
            self.state = self.game.hit()
            if self.state > 21:
                return (state, action, game.LOSING_SCORE, 0, STAND_ACTION)
            return (
                state,
                action,
                game.LOSING_SCORE,
                self.state,
                self.get_action(state),
            )
        else:
            self.game.stand()
            reward = self.game.end_score()
            self.state = self.game.start_game()
            return (state, action, reward, 0, STAND_ACTION)

    def get_action(self, state):
        p1 = np.exp(self.q_function[state][0])
        p2 = np.exp(self.q_function[state][1])
        p = np.random.random()

        if p < p1 / (p1 + p2):
            return 0
        else:
            return 1

        # eps = 1 / self.t
        # p = np.random.random()
        # if p < eps or self.t < self.start_learning:
        #     return np.random.choice([0, 1])
        # else:
        #     return np.argmax(self.q_function[state])

    def train(self):
        did_not_change_streak = 0
        for i in range(self.episodes):
            old = np.copy(self.q_function)
            state, action, reward, next_state, next_action = self.get_sample()
            self.update(state, action, reward, next_state, next_action)
            new = self.q_function
            distance = np.max(abs(new - old))
            if distance == 0:
                did_not_change_streak += 1
            else:
                did_not_change_streak = 0
            if did_not_change_streak > self.convergance:
                print("converged, i = ", i)
        print("did not converged", distance)

    def winning_chance(self, number_of_runs):
        rewards = 0
        self.state = self.game.start_game()
        for i in range(number_of_runs):
            while self.state is not None:
                state, action, reward, next_state = self.get_sample()
            rewards += reward
            self.state = self.game.start_game()
        return rewards / number_of_runs

    def get_starting_distribution(self, number_of_runs):
        self.state = None
        ampirical = np.zeros(np.shape(self.q_function))
        for i in range(number_of_runs):
            state, action, reward, next_state = self.get_sample()
            ampirical[state] += 1
            self.state = None
        return ampirical / number_of_runs

    def get_policy(self):
        return [
            np.argmax(self.q_function[state]) for state in range(len(self.q_function))
        ]


def main():
    TD = SARSA(
        q_function=np.zeros((35, 2)),
        episodes=10 ** 8,
        learning_rate=0.01,
        gamma=1,
    )
    TD.train()
    print(TD.q_function)
    print(TD.get_policy())
    # print(TD.winning_chance(1000000))
    # starting_dist = TD.get_starting_distribution(1000000)
    # print(np.inner(starting_dist, TD.value_function))


if __name__ == "__main__":
    main()
