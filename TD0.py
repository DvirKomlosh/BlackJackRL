from game import AI_game
import game
import numpy as np

HIT_ACTION = 1
STAND_ACTION = 0


class TDzero:
    def __init__(
        self, value_function=None, episodes=100, learning_rate=0.01, gamma=1.0
    ):
        self.value_function = value_function
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.convergance = 100
        self.game = AI_game()
        self.state = None
        self.number_seen = np.zeros(len(value_function))

    def update(self, state, action, reward, next_state):
        self.number_seen[state] += 1

        self.value_function[state] = self.value_function[state] + (
            1 / self.number_seen[state]
        ) * (
            reward
            + self.gamma * self.value_function[next_state]
            - self.value_function[state]
        )

    def get_sample(self):
        if self.state is None:
            self.state = self.game.start_game()

        state = self.state
        if state < 18:
            self.state = self.game.hit()
            return (state, HIT_ACTION, game.LOSING_SCORE, self.state)
        else:
            self.game.stand()
            reward = self.game.end_score()
            self.state = None
            return (state, STAND_ACTION, reward, 0)

    def train(self):
        did_not_change_streak = 0
        for i in range(self.episodes):
            old = np.copy(self.value_function)
            state, action, reward, next_state = self.get_sample()
            self.update(state, action, reward, next_state)
            new = self.value_function
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
        ampirical = np.zeros(len(self.value_function))
        for i in range(number_of_runs):
            state, action, reward, next_state = self.get_sample()
            ampirical[state] += 1
            self.state = None
        return ampirical / number_of_runs


def main():
    TD = TDzero(
        value_function=np.zeros(32), episodes=1000000, learning_rate=0.01, gamma=0.99
    )
    TD.train()
    print(TD.value_function)
    print(TD.winning_chance(1000000))
    starting_dist = TD.get_starting_distribution(1000000)
    print(np.inner(starting_dist, TD.value_function))


if __name__ == "__main__":
    main()
