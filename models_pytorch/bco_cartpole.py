import numpy as np
import gymnasium as gym
import torch
from bco import BCO  # your PyTorch/Gymnasium-converted BCO base class


class BCOCartPole(BCO):
    def __init__(self):
        # CartPole has state dim 4, discrete action dim 2
        super().__init__(state_dim=4, action_dim=2)
        self.env = gym.make(
            "CartPole-v0",
        )

    def enable_rendering(self):
        # close old env and open a new one with human render
        self.env.close()
        self.env = gym.make("CartPole-v0", render_mode="human")

    def disable_rendering(self):
        # close human env and go back to headless
        self.env.close()
        self.env = gym.make("CartPole-v0", render_mode=None)

    def pre_demonstration(self):
        """Collect (s, s', a) by sampling uniform random actions."""
        terminal = True
        States, Nstates, Actions = [], [], []

        num_steps = int(round(self.M / self.alpha))
        for i in range(num_steps):
            if terminal:
                obs, _ = self.env.reset()
                state = obs
                terminal = False

            prev_s = state.copy()

            # one-hot random action
            A = np.random.randint(self.action_dim)
            a = np.zeros(self.action_dim, dtype=np.float32)
            a[A] = 1.0

            # step
            obs, _, terminated, truncated, _ = self.env.step(A)
            terminal = terminated or truncated
            state = obs

            States.append(prev_s)
            Nstates.append(state)
            Actions.append(a)

            if (i + 1) % 10000 == 0:
                print(f"Collecting idm training data {i+1}")

        return States, Nstates, Actions

    def post_demonstration(self):
        """Collect (s, s', a) by rolling out current policy."""
        terminal = True
        States, Nstates, Actions = [], [], []

        for i in range(self.M):
            if terminal:
                obs, _ = self.env.reset()
                state = obs
                terminal = False

            prev_s = state.copy()
            # evaluate policy: returns raw logits or action scores
            policy_out = self.eval_policy(state[np.newaxis, :])[0]
            A = int(np.argmax(policy_out))
            # one-hot of chosen action
            a = np.zeros(self.action_dim, dtype=np.float32)
            a[A] = 1.0

            obs, _, terminated, truncated, _ = self.env.step(A)
            terminal = terminated or truncated
            state = obs

            States.append(prev_s)
            Nstates.append(state)
            Actions.append(a)

        return States, Nstates, Actions

    def eval_rwd_policy(self, display=False):
        """Run one episode with the current policy, return total reward."""
        terminated = False
        truncated = False
        total_reward = 0.0

        if display:
            self.enable_rendering()
        obs, _ = self.env.reset()
        state = obs

        while not (terminated or truncated):
            policy_out = self.eval_policy(state[np.newaxis, :])[0]
            A = int(np.argmax(policy_out))
            obs, reward, terminated, truncated, _ = self.env.step(A)
            state = obs
            total_reward += reward
        if display:
            self.disable_rendering()
        return total_reward


if __name__ == "__main__":
    bco = BCOCartPole()
    bco.run()
