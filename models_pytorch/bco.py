import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from utils import get_shuffle_idx, args, weight_initializer, bias_initializer
from torch.utils.tensorboard import SummaryWriter


class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, action_dim),  # continuous or logits output
        )

    def forward(self, x):
        return self.net(x)


class IDMModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, action_dim),  # continuous or logits output
        )

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


class BCO:
    def __init__(self, state_dim, action_dim):
        # dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # hyperparameters
        self.lr = args.lr
        self.max_episodes = args.max_episodes
        self.batch_size = args.batch_size
        self.alpha = 0.01
        self.M = args.M

        # detect continuous vs discrete from args
        self.continuous = getattr(args, "continuous", False)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize models
        self.policy = PolicyModel(state_dim, action_dim).to(self.device)
        self.idm = IDMModel(state_dim, action_dim).to(self.device)

        # apply TensorFlow-like weight init
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                weight_initializer(m.weight, std=0.1)
                bias_initializer(m.bias, val=0.01)

        self.policy.apply(_init_weights)
        self.idm.apply(_init_weights)

        # set loss functions based on action type
        if self.continuous:
            self.policy_criterion = nn.MSELoss()
            self.idm_criterion = nn.MSELoss()
        else:
            self.policy_criterion = nn.CrossEntropyLoss()
            self.idm_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.opt_idm = optim.Adam(self.idm.parameters(), lr=self.lr)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir="logdir/")

        # placeholders for demonstration data
        self.demo_examples = None
        self.inputs = None
        self.targets = None
        self.num_sample = None

    def load_demonstration(self):
        if args.input_filename is None or not os.path.isfile(args.input_filename):
            raise FileNotFoundError("input filename does not exist")

        inputs, targets = [], []
        for i, line in enumerate(open(args.input_filename)):
            s_str, sp_str = line.strip().replace(",,", ",").split(" ")
            s = eval(s_str)
            sp = eval(sp_str)
            inputs.append(s)
            targets.append(sp)
            if (i + 1) % 10000 == 0:
                print(f"Loading demonstration {i+1}")
            if i + 1 >= 50000:
                break
        return len(inputs), inputs, targets

    def sample_demo(self):
        idx = np.random.choice(
            range(self.demo_examples), self.num_sample, replace=False
        )
        S = [self.inputs[i] for i in idx]
        nS = [self.targets[i] for i in idx]
        return S, nS

    def eval_policy(self, state_batch):
        self.policy.eval()
        with torch.no_grad():
            sb = np.array(state_batch, dtype=np.float32)
            x = torch.from_numpy(sb).to(self.device)
            out = self.policy(x)
        return out.cpu().numpy()

    def eval_idm(self, state_batch, next_state_batch):
        self.idm.eval()
        with torch.no_grad():
            sb = np.array(state_batch, dtype=np.float32)
            nsb = np.array(next_state_batch, dtype=np.float32)
            s = torch.from_numpy(sb).to(self.device)
            ns = torch.from_numpy(nsb).to(self.device)
            out = self.idm(s, ns)
        return out.cpu().numpy()

    def update_policy(self, states, actions):
        self.policy.train()
        for idx in get_shuffle_idx(len(states), self.batch_size):
            batch_s = torch.from_numpy(
                np.array([states[i] for i in idx], dtype=np.float32)
            ).to(self.device)

            if self.continuous:
                batch_a = torch.from_numpy(
                    np.array([actions[i] for i in idx], dtype=np.float32)
                ).to(self.device)
                pred = self.policy(batch_s)
                loss = self.policy_criterion(pred, batch_a)
            else:
                batch_a = np.array([actions[i] for i in idx], dtype=np.float32)
                labels = torch.from_numpy(batch_a.argmax(axis=1).astype(np.int64)).to(
                    self.device
                )
                logits = self.policy(batch_s)
                loss = self.policy_criterion(logits, labels)

            self.opt_policy.zero_grad()
            loss.backward()
            self.opt_policy.step()

    def update_idm(self, states, next_states, actions):
        self.idm.train()
        for idx in get_shuffle_idx(len(states), self.batch_size):
            batch_s = torch.from_numpy(
                np.array([states[i] for i in idx], dtype=np.float32)
            ).to(self.device)
            batch_ns = torch.from_numpy(
                np.array([next_states[i] for i in idx], dtype=np.float32)
            ).to(self.device)

            if self.continuous:
                batch_a = torch.from_numpy(
                    np.array([actions[i] for i in idx], dtype=np.float32)
                ).to(self.device)
                pred = self.idm(batch_s, batch_ns)
                loss = self.idm_criterion(pred, batch_a)
            else:
                batch_a = np.array([actions[i] for i in idx], dtype=np.float32)
                labels = torch.from_numpy(batch_a.argmax(axis=1).astype(np.int64)).to(
                    self.device
                )
                logits = self.idm(batch_s, batch_ns)
                loss = self.idm_criterion(logits, labels)

            self.opt_idm.zero_grad()
            loss.backward()
            self.opt_idm.step()

    def get_policy_loss(self, states, actions):
        self.policy.eval()
        with torch.no_grad():
            s = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)

            if self.continuous:
                a = torch.from_numpy(np.array(actions, dtype=np.float32)).to(
                    self.device
                )
                pred = self.policy(s)
                loss = self.policy_criterion(pred, a)
            else:
                a_np = np.array(actions, dtype=np.float32)
                labels = torch.from_numpy(a_np.argmax(axis=1).astype(np.int64)).to(
                    self.device
                )
                logits = self.policy(s)
                loss = self.policy_criterion(logits, labels)
            return loss.item()

    def get_idm_loss(self, states, next_states, actions):
        self.idm.eval()
        with torch.no_grad():
            s = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
            ns = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
                self.device
            )

            if self.continuous:
                a = torch.from_numpy(np.array(actions, dtype=np.float32)).to(
                    self.device
                )
                pred = self.idm(s, ns)
                loss = self.idm_criterion(pred, a)
            else:
                a_np = np.array(actions, dtype=np.float32)
                labels = torch.from_numpy(a_np.argmax(axis=1).astype(np.int64)).to(
                    self.device
                )
                logits = self.idm(s, ns)
                loss = self.idm_criterion(logits, labels)
            return loss.item()

    def pre_demonstration(self):
        """override: uniform random actions to collect (s, s', a)"""
        raise NotImplementedError

    def post_demonstration(self):
        """override: policy rollout to collect (s, s', a)"""
        raise NotImplementedError

    def eval_rwd_policy(self, display=False):
        """override: compute total reward by running policy in env"""
        raise NotImplementedError

    def train(self):
        S_pre, nS_pre, A_pre = self.pre_demonstration()
        self.update_idm(S_pre, nS_pre, A_pre)

        display = True

        start_time = time.time()
        for ep in range(self.max_episodes):

            def should(freq):
                return freq > 0 and (
                    (ep + 1) % freq == 0 or ep == self.max_episodes - 1
                )

            # POLICY UPDATES
            S, nS = self.sample_demo()
            A_idm = self.eval_idm(S, nS)
            self.update_policy(S, A_idm)
            ploss = self.get_policy_loss(S, A_idm)

            # IDM UPDATES
            S2, nS2, A2 = self.post_demonstration()
            self.update_idm(S2, nS2, A2)
            idmloss = self.get_idm_loss(S2, nS2, A2)

            if should(args.print_freq):
                elapsed = time.time() - start_time
                total_r = self.eval_rwd_policy(display)
                print(
                    f"Episode {ep+1:5d} | reward {total_r:6.1f} | "
                    f"policy loss {ploss:.6e} | idm loss {idmloss:.6e} | "
                    f"{elapsed/args.print_freq:.3f} sec/ep"
                )
                start_time = time.time()

            if should(args.save_freq):
                os.makedirs(args.model_dir, exist_ok=True)
                ckpt = {
                    "policy": self.policy.state_dict(),
                    "idm": self.idm.state_dict(),
                    "opt_policy": self.opt_policy.state_dict(),
                    "opt_idm": self.opt_idm.state_dict(),
                }
                torch.save(ckpt, os.path.join(args.model_dir, "model.pth"))
                print(f"Model saved to {args.model_dir}/model.pth")

    def test(self):
        ckpt_path = os.path.join(args.model_dir, "model.pth")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.idm.load_state_dict(checkpoint["idm"])
        r = self.eval_rwd_policy()
        print(f"\n[Testing]\nFinal reward: {r:6.1f}")

    def run(self):
        os.makedirs(args.model_dir, exist_ok=True)
        if args.mode == "train":
            self.demo_examples, self.inputs, self.targets = self.load_demonstration()
            self.num_sample = self.M
            self.train()
        elif args.mode == "test":
            ckpt_path = os.path.join(args.model_dir, "model.pth")
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f"Checkpoint required for test mode: {ckpt_path}")
            self.test()
        else:
            raise ValueError("Mode must be 'train' or 'test'")
