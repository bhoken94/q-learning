import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T
import numpy as np
import os


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, num_actions, lr, net_name, chkp_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, net_name)

        self.c1 = nn.Conv2d(input_dims[0], 32, kernel_size=(8,8), stride=(4,1))
        self.c2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride=(2, 1))
        self.c3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(self.get_input_dims(input_dims), 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_input_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.c1(state)
        dims = self.c2(dims)
        dims = self.c3(dims)
        return int(np.prod(dims.size()))   # ritorniamo il prodotto delle dimensioni di x e y

    def forward(self, state):
        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        # La dimensione dopo il 3 strato conv è batch_size z num_filters x H x W
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_checkpoint(self):
        print("=> Save checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("=> Load checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DuelingDQN(nn.Module):
    def __init__(self, input_dims, num_actions, lr, net_name, chkp_dir):
        super(DuelingDQN, self).__init__()
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, net_name)

        self.c1 = nn.Conv2d(input_dims[0], 32, kernel_size=(8,8), stride=(4,1))
        self.c2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride=(2, 1))
        self.c3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(self.get_input_dims(input_dims), 512)

        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, num_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_input_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.c1(state)
        dims = self.c2(dims)
        dims = self.c3(dims)
        return int(np.prod(dims.size()))   # ritorniamo il prodotto delle dimensioni di x e y

    def forward(self, state):
        x = F.relu(self.c1(state))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        # La dimensione dopo il 3 strato conv è batch_size z num_filters x H x W
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        V = self.V(x)
        A = self.A(x)
        return V, A

    def save_checkpoint(self):
        print("=> Save checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("=> Load checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))