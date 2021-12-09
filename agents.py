import numpy as np
import torch as T
import networks as net
from experience_replay import ExperienceReplay


class Agent:
    def __init__(self, input_dims, num_actions, lr, mem_size, batch_size, gamma, epsilon, replace=1000,algo=None, env_name=None,
                 chkpt_dir="models/" ,eps_dec=5e-7, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.n_actions = num_actions
        self.gamma = gamma
        self.eps_min = eps_min
        self.replace_target_counter = replace
        self.algo = algo
        self.batch_size = batch_size
        self.env_name = env_name
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.chkpt_dir = chkpt_dir

        self.memory = ExperienceReplay(capacity=mem_size, input_shape=input_dims, n_actions=self.n_actions)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_counter == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        raise NotImplementedError

    def choose_action(self, observation):
        raise NotImplementedError

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = net.DeepQNetwork(input_dims=self.input_dims, num_actions=self.n_actions,
                                        lr=self.lr, chkp_dir=self.chkpt_dir, net_name=self.env_name+"_"+self.algo+"_q_eval")
        self.q_next = net.DeepQNetwork(input_dims=self.input_dims, num_actions=self.n_actions,
                                       lr=self.lr, chkp_dir=self.chkpt_dir,
                                       net_name=self.env_name + "_" + self.algo + "_q_next")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # creo lista di numeri da 0 a batch_size
        q_pred = self.q_eval.forward(states)[indices, actions]  # uso indices perchè dims -> batch_size x num_actions
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        # Calcolo target value
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

class DoubleDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = net.DeepQNetwork(input_dims=self.input_dims, num_actions=self.n_actions,
                                lr=self.lr, chkp_dir=self.chkpt_dir,
                                net_name=self.env_name + "_" + self.algo + "_q_eval")  # rete che serve solo per gradient descent e back propagation
        self.q_next = net.DeepQNetwork(input_dims=self.input_dims, num_actions=self.n_actions,
                                    lr=self.lr, chkp_dir=self.chkpt_dir,
                                    net_name=self.env_name + "_" + self.algo + "_q_next")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # creo lista di numeri da 0 a batch_size
        q_pred = self.q_eval.forward(states)[indices, actions]  # uso indices perchè dims -> batch_size x num_actions
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)
        # Calcolo target value
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0  # settiamo il valore di q_next nel caso in cui done = True
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

class DuelingDoubleDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDoubleDQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = net.DuelingDQN(input_dims=self.input_dims, num_actions=self.n_actions,
                                lr=self.lr, chkp_dir=self.chkpt_dir,
                                net_name=self.env_name + "_" + self.algo + "_q_eval")  # rete che serve solo per gradient descent e back propagation
        self.q_next = net.DuelingDQN(input_dims=self.input_dims, num_actions=self.n_actions,
                                    lr=self.lr, chkp_dir=self.chkpt_dir,
                                    net_name=self.env_name + "_" + self.algo + "_q_next")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)  # creo lista di numeri da 0 a batch_size
        V_s, A_s = self.q_eval.forward(states)
        V_eval, A_eval = self.q_eval.forward(states_)
        V_s_, A_s_ = self.q_next.forward(states_)

        # Calcolo q function per states e states_
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_eval, (A_eval - A_eval.mean(dim=1, keepdim=True)))
        # Calcolo target value
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0  # settiamo il valore di q_next nel caso in cui done = True
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args,**kwargs)
        self.q_eval = net.DuelingDQN(input_dims=self.input_dims, num_actions=self.n_actions,
                                      lr=self.lr, chkp_dir=self.chkpt_dir,
                                      net_name=self.env_name + "_" + self.algo + "_q_eval")  # rete che serve solo per gradient descent e back propagation
        self.q_next = net.DuelingDQN(input_dims=self.input_dims, num_actions=self.n_actions,
                                      lr=self.lr, chkp_dir=self.chkpt_dir,
                                      net_name=self.env_name + "_" + self.algo + "_q_next")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_control < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # creo lista di numeri da 0 a batch_size
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        # Calcolo q function per states e states_
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
