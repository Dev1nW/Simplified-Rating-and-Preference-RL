import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


device = 'cpu'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=0.0001, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 teacher_num_ratings=2, max_reward=20, k=30): 
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.render = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []

        if teacher_num_ratings >= 2:
            self.num_ratings = teacher_num_ratings
        else:
            print('Invalid number of rating classes given, defaulting to 2...')
            self.num_ratings = 2

        self.max_reward = max_reward
        self.k = k

        self.num_timesteps = 0
        self.member_1_pred_reward = []
        self.member_2_pred_reward = []
        self.member_3_pred_reward = []
        self.real_rewards = []

        self.frames = []

        self.add_data_flag = False
        self.needed_feedback = 20


        self.num_timesteps = 0
        self.member_1_pred_reward = []
        self.member_2_pred_reward = []
        self.member_3_pred_reward = []
        self.real_rewards = []
        self.feedback_count = 0

        self.frames = []
    
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
                
    def add_data_batch(self, obses, rewards, render_obs):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
            self.render.append(render_obs[index]) # 32 x 1000 x 84 x 84 x 3

        if len(self.inputs) > 128:
            self.inputs = self.inputs[32:]
            self.targets = self.targets[32:]
            self.render = self.render[32:]

        self.add_data_flag = True
        #for each segment in render_obs since now we have a (32 x 1000 x 84 x 84 x 3) where 32 is the segment 
    
    def get_mean_and_std(self, x_1):
        probs = []
        rewards = []
        for member in range(self.de):
            with torch.no_grad():
                r_hat = self.r_hat_member(x_1, member=member)
                r_hat = r_hat.sum(axis=1)
                rewards.append(r_hat.cpu().numpy())

        rewards = np.array(rewards)

        return np.mean(rewards, axis=0).flatten(), np.std(rewards, axis=0).flatten()
    
    def r_hat_member(self, x, member=-1):
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat = r_hat1                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=100):
        
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1 = None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        train_render = np.array(self.render[:max_len], dtype = np.uint8)
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] 
        r_t_1 = train_targets[batch_index_1] 
        render_sa_t_1 = train_render[batch_index_1]
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) 
        render_sa_t_1 = np.reshape(render_sa_t_1, (-1, 84, 84, 3))
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) 

        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) 
        render_sa_t_1 = np.take(render_sa_t_1, time_index_1, axis=0)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
                
        return sa_t_1, render_sa_t_1, r_t_1

    def put_queries(self, sa_t_1, labels, rewards):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            np.copyto(self.buffer_reward[self.buffer_index:self.capacity], rewards[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                np.copyto(self.buffer_reward[0:remain], rewards[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            np.copyto(self.buffer_reward[self.buffer_index:next_index], rewards)
            self.buffer_index = next_index
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                labels = self.buffer_label[idxs]

                num_ratings = [0]*self.num_ratings
                for label in labels:
                    num_ratings[int(label[0])] += 1
                
                num_ratings_in_class = []

                labels = torch.from_numpy(labels.flatten()).long().to(device)

                target_onehot = F.one_hot(labels, num_classes=self.num_ratings)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat = r_hat1

                pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                sorted_indices = pred[:, 0].sort()[1]
                np_pred = pred[sorted_indices]
                np_pred = np_pred.tolist()

                # bounds calculations
                bounds = [torch.as_tensor([0]).to(device)] * (self.num_ratings + 1)
                for i in range(len(num_ratings)):
                    # Regular
                    bounds[i+1] = torch.as_tensor(np_pred[sum(num_ratings[0:i+1])-1]).to(device)

                Q = [None] * self.num_ratings
                
                for i in range(len(bounds)-1):
                    Q[i] = -(self.k)*(pred-bounds[i])*(pred-bounds[i+1])

                our_Q = torch.cat(Q, axis=-1)
                curr_loss = self.CEloss(our_Q, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                    
                # compute acc
                _, predicted = torch.max(our_Q, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                labels = self.buffer_label[idxs]

                num_ratings = [0]*self.num_ratings
                for label in labels:
                    num_ratings[int(label[0])] += 1
                
                num_ratings_in_class = []

                labels = torch.from_numpy(labels.flatten()).long().to(device)

                target_onehot = F.one_hot(labels, num_classes=self.num_ratings)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat = r_hat1

                pred  = ((r_hat) - (torch.min(r_hat)))/((torch.max(r_hat)) - (torch.min(r_hat)))
                    
                sorted_indices = pred[:, 0].sort()[1]
                np_pred = pred[sorted_indices]
                np_pred = np_pred.tolist()

                # bounds calculations
                bounds = [torch.as_tensor([0]).to(device)] * (self.num_ratings + 1)
                for i in range(len(num_ratings)):
                    # Regular
                    bounds[i+1] = torch.as_tensor(np_pred[sum(num_ratings[0:i+1])-1]).to(device)

                Q = [None] * self.num_ratings
                for i in range(len(bounds)-1):
                    Q[i] = -(self.k)*(pred-bounds[i])*(pred-bounds[i+1])

                our_Q = torch.cat(Q, axis=-1)
                
                curr_loss = self.softXEnt_loss(our_Q, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(our_Q, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    