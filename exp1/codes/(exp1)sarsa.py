import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import time
import copy
import random

# EXPERIMENT 1： indecomposable, true Q visitation, vdn max//vdn sarsa//true Q
# Experiment 2: decomposable vdn max=true Q max
class Action_Selector():
    def __init__(self, world_size, q_mask_perm):
        self.world_size = world_size
        self.prob_tabular = np.random.rand(world_size**4, 16) + 0.5
        self.prob_tabular = th.from_numpy(self.prob_tabular)

    def select_acs(self, state, epsl, q_mask):
        state_id = (state[0]*self.world_size+state[1])*self.world_size**2+(state[2]*self.world_size+state[3])
        q_mask = q_mask.view(1,-1).squeeze(0)
        state_prob = self.prob_tabular[state_id]
        ac = th.multinomial(state_prob,1)
        acs = th.tensor([ac // 4, ac % 4])
        return ac, acs

class Env():
    def __init__(self, world_size, n_agents):
        self.world_size = world_size
        self.n_agents = n_agents
        self.world = np.zeros((self.world_size,self.world_size))
        # assign landmark
        self.tars = np.array([1,0,1,0])
        self.reset()
        self.ret = 0

    def reset(self):
        self.ret = 0
        self.world[1,1] = 7
        self.world[2,2] = 8
        self.agent_active = np.ones((self.n_agents))
        posx = self.world_size-1
        posy = 0
        self.poses = np.array([[posx,posy] for _ in range(self.n_agents)])
        return self.poses
    
    def step(self,ac,render=False):
        reward = 0
        terminated = False
        
        acs = th.zeros(self.n_agents)
        tail = ac
        for i in range(self.n_agents):
            acs[i] = tail // (4 ** (self.n_agents-i-1))
            tail = tail % (4 ** (self.n_agents-i-1))
    
        for i in range(self.n_agents):
            if self.agent_active[i]:
                if acs[i] == 0:
                    self.poses[i][0] -= 1
                elif acs[i] == 1:
                    self.poses[i][0] += 1
                elif acs[i] == 2:
                    self.poses[i][1] -= 1
                elif acs[i] == 3:
                    self.poses[i][1] += 1
                if self.world[self.poses[i][0],self.poses[i][1]] and (self.world[self.poses[i][0],self.poses[i][1]] % 2 == self.tars[i]):
                    reward += 1
                    self.agent_active[i] = 0
                    self.world[self.poses[i][0],self.poses[i][1]] = self.world[self.poses[i][0],self.poses[i][1]] // 2

        if self.world[1,1] == 1 and self.world[2,2] == 2:
            terminated = True
        if render:
            time.sleep(0.1)
            self.render()
            if terminated:
                print("-----terminated")
                time.sleep(1)

        return self.poses, reward, terminated
        
    def render(self):
        render_world = copy.deepcopy(self.world)
        for i in range(self.n_agents):
            if (self.poses[i][0]==1 and self.poses[i][1]==1) or (self.poses[i][0]==2 and self.poses[i][1]==2):
                pass
            else:
                render_world[self.poses[i][0],self.poses[i][1]] = i+1
        print(render_world,"\n")
    
class Env_ind():
    def __init__(self, world_size, n_agents):
        self.world_size = world_size
        self.n_agents = n_agents
        self.world = np.zeros((self.world_size,self.world_size))
        self.reset()
        self.ret = 0

    def reset(self):
        self.ret = 0
        self.world[1,1] = 4
        self.world[2,2] = 4
        self.agent_active = np.ones((self.n_agents))
        posx = self.world_size-1
        posy = 0
        self.poses = np.array([[posx,posy] for _ in range(self.n_agents)])
        return self.poses
    
    def step(self,ac,render=False):
        reward = 0
        terminated = False
        
        acs = th.zeros(self.n_agents)
        tail = ac
        for i in range(self.n_agents):
            acs[i] = tail // (4 ** (self.n_agents-i-1))
            tail = tail % (4 ** (self.n_agents-i-1))
    
        for i in range(self.n_agents):
            if self.agent_active[i]:
                if acs[i] == 0:
                    self.poses[i][0] -= 1
                elif acs[i] == 1:
                    self.poses[i][0] += 1
                elif acs[i] == 2:
                    self.poses[i][1] -= 1
                elif acs[i] == 3:
                    self.poses[i][1] += 1
                if self.world[self.poses[i][0],self.poses[i][1]] and (self.world[self.poses[i][0],self.poses[i][1]] % 2 == 0):
                    reward += 1
                    self.agent_active[i] = 0
                    self.world[self.poses[i][0],self.poses[i][1]] = self.world[self.poses[i][0],self.poses[i][1]] // 2
                    
        if self.world[1,1] == 1 and self.world[2,2] == 1:
            terminated = True
        if render:
            time.sleep(0.1)
            self.render()
            if terminated:
                print("-----terminated")
                time.sleep(1)

        return self.poses, reward, terminated
        
    def render(self):
        render_world = copy.deepcopy(self.world)
        for i in range(self.n_agents):
            if (self.poses[i][0]==1 and self.poses[i][1]==1) or (self.poses[i][0]==2 and self.poses[i][1]==2):
                pass
            else:
                render_world[self.poses[i][0],self.poses[i][1]] = i+1
        print(render_world,"\n")
    

class Qnet_CRC(nn.Module):
    def __init__(self,n_agents, world_size):
        super(Qnet_CRC,self).__init__()
        self.state_shape = n_agents * 2 + 4                    # poses
        self.world_size = world_size
        self.n_agents = n_agents
        self.n_acs = 4 ** self.n_agents                       # 0 1 2 3
        self.net = nn.Sequential(
            nn.Linear(self.state_shape,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self.n_acs)
        )
    
    def forward(self,state):
        return self.net(state)
    
    def select_acs(self, state, epsl):
        state = state.view(1,-1).float()
        q_mask = self._mask_from_state(state)
        for i in range(self.n_agents):
            q_mask[i][1] = 0
            q_mask[i][2] = 0
        qcrc = self.forward(state)

        qcrc = qcrc.reshape(4,4,4,4)
        qcrc[q_mask[0]==0,:,:,:]=-9999
        qcrc[:,q_mask[1]==0,:,:] = -9999
        qcrc[:,:,q_mask[2]==0,:]=-9999
        qcrc[:,:,:,q_mask[3]==0] = -9999

        qcrc = qcrc.reshape(1,-1)

        cc = np.random.rand(1)
        if cc > epsl:
            ac = qcrc.max(dim=-1)[1]
        else:
            ac = th.randint(0, 4**self.n_agents,()).unsqueeze(0)
            while qcrc[0,ac] == -9999:
                ac = th.randint(0, 4**self.n_agents,()).unsqueeze(0)
        
        acs = th.zeros(self.n_agents)
        tail = ac
        for i in range(self.n_agents):
            acs[i] = tail // (4 ** (self.n_agents-i-1))
            tail = tail % (4 ** (self.n_agents-i-1))

        return ac, acs

    def _mask_from_state(self,state):
        poses = state.reshape(-1,2)
        q_mask = th.ones(self.n_agents,4)
        for i in range(self.n_agents):
            if poses[i][0] == 0:
                q_mask[i,0] = 0
            elif poses[i][0] == self.world_size-1:
                q_mask[i,1] = 0

            if poses[i][1] == 0:
                q_mask[i,2] = 0
            elif poses[i][1] == self.world_size-1:
                q_mask[i,3] = 0
        return q_mask


class Qnet_ind(nn.Module):
    def __init__(self):
        super(Qnet_ind,self).__init__()
        self.ob_shape = 6
        self.n_acs = 4
        self.ind_net = nn.Sequential(
            nn.Linear(self.ob_shape,8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,self.n_acs)
        )
    
    def forward(self,obs):
        a = self.ind_net(obs)
        return self.ind_net(obs)
               

class Qnet_vdn():
    def __init__(self,n_agents, world_size):
        self.n_agents = n_agents
        self.world_size = world_size
        self.mods = [Qnet_ind() for _ in range(self.n_agents)]
        self.parameters = list(self.mods[0].parameters())
        for i in range(1,self.n_agents):
            self.parameters += list(self.mods[i].parameters())
    
    def forward(self,obs):
        qtot = th.tensor([],dtype=th.float32)
        for i in range(self.n_agents):
            ac = self.mods[i].forward(obs[i]).unsqueeze(-2)
            qtot = th.cat((qtot,ac),dim=-2)
        return qtot
    
    def select_acs(self, state, obs, epsl):
        q_local = self.forward(obs)
        q_mask = self._mask_from_state(state)
        for i in range(self.n_agents):
            q_mask[i][1] = 0
            q_mask[i][2] = 0
        
        q_local[q_mask==0] = -9999

        cc = np.random.rand(1)
        acs = th.zeros(self.n_agents)
        if cc > epsl:
            acs = q_local.max(dim=-1)[1]
        else:
            for i in range(self.n_agents):
                rand = th.randint(0,4,()).unsqueeze(0)
                while q_mask[i][rand] == 0:
                    rand = th.randint(0,4,()).unsqueeze(0)
                acs[i] = rand
        ac = acs[0] * 4**3 + acs[1] * 4**2 + acs[2] * 4 + acs[3]
        return ac.long().unsqueeze(-1), acs

    def update_target(self, target_mods):
        for i in range(self.n_agents):
            self.mods[i].load_state_dict(target_mods[i].state_dict())
    
    def _mask_from_state(self,state):
        poses = state.reshape(-1,2)
        q_mask = th.ones(self.n_agents,4)
        for i in range(self.n_agents):
            if poses[i][0] == 0:
                q_mask[i,0] = 0
            elif poses[i][0] == self.world_size-1:
                q_mask[i,1] = 0

            if poses[i][1] == 0:
                q_mask[i,2] = 0
            elif poses[i][1] == self.world_size-1:
                q_mask[i,3] = 0
        return q_mask
    
class ReplayBuffer:
    def __init__(self, max_ep_length, buffer_length, n_agents):
        self.n_agents = n_agents    
        self.buffer_length = buffer_length
        self.dict = {"state": th.zeros(self.buffer_length, max_ep_length, 2* self.n_agents+4),
                    "obs": th.zeros(self.buffer_length, max_ep_length, self.n_agents, 6),
                    "terminated": th.ones(self.buffer_length, max_ep_length, 1),
                    "reward": th.zeros(self.buffer_length, max_ep_length, 1),
                    "ac": th.zeros(self.buffer_length, max_ep_length, 1, dtype=th.long),
                    "acs": th.zeros(self.buffer_length, max_ep_length, self.n_agents)
        }
        self.trajs_in_buffer = 0
        self.num_trajs = 0
        self.batch = {}

    def add_ep_traj(self, ep_traj):
        if self.trajs_in_buffer == self.buffer_length:
            for k,v in ep_traj.items():
                ep_length = v.shape[0]
                if k == "terminated":
                    self.dict[k][self.num_trajs % self.buffer_length] = 1
                else:
                    self.dict[k][self.num_trajs % self.buffer_length] = 0
                self.dict[k][self.num_trajs % self.buffer_length][:ep_length] = v
        else:
            for k,v in ep_traj.items():
                ep_length = v.shape[0]
                self.dict[k][self.trajs_in_buffer][:ep_length] = v
            self.trajs_in_buffer += 1

        self.num_trajs += 1

    def sample(self, batch_size):
        ep_ids = np.random.choice(self.trajs_in_buffer, batch_size, replace=False)
        for k,v in self.dict.items():
            self.batch[k] = v[ep_ids]
        return self.batch
    

def run():
    world_size = 4
    target_landmarks = np.array([1,1,2,2])
    ret_list = [[],[]]
    sample_ret_list = []
    th.set_printoptions(profile="short")
    target_update_interval = 50
    n_agents = 4
    epsl = 1.0
    lr = 0.0005
    max_ep_length = 6
    training_iters = 6001
    batch_size = 64
    iters = 0    
    decomposable = False                    # ########
    rl_manner = "sarsa"                     # "sarsa" or "qlearning"
    evaluate_algo = "vdn"                   # qcrc or vdn
    
    file_name = "epgredy_w4a4" 
    if decomposable:
        file_name += "_de"
    else:
        file_name += "_inde_"
    file_name += evaluate_algo
    rand = random.random()
    file_name += str(rand)

    if rl_manner == "sarsa":
        buffer_length = batch_size
    else:
        buffer_length = 1000
    buffer = ReplayBuffer(max_ep_length, buffer_length, n_agents)
    
    if decomposable:
        env = Env(world_size,n_agents)
    else:
        env = Env_ind(world_size,n_agents)

    qnet_true = Qnet_CRC(n_agents, world_size)
    target_qnet = copy.deepcopy(qnet_true)
    q_vdn = Qnet_vdn(n_agents, world_size)
    target_q_vdn = copy.deepcopy(q_vdn)
    qtrue_optim = RMSprop(params=qnet_true.parameters(),lr=lr)
    qvdn_optim = RMSprop(params=q_vdn.parameters,lr=lr)
    
    while iters < training_iters:
        state = env.reset().reshape(1,-1).squeeze(0)
        terminated = False
        episode_data = {"state": th.tensor([]),
                    "obs": th.tensor([]),
                    "terminated": th.tensor([]),
                    "reward": th.tensor([]),
                    "ac": th.tensor([],dtype=th.long),
                    "acs": th.tensor([])
            }

        t = 0
        while t < max_ep_length and (not terminated):
            state = th.from_numpy(state)
            obs = state.reshape(n_agents,-1)
            state = th.cat((state,th.from_numpy(target_landmarks)),dim=0).unsqueeze(0)
            obs = th.cat((obs,th.from_numpy(target_landmarks).repeat(n_agents,1)),dim=-1).float()
            if iters % 100 == 0 and iters > 0:    # test mode
                if iters % 1000 == 0:
                    render = True
                else:
                    render = False
                if evaluate_algo == "qcrc":
                    ac,acs = qnet_true.select_acs(state, 0)
                else:
                    ac,acs = q_vdn.select_acs(state, obs, 0)                
                next_state, reward, terminated = env.step(ac, render=render)

            else:
                if evaluate_algo == "qcrc":
                    ac, acs = qnet_true.select_acs(state, epsl)
                else:
                    ac,acs = q_vdn.select_acs(state, obs, epsl)
                next_state, reward, terminated = env.step(ac)
            
            episode_data["state"] = th.cat((episode_data["state"], state.float()), dim=0)
            episode_data["reward"] = th.cat((episode_data["reward"], th.tensor([reward]).float().unsqueeze(0)), dim=0)
            episode_data["ac"] = th.cat((episode_data["ac"], ac.unsqueeze(0)), dim=0)
            episode_data["acs"] = th.cat((episode_data["acs"], acs.float().unsqueeze(0)), dim=0)
            episode_data["obs"] = th.cat((episode_data["obs"], obs.unsqueeze(0)), dim=0)

            t += 1
            env.ret += reward
            state = next_state.reshape(1,-1).squeeze(0)

            if t >= max_ep_length:
                terminated = True
                sample_ret_list.append(env.ret)
            episode_data["terminated"] = th.cat((episode_data["terminated"], th.tensor([terminated]).float().unsqueeze(0)), dim=0)
        
        if iters % 100 == 0 and iters > 0:
            ret_list[0].append(iters)
            ret_list[1].append(env.ret)
        else:   
            buffer.add_ep_traj(episode_data)

        if epsl >= 0.2:
            epsl -= 0.0005

        # train
        if buffer.trajs_in_buffer >= batch_size:
            data = buffer.sample(batch_size)
            batch_reward = data["reward"]
            batch_ac = data["ac"]
            batch_terminated = data["terminated"]
            batch_state = data["state"]
            batch_acs = data["acs"].unsqueeze(-1)

            batch_obs = th.cat((data["obs"][:,:,0].unsqueeze(0),data["obs"][:,:,1].unsqueeze(0),data["obs"][:,:,2].unsqueeze(0),data["obs"][:,:,3].unsqueeze(0)),dim=0)
            vdn_qs = q_vdn.forward(batch_obs)
            vdn_chosen_qvals = th.gather(vdn_qs, dim=-1, index=batch_acs.long())
            vdn_mixed_qvals = vdn_chosen_qvals.sum(dim=-2)

            true_q_outs = qnet_true.forward(batch_state).reshape(batch_size,-1,4**n_agents)
            chosen_action_qvals = th.gather(true_q_outs, dim=-1, index=batch_ac)
            b = th.zeros_like(chosen_action_qvals)
            chosen_action_qvals_extend = th.cat((chosen_action_qvals,b[:,1].unsqueeze(1)),dim=1)
            vdn_mixed_qvals_extend = th.cat((vdn_mixed_qvals,b[:,1].unsqueeze(1)),dim=1)

            if rl_manner == "sarsa":
                targets = batch_reward + 0.99 * (1 - batch_terminated) * chosen_action_qvals_extend[:,1:]
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * vdn_mixed_qvals_extend[:,1:]
                mask = 1- th.cat((th.zeros_like(batch_terminated)[:,0].unsqueeze(1),batch_terminated),dim=1)[:,:-1]
                
            vdn_error = (vdn_mixed_qvals - vdn_targets.detach())*mask
            td_error = (chosen_action_qvals - targets.detach())*mask
            
            vdn_loss = (vdn_error ** 2).sum()/mask.sum()
            loss = (td_error ** 2).sum()/mask.sum()

            qvdn_optim.zero_grad()
            vdn_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(q_vdn.parameters, 10)
            qvdn_optim.step()

            qtrue_optim.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(qnet_true.parameters(), 10)
            qtrue_optim.step()

            if iters % 100 == 0 and iters > 0:
                print("-------------iters:",iters)
                print("epsl:",epsl, "loss:", loss.item(), "vdn_loss:", vdn_loss.item())
                print("test ret:", env.ret)
                    
                time.sleep(1)
            
            iters += 1
            if iters % target_update_interval == 0:
                target_qnet.load_state_dict(qnet_true.state_dict())
                target_q_vdn.update_target(q_vdn.mods)
        
        if iters % 1000 == 0 and iters > 0:
            print(ret_list)
    
    with open("./"+file_name, mode="a+") as f:
        f.write(str(ret_list))

run()