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
    

class Qnet_ind(nn.Module):
    def __init__(self):
        super(Qnet_ind,self).__init__()
        self.ob_shape = 10
        self.n_acs = 4 ** 3
        self.ind_net = nn.Sequential(
            nn.Linear(self.ob_shape,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,self.n_acs)
        )
    
    def forward(self,obs):
        return self.ind_net(obs)
               

class Qnet_vdn():
    def __init__(self,n_agents, world_size):
        self.n_agents = n_agents
        self.n_models = 2
        self.world_size = world_size
        self.mods = [Qnet_ind() for _ in range(self.n_models)]
        self.parameters = list(self.mods[0].parameters())
        for i in range(1,self.n_models):
            self.parameters += list(self.mods[i].parameters())
    
    def forward(self,obs):
        qtot = th.tensor([],dtype=th.float32)
        for i in range(self.n_models):
            ac = self.mods[i].forward(obs[i]).unsqueeze(0)
            qtot = th.cat((qtot,ac),dim=0)
        return qtot
    
    def select_acs(self, state, obs, epsl):
        q_local = self.forward(obs).unsqueeze(-1)
        q_local_1 = q_local[0].reshape(4,4,4)
        q_local_2 = q_local[1].reshape(4,4,4)
        qtot = th.zeros(4,4,16)

        for i in range(4):
            for j in range(4):
                qtot[:,:,4*i+j] = q_local_1[:,:,i] + q_local_2[:,:,j]

        q_mask = self._mask_from_state(state)
        for i in range(self.n_agents):
            q_mask[i][1] = 0
            q_mask[i][2] = 0
        
        qtot = qtot.unsqueeze(-1).reshape(4,4,4,4)
        qtot[q_mask[0]==0,:,:,:]=-9999
        qtot[:,q_mask[1]==0,:,:] = -9999
        qtot[:,:,q_mask[2]==0,:]=-9999
        qtot[:,:,:,q_mask[3]==0] = -9999

        qtot = qtot.reshape(1,-1)

        qtot_mask = th.ones_like(qtot)
        qtot_mask[qtot==-9999] = 0

        cc = np.random.rand(1)
        if cc > epsl:
            ac = qtot.max(dim=-1)[1]
        else:
            ac = th.randint(0, 4**self.n_agents,()).unsqueeze(0)
            while qtot[0,ac] == -9999:
                ac = th.randint(0, 4**self.n_agents,()).unsqueeze(0)
        
        acs = th.zeros(self.n_agents)
        tail = ac
        for i in range(self.n_agents):
            acs[i] = tail // (4 ** (self.n_agents-i-1))
            tail = tail % (4 ** (self.n_agents-i-1))

        return ac, acs,qtot_mask

    def update_target(self, target_mods):
        for i in range(self.n_models):
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
                    "obs": th.zeros(self.buffer_length, max_ep_length, 2, 10),
                    "terminated": th.ones(self.buffer_length, max_ep_length, 1),
                    "reward": th.zeros(self.buffer_length, max_ep_length, 1),
                    "ac": th.zeros(self.buffer_length, max_ep_length, 1, dtype=th.long),
                    "acs": th.zeros(self.buffer_length, max_ep_length, self.n_agents),
                    "qtot_mask": th.zeros(self.buffer_length, max_ep_length, 4**self.n_agents),
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
    agent_groups = [[0,1,2],[0,1,3]]
    target_landmarks = np.array([1,1,2,2])
    ret_list = [[],[]]
    sample_ret_list = []
    th.set_printoptions(profile="short")
    target_update_interval = 50
    n_agents = 4
    epsl = 1.0
    lr = 0.001
    max_ep_length = 6
    training_iters = 6001
    batch_size = 64
    iters = 0    
    decomposable = True                    # ########
    rl_manner = "qlearning"                     # "sarsa" or "qlearning"
    evaluate_algo = "vdn"                   # qcrc or vdn
    
    file_name = "vdn3+3_qqq" 
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

    q_vdn = Qnet_vdn(n_agents, world_size)
    qvdn_optim = RMSprop(params=q_vdn.parameters,lr=lr)
    
    while iters < training_iters:
        state = env.reset().reshape(1,-1).squeeze(0)
        terminated = False
        episode_data = {"state": th.tensor([]),
                    "obs": th.tensor([]),
                    "terminated": th.tensor([]),
                    "reward": th.tensor([]),
                    "ac": th.tensor([],dtype=th.long),
                    "acs": th.tensor([]),
                    "qtot_mask": th.tensor([])
            }

        t = 0
        while t < max_ep_length and (not terminated):
            state = th.from_numpy(state)
            obs = th.tensor([])
            for group in agent_groups:
                obs_group = th.tensor([])
                for a in group:
                    obs_group = th.cat((obs_group,state[2*a:2*a+2].float()),dim=0)
                obs = th.cat((obs,obs_group.unsqueeze(0)),dim=0)
            state = th.cat((state,th.from_numpy(target_landmarks)),dim=0).unsqueeze(0)
            obs = th.cat((obs,th.from_numpy(target_landmarks).repeat(len(agent_groups),1).float()),dim=-1)
            if iters % 100 == 0 and iters > 0:    # test mode
                if iters % 1000 == 0:
                    render = True
                else:
                    render = False
                ac,acs,qtot_mask = q_vdn.select_acs(state, obs, 0)                
                next_state, reward, terminated = env.step(ac, render=render)

            else:
                ac,acs,qtot_mask = q_vdn.select_acs(state, obs, 1)
                next_state, reward, terminated = env.step(ac)
            
            episode_data["state"] = th.cat((episode_data["state"], state.float()), dim=0)
            episode_data["reward"] = th.cat((episode_data["reward"], th.tensor([reward]).float().unsqueeze(0)), dim=0)
            episode_data["ac"] = th.cat((episode_data["ac"], ac.unsqueeze(0)), dim=0)
            episode_data["acs"] = th.cat((episode_data["acs"], acs.float().unsqueeze(0)), dim=0)
            episode_data["obs"] = th.cat((episode_data["obs"], obs.unsqueeze(0)), dim=0)
            episode_data["qtot_mask"] = th.cat((episode_data["qtot_mask"], qtot_mask), dim=0)

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
            batch_terminated = data["terminated"]
            batch_state = data["state"]
            batch_acs = data["acs"]
            batch_qmask = data["qtot_mask"]

            batch_obs = th.cat((data["obs"][:,:,0].unsqueeze(0),data["obs"][:,:,1].unsqueeze(0)),dim=0)
            vdn_qs = q_vdn.forward(batch_obs)
            ac_1 = batch_acs[:,:,0]*4**2 + batch_acs[:,:,1]*4 + batch_acs[:,:,2]
            ac_2 = batch_acs[:,:,0]*4**2 + batch_acs[:,:,1]*4 + batch_acs[:,:,3]
            batch_ac = th.cat((ac_1.unsqueeze(0),ac_2.unsqueeze(0)),dim=0).unsqueeze(-1)

            vdn_chosen_qvals = th.gather(vdn_qs, dim=-1, index=batch_ac.long())
            vdn_mixed_qvals = vdn_chosen_qvals.sum(dim=0)
        
            b = th.zeros_like(vdn_mixed_qvals)           
            vdn_mixed_qvals_extend = th.cat((vdn_mixed_qvals,b[:,1].unsqueeze(1)),dim=1)
            mask = 1- th.cat((th.zeros_like(batch_terminated)[:,0].unsqueeze(1),batch_terminated),dim=1)[:,:-1]

            if rl_manner == "sarsa":
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * vdn_mixed_qvals_extend[:,1:]
            else:
                q_local_1 = vdn_qs[0].reshape(batch_size,max_ep_length,4,4,4)
                q_local_2 = vdn_qs[1].reshape(batch_size,max_ep_length,4,4,4)
                qtot = th.zeros(batch_size,max_ep_length,4,4,16)

                for i in range(4):
                    for j in range(4):
                        qtot[:,:,:,:,4*i+j] = q_local_1[:,:,:,:,i] + q_local_2[:,:,:,:,j]
                
                qtot = qtot.reshape(batch_size,max_ep_length,4**n_agents)
                qtot[batch_qmask==0] = -9999

                curr_mixed_maxq = qtot.max(dim=-1)[0].unsqueeze(-1)
                curr_mixed_qvals_extend = th.cat((curr_mixed_maxq,b[:,1].unsqueeze(1)),dim=1)
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * curr_mixed_qvals_extend[:,1:]
                
            vdn_error = (vdn_mixed_qvals - vdn_targets.detach())*mask
            vdn_loss = (vdn_error ** 2).sum()/mask.sum()

            qvdn_optim.zero_grad()
            vdn_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(q_vdn.parameters, 10)
            qvdn_optim.step()

            if iters % 100 == 0 and iters > 0:
                print("-------------iters:",iters)
                print("epsl:",epsl, "vdn_loss:", vdn_loss.item())
                print("test ret:", env.ret)
                    
                time.sleep(1)
            
            iters += 1

        
        if iters % 1000 == 0 and iters > 0:
            print(ret_list)
    
    with open("./"+file_name, mode="a+") as f:
        f.write(str(ret_list))

run()