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
        self.tars = np.array([1,0])
        self.reset()
        self.ret = 0

    def reset(self):
        self.ret = 0
        self.world[1,1] = 3
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
        self.world[1,1] = 2
        self.world[2,2] = 2
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
        self.ob_shape = 6
        self.n_acs = 4
        self.ind_net = nn.Sequential(
            nn.Linear(self.ob_shape,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,self.n_acs)
        )
    
    def forward(self,obs):
        return self.ind_net(obs)
               

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
        ac = acs[0] * 4 + acs[1]

        q_mask_tot = th.ones(4,4)
        if obs[0][0] == 0:
            q_mask_tot[0,:] = 0
        if obs[1][0] == 0:
            q_mask_tot[:,0] = 0
        if obs[0][1] == self.world_size - 1:
            q_mask_tot[3,:] = 0
        if obs[1][1] == self.world_size - 1:
            q_mask_tot[:,3] = 0

        q_mask_tot[:,1] = 0
        q_mask_tot[:,2] = 0
        q_mask_tot[1,:] = 0
        q_mask_tot[2,:] = 0
        
        return ac.long().unsqueeze(-1), acs, q_mask, q_mask_tot.reshape(1,-1)

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
                    "acs": th.zeros(self.buffer_length, max_ep_length, self.n_agents),
                    "q_mask_tot": th.zeros(self.buffer_length, max_ep_length, 4**self.n_agents),
                    "q_mask": th.zeros(self.buffer_length, max_ep_length, self.n_agents,4)
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
    tar = th.tensor([[1,1],[2,2]]).float()
    n_agents = 2
    world_size = 4
    ac_perm = th.zeros(4,4).float()
    ac_perm[0][1]=1
    ac_perm[0][3]=1
    ac_perm[1][0]=-1
    ac_perm[1][2]=-1
    ac_perm[2][1]=1
    ac_perm[2][2]=-1
    ac_perm[3][0]=-1
    ac_perm[3][3]=1
    state_perm = th.tensor([])
    s0 = th.tensor([3,0,3,0]).unsqueeze(0).float()
    state_perm = th.cat((state_perm,s0),dim=0)
    s1 = s0.repeat(4,1)
    s1 += ac_perm
    state_perm = th.cat((state_perm,s1),dim=0)
    s2 = s1.repeat(4,1)
    for i in range(4):
        s2[4*i:4*i+4] += ac_perm[i]
    s2 = th.unique(s2,dim=0)
    state_perm = th.cat((state_perm,s2),dim=0)

    # s3 = s2.repeat(4,1)
    # for i in range(4):
    #     s3[9*i:9*i+9] += ac_perm[i]
    # s3 = th.unique(s3,dim=0)
    # state_perm = th.cat((state_perm,s3),dim=0)

    # s4_1 = th.tensor([[0,1,0,1],[0,1,1,1],[0,1,2,2],[0,1,2,3]]).float()
    # s4_2 = th.tensor([[2,3,0,1],[2,3,1,1],[2,3,2,2],[2,3,2,3]]).float()
    # s4_3 = th.tensor([[1,1,0,1],[1,1,1,2],[1,1,2,3],[2,2,0,1],[2,2,1,2],[2,2,2,3]]).float()
    # state_perm = th.cat((state_perm,s4_1,s4_2,s4_3),dim=0)
    
    # s5 = th.tensor([[0,2,0,2],[1,3,1,3],[0,2,1,3],[1,3,0,2]]).float().repeat(5,1)
    # for i in range(4):
    #     s5[0:4][i][0:2] = tar[0]
    #     s5[4:8][i][0:2] = tar[1]
    #     s5[8:12][i][2:4] = tar[0]
    #     s5[12:16][i][2:4] = tar[1]
    # s5 = th.unique(s5,dim=0)
    # state_perm = th.cat((state_perm,s5),dim=0)
   
    # s6 = th.tensor([0,3,0,3]).unsqueeze(0).float().repeat(5,1)
    # s6[1][0:2] = tar[0]
    # s6[2][0:2] = tar[1]
    # s6[3][2:4] = tar[0]
    # s6[4][2:4] = tar[1]
    # state_perm = th.cat((state_perm,s6),dim=0)
    poses_perm = state_perm.unsqueeze(1).reshape(-1,n_agents,2)
    obs_perm = th.cat((poses_perm,tar.reshape(1,4).unsqueeze(0).repeat(state_perm.shape[0],n_agents,1)),dim=-1)
    obs_perm = th.cat((obs_perm[:,0].unsqueeze(0),obs_perm[:,1].unsqueeze(0)),dim=0)

    state_perm = th.cat((state_perm,tar.reshape(1,4).repeat(state_perm.shape[0],1)),dim=-1)

    # mask
    q_mask_perm = th.ones(state_perm.shape[0],4,4)
    for i in range(poses_perm.shape[0]):
        if poses_perm[i][0][0] == 0:
            q_mask_perm[i,0,:] = 0
        if poses_perm[i][1][0] == 0:
            q_mask_perm[i,:,0] = 0
        if poses_perm[i][0][1] == world_size - 1:
            q_mask_perm[i,3,:] = 0
        if poses_perm[i][1][1] == world_size - 1:
            q_mask_perm[i,:,3] = 0

    q_mask_perm[:,:,1] = 0
    q_mask_perm[:,:,2] = 0
    q_mask_perm[:,1,:] = 0
    q_mask_perm[:,2,:] = 0
    q_mask_perm = q_mask_perm.reshape(-1,16)

    target_landmarks = np.array([1,1,2,2])
    ret_list = [[],[]]
    sample_ret_list = []
    th.set_printoptions(profile="short")
    epsl = 1.0
    lr = 0.001
    max_ep_length = 6
    training_iters = 50001
    batch_size = 64
    iters = 0    
    decomposable = True                    # ########
    rl_manner = "qlearning"                     # "sarsa" or "qlearning"
    evaluate_algo = "vdn"                   # qcrc or vdn
    
    file_name = "error" 
    if decomposable:
        file_name += "_de_"
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
    
    qnet_true = Qnet_CRC(n_agents, world_size)    
    qtrue_optim = RMSprop(params=qnet_true.parameters(),lr=lr)

    while iters < training_iters:
        state = env.reset().reshape(1,-1).squeeze(0)
        terminated = False
        episode_data = {"state": th.tensor([]),
                    "obs": th.tensor([]),
                    "terminated": th.tensor([]),
                    "reward": th.tensor([]),
                    "ac": th.tensor([],dtype=th.long),
                    "acs": th.tensor([]),
                    "q_mask": th.tensor([]),
                    "q_mask_tot": th.tensor([])
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
                ac,acs,q_mask, q_mask_tot = q_vdn.select_acs(state, obs, 0)                
                next_state, reward, terminated = env.step(ac, render=False)

            else:
                ac,acs,q_mask, q_mask_tot = q_vdn.select_acs(state, obs, 1)
                next_state, reward, terminated = env.step(ac)
            
            episode_data["state"] = th.cat((episode_data["state"], state.float()), dim=0)
            episode_data["reward"] = th.cat((episode_data["reward"], th.tensor([reward]).float().unsqueeze(0)), dim=0)
            episode_data["ac"] = th.cat((episode_data["ac"], ac.unsqueeze(0)), dim=0)
            episode_data["acs"] = th.cat((episode_data["acs"], acs.float().unsqueeze(0)), dim=0)
            episode_data["obs"] = th.cat((episode_data["obs"], obs.unsqueeze(0)), dim=0)
            episode_data["q_mask_tot"] = th.cat((episode_data["q_mask_tot"], q_mask_tot), dim=0)
            episode_data["q_mask"] = th.cat((episode_data["q_mask"], q_mask.unsqueeze(0)), dim=0)
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
            batch_qmask = data["q_mask"]
            batch_qmask_tot = data["q_mask_tot"]

            batch_obs = th.cat((data["obs"][:,:,0].unsqueeze(0),data["obs"][:,:,1].unsqueeze(0)),dim=0)
            vdn_qs = q_vdn.forward(batch_obs)
            vdn_qs[batch_qmask==0] = -999/2
            vdn_chosen_qvals = th.gather(vdn_qs, dim=-1, index=batch_acs.long())
            vdn_mixed_qvals = vdn_chosen_qvals.sum(dim=-2)

            b = th.zeros_like(vdn_mixed_qvals)
            vdn_mixed_qvals_extend = th.cat((vdn_mixed_qvals,b[:,1].unsqueeze(1)),dim=1)

            true_q_outs = qnet_true.forward(batch_state).reshape(batch_size,-1,4**n_agents)
            true_q_outs[batch_qmask_tot==0] = -999
            chosen_action_qvals = th.gather(true_q_outs, dim=-1, index=batch_ac)
            chosen_action_qvals_extend = th.cat((chosen_action_qvals,b[:,1].unsqueeze(1)),dim=1)

            if rl_manner == "sarsa":
                targets = batch_reward + 0.99 * (1 - batch_terminated) * chosen_action_qvals_extend[:,1:]
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * vdn_mixed_qvals_extend[:,1:]
                mask = 1- th.cat((th.zeros_like(batch_terminated)[:,0].unsqueeze(1),batch_terminated),dim=1)[:,:-1]
            
            elif rl_manner == "qlearning":
                max_q = true_q_outs.max(dim=-1)[0].unsqueeze(-1)
                max_q_extend = th.cat((max_q,b[:,1].unsqueeze(1)),dim=1)
                targets = batch_reward + 0.99 * (1 - batch_terminated) * max_q_extend[:,1:]

                vdn_max_qs = vdn_qs.max(dim=-1)[0]
                vdn_mixed_maxq = vdn_max_qs.sum(dim=-1).unsqueeze(-1)
                vdn_mixed_maxq_extend = th.cat((vdn_mixed_maxq,b[:,1].unsqueeze(1)),dim=1)
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * vdn_mixed_maxq_extend[:,1:]
                
                mask = 1- th.cat((th.zeros_like(batch_terminated)[:,0].unsqueeze(1),batch_terminated),dim=1)[:,:-1]
                
            vdn_error = (vdn_mixed_qvals - vdn_targets.detach())*mask
            vdn_loss = (vdn_error ** 2).sum()/mask.sum()

            td_error = (chosen_action_qvals - targets.detach())*mask
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
                q_true_perm = qnet_true.forward(state_perm)
                q_perm_vdn = q_vdn.forward(obs_perm)

                qtot_perm_vdn = q_perm_vdn[:,1].repeat(1,4)
                qtot_perm_vdn[:,0:4]+=q_perm_vdn[:,0,0].unsqueeze(-1)
                qtot_perm_vdn[:,4:8]+=q_perm_vdn[:,0,1].unsqueeze(-1)
                qtot_perm_vdn[:,8:12]+=q_perm_vdn[:,0,2].unsqueeze(-1)
                qtot_perm_vdn[:,12:16]+=q_perm_vdn[:,0,3].unsqueeze(-1)
                
                if rl_manner == "sarsa":
                    q_true_perm = q_true_perm*q_mask_perm
                    qtot_perm_vdn = qtot_perm_vdn*q_mask_perm

                    vdn_vperm = qtot_perm_vdn.sum(dim=-1)/q_mask_perm.sum(dim=-1)
                    vtrue_perm = q_true_perm.sum(dim=-1)/q_mask_perm.sum(dim=-1)

                    q_true = th.sqrt((q_true_perm ** 2).sum(dim=-1)/q_mask_perm.sum(dim=-1))
                    q_error = th.sqrt(((q_true_perm-qtot_perm_vdn)**2).sum(dim=-1)/q_mask_perm.sum(dim=-1))

                    print("q_true", q_true)
                    print("vtrue-vdn", q_error)
                else:
                    q_true_perm = q_true_perm*q_mask_perm
                    qtot_perm_vdn = qtot_perm_vdn*q_mask_perm
                    q_true = th.sqrt((q_true_perm ** 2).sum(dim=-1)/q_mask_perm.sum(dim=-1))
                    q_error = th.sqrt(((q_true_perm-qtot_perm_vdn)**2).sum(dim=-1)/q_mask_perm.sum(dim=-1))

                    q_true_perm[q_mask_perm==0] = -999
                    qtot_perm_vdn[q_mask_perm==0] = -999
                    vdn_vperm = qtot_perm_vdn.max(dim=-1)[0]
                    vtrue_perm = q_true_perm.max(dim=-1)[0]

                    print("qgreedy_true", q_true)
                    print("qgreedy-vdn", q_error)

                print("-------------iters:",iters)
                print("epsl:",epsl, "loss:", loss.item(), "vdn_loss:", vdn_loss.item())
                print("test ret:", env.ret)
                    
                time.sleep(1)
            
            iters += 1
        
        if iters % 1000 == 0 and iters > 0:
            print(ret_list)
    
    with open("./"+file_name, mode="a+") as f:
        f.write(str(ret_list))

run()