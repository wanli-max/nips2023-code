import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import time
import copy
import random

# EXPERIMENT 1： indecomposable, true Q visitation, vdn max//vdn sarsa//true Q
# expected: vdn sarsa = true q \neq vdn max
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
    def __init__(self, world_size, punish, decomposable):
        self.world_size = world_size
        self.n_agents = 2
        self.world = np.zeros((self.world_size,self.world_size))
        # assign landmark
        if decomposable:
            self.tars = np.array([1,2])
        else:
            self.tars = np.array([1,1])
        self.reset()
        self.punish = punish
        self.ret = 0
        self.generate_q_mask_perm()

    def reset(self):
        self.ret = 0
        self.world[0,0] = self.tars[0]
        self.world[0,-1] = self.tars[1]
        self.agent_active = np.array([1,1])
        posx = self.world_size-1
        posy = self.world_size//2
        self.poses = np.array([[posx,posy],[posx,posy]])
        state = np.array([posx,posy,self.agent_active[0],posx,posy,self.agent_active[1]])
        return self.poses
    
    def step(self,ac,render=False):
        reward = 0
        terminated = False
        acs = th.tensor([ac // 4, ac % 4])
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

                self.poses[i] = np.clip(self.poses[i],0,self.world_size-1)
                if self.world[self.poses[i][0],self.poses[i][1]] == self.tars[i]:
                    reward += 1
                    self.agent_active[i] = 0
                    self.world[self.poses[i][0],self.poses[i][1]] = -1
                if self.agent_active[i]:
                    if self.world[self.poses[i][0],self.poses[i][1]] == -1:
                        reward -= self.punish
        if self.world[0,0] == -1 and self.world[0,-1] == -1:
            # reward += 1
            terminated = True
        if render:
            time.sleep(0.2)
            self.render()
            if terminated:
                print("-----terminated")
                time.sleep(2)
        
        poses_id = (self.poses[0][0]*self.world_size+self.poses[0][1])*self.world_size**2+(self.poses[1][0]*self.world_size+self.poses[1][1])
        q_mask = self.q_mask_perm[poses_id]
        return self.poses, reward, terminated, q_mask
        
    def render(self):
        render_world = copy.deepcopy(self.world)
        render_world[self.poses[0][0],self.poses[0][1]] = 5
        render_world[self.poses[1][0],self.poses[1][1]] = 7
        print(render_world,"\n")

    def generate_q_mask_perm(self):
        self.q_mask_perm = th.tensor([],dtype=th.float32)
        self.indq_mask_perm = th.tensor([],dtype=th.float32)
        pose_perm = th.tensor([],dtype=th.long)
        for i in range(self.world_size**2):
            for j in range(self.world_size**2):
                pose_perm = th.cat((pose_perm,th.tensor([[i // self.world_size, i % self.world_size],[j//self.world_size,j%self.world_size]]).unsqueeze(0)),dim=0)
        
        for i in range(self.world_size**4):
            q_mask = th.ones(4,4)
            indq_mask = th.ones(2,4)
            if pose_perm[i][0][0] == 0:
                q_mask[0,:] = 0
                indq_mask[0][0] = 0
            elif pose_perm[i][0][0] == self.world_size-1:
                q_mask[1,:] = 0
                indq_mask[0][1] = 0

            if pose_perm[i][0][1] == 0:
                q_mask[2,:] = 0
                indq_mask[0][2] = 0
            elif pose_perm[i][0][1] == self.world_size-1:
                q_mask[3,:] = 0
                indq_mask[0][3] = 0
            
            if pose_perm[i][1][0] == 0:
                q_mask[:,0] = 0
                indq_mask[1][0] = 0
            elif pose_perm[i][1][0] == self.world_size-1:
                q_mask[:,1] = 0
                indq_mask[1][1] = 0

            if pose_perm[i][1][1] == 0:
                q_mask[:,2] = 0
                indq_mask[1][2] = 0
            elif pose_perm[i][1][1] == self.world_size-1:
                q_mask[:,3] = 0
                indq_mask[1][3] = 0
            self.q_mask_perm = th.cat((self.q_mask_perm,q_mask.unsqueeze(0)),dim=0)
            self.indq_mask_perm = th.cat((self.indq_mask_perm,indq_mask.unsqueeze(0)),dim=0)

    
class Qnet_CRC(nn.Module):
    def __init__(self,n_agents):
        super(Qnet_CRC,self).__init__()
        self.state_shape = 4                      # poses
        self.n_agents = n_agents
        self.n_acs = 4 ** self.n_agents          # 0 1 2 3
        self.net = nn.Sequential(
            nn.Linear(self.state_shape,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self.n_acs)
        )
    
    def forward(self,state):
        return self.net(state)
    
    def select_acs(self, state, epsl, q_mask):
        state = th.from_numpy(state).view(1,-1).float()
        q_mask = q_mask.view(1,-1)
        qcrc = self.forward(state)
        qcrc[q_mask==0] = -9999
        cc = np.random.rand(1)
        if cc > epsl:
            ac = qcrc.max(dim=-1)[1]
        else:
            ac = th.randint(0,4**self.n_agents,()).unsqueeze(0)
            while q_mask[0][ac] == 0:
                ac = th.randint(0,4**self.n_agents,()).unsqueeze(0)
        acs = th.tensor([ac // 4, ac % 4])
        return ac, acs
                
class Qnet_ind(nn.Module):
    def __init__(self):
        super(Qnet_ind,self).__init__()
        self.ob_shape = 2
        self.n_acs = 4
        self.ind_net = nn.Sequential(
            nn.Linear(self.ob_shape,8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,self.n_acs)
        )
    
    def forward(self,obs):
        return self.ind_net(obs)

class Qnet_vdn():
    def __init__(self,n_agents):
        self.n_agents = n_agents
        self.mods = [Qnet_ind() for _ in range(self.n_agents)]
        self.parameters = list(self.mods[0].parameters())
        self.parameters += list(self.mods[1].parameters())
    
    def forward(self,obs):
        qtot = th.tensor([],dtype=th.float32)
        for i in range(self.n_agents):
            ac = self.mods[i].forward(obs[i]).unsqueeze(-2)
            qtot = th.cat((qtot,ac),dim=-2)
        return qtot
    
    def select_acs(self, obs, epsl, q_mask):
        q_mask = q_mask.view(1,-1)
        qtot = th.zeros(4,4)
        q_local = self.forward(obs)
        for i in range(4):
            for j in range(4):
                qtot[i][j] = q_local[0,i]+q_local[1,j]
        qtot = qtot.view(1,-1)
        qtot[q_mask==0] = -9999
        cc = np.random.rand(1)
        if cc > epsl:
            ac = qtot.max(dim=-1)[1]
        else:
            ac = th.randint(0,4**self.n_agents,()).unsqueeze(0)
            while q_mask[0][ac] == 0:
                ac = th.randint(0,4**self.n_agents,()).unsqueeze(0)
        acs = th.tensor([ac // 4, ac % 4])
        return ac, acs

    def update_target(self, target_mods):
        for i in range(self.n_agents):
            self.mods[i].load_state_dict(target_mods[i].state_dict())
    
    
class ReplayBuffer:
    def __init__(self, max_ep_length, buffer_length):
        self.buffer_length = buffer_length
        self.dict = {"state": th.zeros(self.buffer_length, max_ep_length, 4),
                    "obs": th.zeros(self.buffer_length, max_ep_length, 2, 2),
                    "terminated": th.ones(self.buffer_length, max_ep_length, 1),
                    "reward": th.zeros(self.buffer_length, max_ep_length, 1),
                    "ac": th.zeros(self.buffer_length, max_ep_length, 1, dtype=th.long),
                    "acs": th.zeros(self.buffer_length, max_ep_length, 2, dtype=th.long)
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
    world_size = 3
    ret_list = [[],[],[]]
    th.set_printoptions(profile="short")
    target_update_interval = 50
    n_agents = 2
    epsl = 1.0
    lr = 0.0001
    max_ep_length = 20
    training_iters = 100000
    batch_size = 64
    iters = 0    
    decomposable = True                    # ########
    rl_manner = "qlearning"                     # "sarsa" or "qlearning"
    
    evaluate_algo = "vdn"                    # qcrc or vdn
    
    file_name = rl_manner 
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
        buffer_length = batch_size
    buffer = ReplayBuffer(max_ep_length, buffer_length)
    
    if decomposable:
        punish = 0
    else:
        punish = 0.2

    env = Env(world_size,punish=punish,decomposable=decomposable)
    action_selector = Action_Selector(world_size,env.q_mask_perm)
    action_selector.prob_tabular[env.q_mask_perm.view(-1,1,16).squeeze(1)==0] = -999
    action_selector.prob_tabular = F.softmax(action_selector.prob_tabular)
    qnet_true = Qnet_CRC(n_agents)
    target_qnet = copy.deepcopy(qnet_true)
    q_vdn = Qnet_vdn(n_agents)
    target_q_vdn = copy.deepcopy(q_vdn)
    qtrue_optim = RMSprop(params=qnet_true.parameters(),lr=lr)
    qvdn_optim = RMSprop(params=q_vdn.parameters,lr=lr)
    
    while iters < training_iters:
        state = env.reset().reshape(1,-1).squeeze(0)
        q_mask = th.ones(4,4)
        q_mask[:,1] = 0
        q_mask[1,:] = 0
        terminated = False
        episode_data = {"state": th.tensor([]),
                    "obs": th.tensor([]),
                    "terminated": th.tensor([]),
                    "reward": th.tensor([]),
                    "ac": th.tensor([],dtype=th.long),
                    "acs": th.tensor([],dtype=th.long)
            }

        t = 0
        while not terminated:
            obs = th.tensor([state[0:2],state[2:4]]).float()
            if iters % 100 == 0 and iters > 0:    # test mode
                if iters % 1000 == 0:
                    render = True
                else:
                    render = False
                ac,acs = q_vdn.select_acs(obs, 0, q_mask)
                next_state, reward, terminated, q_mask = env.step(ac, render=render)
            elif iters % 101 == 0 and iters > 0:    # test mode
                ac,acs = qnet_true.select_acs(state, 0, q_mask)
                next_state, reward, terminated, q_mask = env.step(ac, render=render)
            else:
                # ac,acs = qnet_true.select_acs(state, epsl, q_mask)
                # ac,acs = q_vdn.select_acs(obs, epsl, q_mask)
                ac,acs = action_selector.select_acs(state, 0, q_mask)
                next_state, reward, terminated, q_mask = env.step(ac)
                       
            episode_data["state"] = th.cat((episode_data["state"], th.tensor([state]).float()), dim=0)
            episode_data["reward"] = th.cat((episode_data["reward"], th.tensor([reward]).float().unsqueeze(0)), dim=0)
            episode_data["ac"] = th.cat((episode_data["ac"], ac.unsqueeze(0)), dim=0)
            episode_data["acs"] = th.cat((episode_data["acs"], acs.unsqueeze(0)), dim=0)
            episode_data["obs"] = th.cat((episode_data["obs"], th.tensor([state]).reshape(2,2).unsqueeze(0).float()), dim=0)

            t += 1
            env.ret += reward
            state = next_state.reshape(1,-1).squeeze(0)
            obs = state.reshape(n_agents,-1)
            if t >= max_ep_length:
                terminated = True
            
            episode_data["terminated"] = th.cat((episode_data["terminated"], th.tensor([terminated]).float().unsqueeze(0)), dim=0)

        if iters % 100 == 0 and iters > 0:
            ret_list[0].append(iters)
            ret_list[1].append(env.ret)
        elif iters % 101 == 0 and iters > 0:
            ret_list[2].append(env.ret)
        else:   
            buffer.add_ep_traj(episode_data)

        if epsl >= 0.2:
            epsl -= 0.0005

        # train
        if buffer.trajs_in_buffer >= batch_size:
            data = buffer.sample(batch_size)
            batch_reward = data["reward"][:,:-1]
            batch_ac = data["ac"]
            batch_terminated = data["terminated"][:,:-1]
            batch_state = data["state"]
            batch_acs = data["acs"].unsqueeze(-1)
            batch_obs = th.cat((data["obs"][:,:,0].unsqueeze(0),data["obs"][:,:,1].unsqueeze(0)),dim=0)

            mask = th.ones_like(batch_terminated)
            mask[:, 1:] = 1 - batch_terminated[:,:-1]
        
            vdn_qs = q_vdn.forward(batch_obs)
            vdn_chosen_qvals = th.gather(vdn_qs, dim=-1, index=batch_acs)
            vdn_mixed_qvals = vdn_chosen_qvals.sum(dim=-2)

            true_q_outs = qnet_true.forward(batch_state).reshape(batch_size,-1,4**n_agents)
            chosen_action_qvals = th.gather(true_q_outs, dim=-1, index=batch_ac)
            
            if rl_manner == "qlearning":
                curr_max_acs = true_q_outs.max(dim=-1)[1].unsqueeze(-1)
                target_q_outs = target_qnet.forward(batch_state).reshape(batch_size,-1,4**n_agents)
                target_max_q = th.gather(target_q_outs, dim=-1, index=curr_max_acs)
                targets = batch_reward*0.2 + 0.99 * (1 - batch_terminated) * target_max_q[:,1:]

                curr_max_ac = vdn_qs.max(dim=-1)[1].unsqueeze(-1)
                target_vdn_qvals = target_q_vdn.forward(batch_obs)
                target_max_qvdn = th.gather(target_vdn_qvals, dim=-1, index=curr_max_ac)
                target_mixed_qvals = target_max_qvdn.sum(dim=-2)
                vdn_targets = batch_reward*0.2 + 0.99 * (1 - batch_terminated) * target_mixed_qvals[:,1:]

            elif rl_manner == "sarsa":
                targets = batch_reward + 0.99 * (1 - batch_terminated) * chosen_action_qvals[:,1:]
                vdn_targets = batch_reward + 0.99 * (1 - batch_terminated) * vdn_mixed_qvals[:,1:]

            vdn_error = (vdn_mixed_qvals[:,:-1] - vdn_targets.detach()) * mask
            td_error = (chosen_action_qvals[:,:-1] - targets.detach()) * mask
            
            vdn_loss = (vdn_error ** 2).sum()/ mask.sum()
            loss = (td_error ** 2).sum()/ mask.sum()

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
                
                pose_perm = th.tensor([])         
                for x in range(world_size):
                    for y in range(world_size):
                        pose_perm = th.cat((pose_perm,th.tensor([x,y]).unsqueeze(0).float()),dim=0)
                state_perm = th.tensor([])  
                for x in pose_perm:
                    for y in pose_perm:
                        xy = th.cat((x,y),dim=-1)              
                        state_perm = th.cat((state_perm,xy.unsqueeze(0)),dim=0)
                obs_perm = state_perm.reshape(-1,n_agents,2)
                obs_perm = th.cat((obs_perm[:,0].unsqueeze(0),obs_perm[:,1].unsqueeze(0)),dim=0)

                qperm = qnet_true(state_perm)
                q_mask_perm_cp = env.q_mask_perm.view(world_size**4,16)
                qperm[q_mask_perm_cp==0] = -999
                greedy_qperm = qperm.max(dim=-1)[0]
                vperm = qperm * action_selector.prob_tabular
                vperm = vperm.sum(dim=-1)

                qperm_vdn = th.ones_like(qperm)
                indv_qperm_vdn = q_vdn.forward(obs_perm)
                for x in range(4):
                    for y in range(4):
                        qperm_vdn[:,4*x+y] = indv_qperm_vdn[:,0,x] + indv_qperm_vdn[:,1,y]
                        
                qperm_vdn[q_mask_perm_cp==0] = -999
                greedy_qperm_vdn = qperm_vdn.max(dim=-1)[0]
                vperm_vdn = qperm_vdn * action_selector.prob_tabular
                vperm_vdn = vperm_vdn.sum(dim=-1)

                # q_error = th.abs(qperm - qperm_vdn).mean(dim=-1)
                if rl_manner == "qlearning":
                    print(greedy_qperm)
                    print(greedy_qperm-greedy_qperm_vdn)
                elif rl_manner == "sarsa":
                    print(vperm)
                    print(vperm_vdn-vperm)

                    
                time.sleep(2)
            
            iters += 1
            if iters % target_update_interval == 0:
                target_qnet.load_state_dict(qnet_true.state_dict())
                target_q_vdn.update_target(q_vdn.mods)
        
        if iters % 1000 == 0:
            print(ret_list)

    with open("./"+file_name, mode="a+") as f:
        f.write(str(ret_list))

run()