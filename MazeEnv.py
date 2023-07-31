#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np


# In[1]:


class MazeEnv(gym.Env):
    def __init__(self,stochastic=True):
        self.map=np.array(["SWFWG","OOOOO","WOOOW","FOWFW"])
        self.dim=(4,5)
        self.distinct_states = 112 
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(4)
        self.img_map = np.ones(self.dim)
        self.obstacles=[(0,1),(0,3),(2,0),(2,4),(3,2),(3,4)]
        for x in self.obstacles:
            self.img_map[x[0]][x[1]] = 0
        self.slip_action={0:3,1:2,2:0,3:1}
        self.index_to_coordinate_map={0:(0,0),1:(1,0),2:(3,0),3:(1,1),4:(2,1),5:(3,1),6:(0,2),7:(1,2),8:(2,2),9:(1,3), 
                                        10:(2,3),11:(3,3),12:(0,4),13:(1,4)}
        self.coordinate_to_index_map=dict((val,key) for key,val in self.index_to_coordinate_map.items())
        self.goal_pos=(0,4)
        self.action_space = gym.spaces.Discrete(4)
        self.slip_probability = 0.1

    def num2Coin(self,n:int):
        coinList = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
        return coinList[n]
    
    def coin2num(self,v:list):
        if(sum(v)<2):
            return np.inner(v,[1,2,3])
        else:
            return np.inner(v,[1,2,3])+1
    
    def set_state(self,state:int)->None:
        self.state = state

    def step(self,action,slip=True):
        
        self.slip = True
        if self.slip:
            if np.random.rand() < self.slip_probability:
                action = self.slip_action[action]
        cell = self.index_to_coordinate_map[int(self.state/8)]
        if action ==0:
            c_next = cell[1]
            r_next = max(0,cell[0]-1)
        elif action == 1:
            c_next = cell[1]
            r_next = min(self.dim[0]-1,cell[0]+1)
        elif action ==2:
            c_next = max(0,cell[1]-1)
            r_next = cell[0]
        elif action==3:
            c_next = min(self.dim[1]-1,cell[1]+1)
            r_next = cell[0]
        else:
            raise ValueError(f"Invalid action:{action}")
        

        if(r_next==self.goal_pos[0] and c_next==self.goal_pos[1]):
            #print(self.goal_pos," reached")
            self.state = 8*self.coordinate_to_index_map[(r_next,c_next)]+self.state % 8
            reward = float(sum(self.num2Coin(self.state%8)))
            done = True
            return (self.state,reward,done)

        else:
            if (r_next,c_next) in self.obstacles:
                return (self.state,0,False)
            
            else:
                vcoin = self.num2Coin(self.state%8)
                if (r_next,c_next) == (0,2):
                    vcoin[0] = 1
                elif (r_next,c_next)==(3,0):
                    vcoin[1] = 1
                elif (r_next,c_next) ==(3,3):
                    vcoin[2] = 1
                
                self.state = 8*self.coordinate_to_index_map[(r_next,c_next)]+self.coin2num(vcoin)
                return (self.state,0,False)
    
    def render(self):
        cell = self.index_to_coordinate_map[int(self.state / 8)]
        desc = self.map.tolist()

        desc[cell[0]] = (
            desc[cell[0]][: cell[1]]
            + "\x1b[1;34m"  # Blue font
            + "\x1b[4m"  # Underline
            + "\x1b[1m"  # Bold
            + "\x1b[7m"  # Reversed
            + desc[cell[0]][cell[1]]
            + "\x1b[0m"
            + desc[cell[0]][cell[1] + 1 :]
        )

        print("\n".join("".join(row) for row in desc))

    def reset(self):
        self.state = 0


if __name__ == "__main__":
    env = MazeEnv()
    obs = env.reset()
    env.render()
    done = False
    step_num=1
    action_list=["UP","DOWN","LEFT","RIGHT"]

    while not done:
        action = env.action_space.sample()
        next_obs,reward,done = env.step(action)
        print(f"step # {step_num} action = {action_list[action]} reward= {reward} done:{done}")
        step_num+=1
        env.render()
    env.close()




        

