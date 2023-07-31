#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
import gym




import argparse
from datetime import datetime
import os
import random
from collections import deque




tf.keras.backend.set_floatx("float64")



parser = argparse.ArgumentParser(prog="DoubleDQN")
parser.add_argument("--env",default="CartPole-v1")
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--batch_size",type=int,default=256)
parser.add_argument("--gamma",type=float,default=0.95)
parser.add_argument("--eps",type=float,default=1)
parser.add_argument("--eps_decay",type=float,default=0.995)
parser.add_argument("--eps_min",type=float,default=0.01)
parser.add_argument("--logdir",default="logs")



args = parser.parse_args([])
logdir = os.path.join(args.logdir,parser.prog,args.env,datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"saving log to {logdir}")
#writer = tf.summary.create_file_writer(logdir)


class ReplayBuffer:
    def __init__(self,capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def store(self,state,action,reward,next_state,done):
        self.buffer.append([state,action,reward,next_state,done])
    
    
    def sample(self):
        sample = random.sample(self.buffer,args.batch_size)
        states,actions,rewards,next_states,done = map(np.asarray,zip(*sample))
        states = np.array(states).reshape(args.batch_size,-1)
        next_states = np.array(next_states).reshape(args.batch_size,-1)
        return states,actions,rewards,next_states,done

    def size(self):
        return len(self.buffer)


class DQN:
    def __init__(self,state_dim,action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = args.eps
        self.model = self.nn_model()
    
    def nn_model(self):
        model = tf.keras.Sequential(
            [keras.layers.Input((self.state_dim,)),
             keras.layers.Dense(32,activation="relu"),
             keras.layers.Dense(16,activation="relu"),
             keras.layers.Dense(self.action_dim),
             ]
        )
        model.compile(loss = 'mse',optimizer=tf.keras.optimizers.Adam(args.lr))
        return model

    def predict(self,state):
        return self.model.predict(state)
    
    def get_action(self,state):
        state = np.reshape(state,[1,self.state_dim])
        self.eps*=args.eps_decay
        self.eps = max(self.eps,args.eps_min)
        if(random.random()<self.eps):
            action = random.randint(0,self.action_dim-1)
        else:

            actions = self.predict(state)
            actions = actions[0]
            action = np.argmax(actions)
        return action
    
    def train(self,states,target_vals):
        self.model.fit(x=states,y=target_vals,epochs=1)


class Agent:
    def __init__(self,env:gym.Env):
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.model = DQN(self.state_dim,self.action_dim)
        self.targetModel = DQN(self.state_dim,self.action_dim)
        self.env = env
        self.update_weights()
        self.buffer = ReplayBuffer()
    
    def update_weights(self):
        weights = self.model.model.get_weights()
        self.targetModel.model.set_weights(weights=weights)
    
    def replayExperience(self):
        for i in range(10):
            states,actions,rewards,next_states,done = self.buffer.sample()
            targets = self.model.predict(states)
            next_q_vals = self.targetModel.predict(next_states).max(axis=1)
            targets[range(args.batch_size),actions] = rewards+(1-done)*next_q_vals*args.gamma
            self.model.train(states,targets)
    
    def train(self,env:gym.Env,max_episodes):

        if(os.path.isfile("DoubleDQN_weights.h5")):
            self.model.model.load_weights("DoubleDQN_weights.h5")
            print("loaded previous training weights")
        max_avg = 250
        reward_queue =  deque(maxlen=20)
        self.update_weights()
        for ep in range(max_episodes):
            total_reward = 0
            done = False
            state= env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state,reward,done,_ = env.step(action)
                self.buffer.store(state,action,reward*0.01,next_state,done)
                total_reward+=reward
                state = next_state
            
            if(self.buffer.size()>=args.batch_size):
                self.replayExperience()
                print("doing experience replay")
            self.update_weights()

            reward_queue.append(total_reward)

            if(len(reward_queue)==20 and sum(reward_queue)/20>=max_avg):
                max_avg = sum(reward_queue)/20
                print("saving weights",max_avg)

                self.model.model.save_weights("DoubleDQN_weights.h5")

            print(f"episode {ep} total_reward={total_reward} avg {sum(reward_queue)/20},{max_avg}")
            #tf.summary.scalar("episode reward",total_reward,step=ep)
            #writer.flush()
    
    def test(self,env:gym.Env,max_episodes):

        if(os.path.isfile("DoubleDQN_weights.h5")):
            self.model.model.load_weights("DoubleDQN_weights.h5")
            print("loaded previous training weights")
        max_avg = 250
        reward_queue =  deque(maxlen=20)
        self.update_weights()
        for ep in range(max_episodes):
            total_reward = 0
            done = False
            state= env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state,reward,done,_ = env.step(action)
                self.buffer.store(state,action,reward*0.01,next_state,done)
                total_reward+=reward
                state = next_state
                env.render()

            reward_queue.append(total_reward)

            
            print(f"episode {ep} total_reward={total_reward} avg {sum(reward_queue)/20},{max_avg}")
            #tf.summary.scalar("episode reward",total_reward,step=ep)
            #writer.flush()




if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    #agent.train(env,10000)
    agent.test(env,1000)






 