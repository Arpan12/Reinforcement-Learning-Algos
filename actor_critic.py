import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import gym
import os.path


class Actor(keras.Model):
    def __init__(self,action_dim):
        super(Actor,self).__init__()
        self.layer1 = layers.Dense(512,activation="relu")
        self.layer2 = layers.Dense(128,activation="relu")
        self.layer3 = layers.Dense(action_dim,activation=None)
    
    def call(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    

class Critic(keras.Model):
    def __init__(self,action_dim):
        super(Critic,self).__init__()
        self.layer1 = layers.Dense(512,activation="relu")
        self.layer2 = layers.Dense(128,activation="relu")
        self.layer3 = layers.Dense(1,activation=None)
    
    def call(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Agent():
    def __init__(self,env,action_dim):
        self.actor = Actor(action_dim=action_dim)
        self.critic = Critic(action_dim=action_dim)
        state = env.reset()
        self.actor.build(np.array([state]).shape)
        self.critic.build(np.array([state]).shape)
        self.gamma = 0.99
        self.opt = tf.optimizers.Adam(learning_rate=1e-4)
    
    def getAction(self,state):
        action_dist = self.actor(np.array([state]))
        action_dist = tf.nn.softmax(action_dist)
        action_dist = action_dist.numpy()
        dist = tfp.distributions.Categorical(probs = action_dist,dtype = tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def getActorLoss(self,prob,action,td):
        action_dist = tf.nn.softmax(prob)
        dist = tfp.distributions.Categorical(probs=action_dist,dtype=tf.float32)
        loss = dist.log_prob(action)
        loss = -loss*td
        return loss
    
    def getCriticLoss(self,td):
        return td**2
    

    def learn(self,state,action,reward,next_state):
        
        with tf.GradientTape(persistent=True) as tape:
            next_val = self.critic(np.array([next_state]))
            val = self.critic(np.array([state]))
            td = reward + self.gamma*next_val - val
            criticLoss = self.getCriticLoss(td)
            prob = self.actor(np.array([state]))
            actorLoss  = self.getActorLoss(prob,action,td)
        actor_grad = tape.gradient(actorLoss,self.actor.trainable_variables)
        self.opt.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        critic_grad = tape.gradient(criticLoss,self.critic.trainable_variables)
        self.opt.apply_gradients(zip(critic_grad,self.critic.trainable_variables))


    def train(self,env:gym.Env,episodes:int,render:bool):
        avg_reward=0
        max_reward=0
        queue=[]
        
        if os.path.isfile("MountainCar_actor.h5"):
            self.actor.load_weights("MountainCar_actor.h5")
        if os.path.isfile("MountainCar_critic.h5"):
            self.critic.load_weights("MountainCar_critic.h5")

        for ep in range(episodes):
            state = env.reset()
            total_reward=0
            done = False
            while not done:
                action = self.getAction(state)
                next_state,reward,done,_ = env.step(action)
                self.learn(state,action,reward,next_state)
                state = next_state
                total_reward+=reward

                if(render):
                    env.render()

                if done:
                    queue.append(total_reward)
                    if(len(queue)>10):
                        queue.pop(0)
                
                    if(len(queue)==10):
                        avg_reward = sum(queue)/len(queue)
                        
                    
                    if(max_reward<avg_reward):
                        max_reward = avg_reward
                        self.actor.save_weights("MountainCar_actor.h5")
                        self.critic.save_weights("MountainCar_critic.h5")
                        print("saving weights")


                    print(f"episode {ep} total_reward {total_reward} avg_reward = {avg_reward}")
    
    def test(self,env:gym.Env,episodes:int,render:bool):
        avg_reward=0
        max_reward=0
        queue=[]
        
        if os.path.isfile("MountainCar_actor.h5"):
            self.actor.load_weights("MountainCar_actor.h5")
        if os.path.isfile("MountainCar_critic.h5"):
            self.critic.load_weights("MountainCar_critic.h5")

        for ep in range(episodes):
            state = env.reset()
            total_reward=0
            done = False
            while not done:
                action = self.getAction(state)
                next_state,reward,done,_ = env.step(action)
                print(next_state,reward)

                #self.learn(state,action,reward,next_state)
                state = next_state
                total_reward+=reward

                if(render):
                    env.render()
                if done:    
                    print(f"episode {ep} total_reward {total_reward} avg_reward = {avg_reward}")


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Agent(env,env.action_space.n)
    num_episodes = 10000
    #agent.train(env,num_episodes,False)
    agent.test(env,num_episodes,True)









    


    





        

