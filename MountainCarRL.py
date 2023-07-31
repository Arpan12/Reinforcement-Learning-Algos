import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym



class PolicyNet(keras.Model):
    def __init__(self,action_dim=1):
        super(PolicyNet,self).__init__()
        self.fc1 = layers.Dense(24,activation="relu")
        self.layer2 = layers.Dense(36,activation="relu")
        self.layer3 = layers.Dense(action_dim,activation="softmax")

    def call(self,x):
        x = self.fc1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    def process(self,observations):
        action_probs = self.predict_on_batch(observations)
        return action_probs
    

class Agent():
    def __init__(self,action_dim=1):
        self.policy_net = PolicyNet(action_dim)
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.gamma=0.99
    
    def policy(self,observation):
        observation = observation.reshape(1,-1)
        observation = tf.convert_to_tensor(observation,dtype = tf.float32)
        action_logits = self.policy_net(observation)
        action = tf.random.categorical(tf.math.log(action_logits),num_samples=1)
        return action

    def get_action(self,observation):
        action = self.policy(observation).numpy()
        return action.squeeze()
    

    def learn(self,states,actions,rewards):
        discounted_reward=0
        discounted_rewards=[]
        rewards.reverse()

        for reward in rewards:
            discounted_reward = reward + self.gamma*discounted_reward
            discounted_rewards.append(discounted_reward)
        
        discounted_rewards.reverse()

        for state,reward,action in zip(states,discounted_rewards,actions):
            with tf.GradientTape() as tape:
                action_probabilities = self.policy_net(np.array([state]),training=True)
                loss = self.loss(action_probabilities,action,reward)
            grads = tape.gradient(loss,self.policy_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.policy_net.trainable_variables))
            #print(grads)
    

    def loss(self,action_probabilites,action,reward):
        dist = tfp.distributions.Categorical(probs =action_probabilites,dtype=tf.float32)
        loss = -dist.log_prob(action)*reward
        return loss
    


def train(agent:Agent ,env:gym.Env,episodes:int,render=True):

    for ep in range(episodes):
        states = []
        rewards=[]
        actions=[]
        state = env.reset()
        done = False
        total_reward=0
        while not done:
            action = agent.get_action(state)
            print(action)
            actions.append(action)
            next_state,next_reward,done,_ = env.step(action)
            states.append(next_state)
            rewards.append(next_reward)
            total_reward+=next_reward
            state+=next_state

            if(render):
                env.render()
            
            if(done):
                agent.learn(states,actions,rewards)
                print("\n")
                print(f"episode {ep} ep_reward= {total_reward}")


if __name__ == "__main__":
    agent = Agent(3)
    episode = 600
    env = gym.make("MountainCar-v0");
    train(agent,env,episode,False)
    env.close()
        
        
        







