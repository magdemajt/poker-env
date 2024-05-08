from keras import Sequential
from keras.src.layers import Dense, Activation
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from torch.nn import Flatten
from torch.optim import Adam


def initialize_model(env):

    nb_actions = env.action_space.n

    model = Sequential()

    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn

def save_model(model, path):
    model.save_weights(path, overwrite=True)

def load_model(model, path):
    model.load_weights(path)
    return model

def train_model(model, env, nb_steps):
    model.fit(env, nb_steps=nb_steps, visualize=False, verbose=2)
    return model

def test_model(model, env, nb_episodes):
    model.test(env, nb_episodes=nb_episodes, visualize=False)
    return model