import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with less than 20 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair.

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        probs = []

        for state, action in zip(states, actions):
            player_sum = state[0]
            
            if (player_sum >= 20 and action == 0) or (player_sum < 20 and action == 1):
                probs.append(1.0)
            else:
                probs.append(0.0)

        return np.array(probs)

    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        player_sum = state[0]

        if player_sum >= 20:
            action = 0
        else:
            action = 1

        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length.
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    # YOUR CODE HERE

    state = env.reset()
    done = False

    while not done:
        action = policy.sample_action(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
        
    return states, actions, rewards, dones

def mc_prediction(policy, env, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A policy which allows us to sample actions with its sample_action method.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)

    # YOUR CODE HERE
    for i in tqdm(range(num_episodes)):
        states, _, rewards, _ = sampling_function(env, policy)
        value = 0
        
        for j in reversed(range(len(states))):
            value = discount_factor * value + rewards[j]
            
            if states[j] not in states[:j]:
                returns_count[states[j]] += 1
                V[states[j]] += value
    
    V = {k: v / returns_count[k] for k, v in V.items()}

    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains
        a probability of perfoming action in given state for every corresponding state action pair.

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """

        # YOUR CODE HERE
        probs = []

        for state, action in zip(states, actions):
            probs.append(0.5) 

        return np.array(probs)

    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            state: current state

        Returns:
            An action (int).
        """

        # YOUR CODE HERE
        action = np.random.choice([0, 1])

        return action

def mc_importance_sampling(behavior_policy, target_policy, env, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.

    Args:
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    V = defaultdict(float)
    returns_count = defaultdict(float)

    for i in tqdm(range(num_episodes)):
        states, actions, rewards, _ = sampling_function(env, behavior_policy)

        G = 0
        W = 1
        C = 1

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            G = discount_factor * G + rewards[t]

            if C > 0:
                V[state] += (W / C) * (G - V[state])
                returns_count[state] += W / C

            W *= target_policy.get_probs([state], [actions[t]]) / behavior_policy.get_probs([state], [actions[t]])

            C += 1

            if W == 0:
                break

    for state in V:
        V[state] /= returns_count[state]

    return V