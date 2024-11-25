import numpy as np
from games import MatrixGame


# for NAHT experiments
def define_bit_game_payoff(n_agents):
    '''
    game description: at each turn, each agent picks one bit bi ∈ {0, 1}; 
    at the end of each turn, all the bits are summed. 
    the team wins if and only if the sum of the chosen bits is exactly 1.
    '''
    n_actions = 2
    ind_matrix_shape = [n_actions for _ in range(n_agents)]
    payoff = np.zeros(ind_matrix_shape)
    # only place in payoff matrix that should be 1 is where one agent selects 1 and 
    # all others select 0
    for i in range(n_agents):
        indice = [0 for _ in range(n_agents)]
        indice[i] = 1
        payoff[tuple(indice)] = 1
    return np.array([payoff for _ in range(n_agents)])

_payoff_bit_game = define_bit_game_payoff(n_agents=3)
game = MatrixGame(_payoff_bit_game, ep_length=1000, last_action_state=True,
                  action_onehot_repr=True)
# note that last_action_state includes the last JOINT action to the state

obs = game.reset()
print("Observation space:", game.observation_space)
print("Action space:", game.action_space)
print("Initial observation:", obs)

ret = 0
for i in range(10):
    actions = [np.random.randint(0, 2) for _ in range(3)]
    obs, rewards, done, info = game.step(actions)
    ret += sum(rewards)
    # print(f"Step {i + 1}")
    # print(f"\tActions: {actions}")
    # print(f"\tObservation: {obs}")
    # print(f"\tRewards: {rewards}")
    # print(f"\tDone: {done}")
    # print(f"\tInfo: {info}")
    if all(done):
        break

print("Total return:", ret)

