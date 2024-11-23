import pandas as pd
import DDPG as ddpg

if "__main__" == __name__:
    epoch = 6  # 已经训练好了5 epoch
    state_dim = 11
    action_dim = 10
    # ddpg.generate_decision(epoch, state_dim, action_dim)
    ddpg.train_DDPG(epoch, state_dim, action_dim)


