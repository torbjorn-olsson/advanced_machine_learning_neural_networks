import numpy as np
import random
import math
import h5py
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import clone_model 


# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        self.Q = {}
        self.reward_tots = np.array([0 for _ in range(self.episode_count)])
        self.avg_rewards = []

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

    def fn_load_strategy(self,strategy_file):
        with open(strategy_file, 'rb') as f:
            self.Q = pickle.load(f)
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        board = self.gameboard.board
        m = self.gameboard.N_row
        n = self.gameboard.N_col
        tiles = self.gameboard.tiles
        state = 0
        exp = 1

        for i in range(m):
            for j in range(n):
                state += (int(board[i][j]) +1) // 2*exp
                exp *= 2

        state *= len(self.gameboard.tiles)
        state += self.gameboard.cur_tile_type

        self.state = state
        # print(state)

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_rand_action(self):
        ct = self.gameboard.cur_tile_type
        tiles = self.gameboard.tiles
        state = self.state
        Q = self.Q

        tile_x, tile_orientation = random.choice(list(Q[state].keys()))
        self.curr_action = (tile_x, tile_orientation)
        if (self.gameboard.fn_move(tile_x, tile_orientation) == 1):
            print(f'Invalid action {self.curr_action}')

    def fn_init_state(self):
        Q = self.Q
        state = self.state
        ct = self.gameboard.cur_tile_type
        tiles = self.gameboard.tiles

        Q[state] = {}
        for tile_x in range(0, self.gameboard.N_col+1):
            for tile_orientation in range(0, len(tiles[ct])+1):
                if (self.gameboard.fn_move(tile_x, tile_orientation) == 0):
                    Q[state][(tile_x, tile_orientation)] = 0

    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        ct = self.gameboard.cur_tile_type
        tiles = self.gameboard.tiles
        state = self.state
        Q = self.Q

        if state not in Q:
            self.fn_init_state()

        if (random.random()<self.epsilon):
            self.fn_select_rand_action()
        else:
            maxrev, action = max([(rev, act) for act,rev in Q[state].items()])
            best_actions = [act for act, rev in Q[state].items() if rev == maxrev]
            self.curr_action = random.choice(best_actions)
            if (self.gameboard.fn_move(self.curr_action[0], self.curr_action[1])==1):
                print(f'Invalid action {self.curr_action}')

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
    
    def fn_reinforce(self,old_state,reward):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        ct = self.gameboard.cur_tile_type
        state = self.state
        Q = self.Q


        if state not in Q:
            self.fn_init_state()

        maxrev, action = max([(rev, act) for act, rev in Q[state].items()])
        old_val =  Q[old_state][self.curr_action]

        new_val = old_val + self.alpha * (reward + maxrev - old_val)
        Q[old_state][self.curr_action] = new_val

        # Useful variables: 
        # 'self.alpha' learning rate

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                avg = np.average(self.reward_tots[range(self.episode-100,self.episode)])
                self.avg_rewards.append(avg)
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(avg),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    print(self.Q)
                    print(self.reward_tots)
                    # with open(f'Q_{self.episode}.json', 'w') as f:
                    #     f.write(json.dumps(self.Q))
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays

            if self.episode>=self.episode_count:
                with open(f'Q{self.episode_count}', 'wb') as f:
                    pickle.dump(self.Q, f)

                with open(f'rev{self.episode_count}', 'wb') as f:
                    pickle.dump(self.reward_tots, f)

                with open(f'revavg{self.episode_count}', 'wb') as f:
                    pickle.dump(self.avg_rewards, f)

                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            #print(self.curr_action)
            #print(self.gameboard.tile_x, self.gameboard.tile_orientation)
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()

            old_state = self.state

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        # self.replay_buffer_size = 1000
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks, experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        self.eval = False
        self.exp_buffer = []
        self.reward_tots = np.array([0 for _ in range(self.episode_count)])
        self.avg_rewards = []

        m = self.gameboard.N_row
        n = self.gameboard.N_col

        self.init_actions()
        max_act = max([len(self.acts[i]) for i in self.acts])

        Qnn = tf.keras.Sequential()
        # input layer
        Qnn.add(tf.keras.layers.Input(shape=(m*n + len(self.gameboard.tiles),)))

        # hidden layers
        Qnn.add(tf.keras.layers.Dense(64, activation='relu', trainable=True))
        Qnn.add(tf.keras.layers.Dense(64, activation='relu', trainable=True))
        
        # output layer
        Qnn.add(tf.keras.layers.Dense(max_act, trainable=True))

        # optimizer
        self.opt = tf.keras.optimizers.SGD(lr = self.alpha)
        self.loss = tf.keras.losses.MeanSquaredError()

        Qnn.compile(optimizer=self.opt, loss=self.loss)

        self.Qnn = Qnn
        self.Qnnh = clone_model(Qnn)
        self.Qnnh.compile(optimizer=self.opt, loss=self.loss)

        # Useful variables:
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file
        self.eval = True
        self.Qnn = tf.keras.models.load_model(strategy_file)
        for i in range(len(self.Qnn.layers)):
            w = self.Qnn.layers[i].get_weights()
            self.Qnnh.layers[i].set_weights(w)


    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        board = self.gameboard.board
        m = self.gameboard.N_row
        n = self.gameboard.N_col
        tiles = self.gameboard.tiles
        n_tiles = len(tiles)

        state = [0 for _ in range(m*n + n_tiles)]

        k = 0
        for i in range(m):
            for j in range(n):
                state[k] = board[i][j]
                k += 1

        for i in range(n_tiles):
            if i == self.gameboard.cur_tile_type:
                state[k] = 1
            else:
                state[k] = -1
            k += 1

        self.state = state

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def init_actions(self):
        tiles = self.gameboard.tiles
        n = self.gameboard.N_col
        acts = {}

        for ct in range(len(tiles)):
            self.gameboard.cur_tile_type = ct
            acts[ct] = []
            for tile_x in range(0, n+1):
                for tile_orientation in range(0, len(tiles[ct])+1):
                    if (self.gameboard.fn_move(tile_x, tile_orientation) == 0):
                        acts[ct].append((tile_x, tile_orientation))

        self.acts = acts

    def fn_select_action(self):

        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        eps_e = max([self.epsilon, 1 - self.episode / self.epsilon_scale])
        ct = self.gameboard.cur_tile_type
        Qnn = self.Qnn
        acts = self.acts

        if random.random()<eps_e and not self.eval:
            self.action = random.randint(0, len(self.acts[ct])-1)
        else:
            res = Qnn.predict(np.atleast_2d(np.array(self.state)))
            Qvals = list(res[0])
            maxQ = max(Qvals[:len(acts[ct])])
            best_actions = [i for i in range(len(acts[ct])) if Qvals[i] == maxQ]
            self.action = random.choice(best_actions)

        tile_x, tile_orientation = acts[ct][self.action]
        self.gameboard.fn_move(tile_x, tile_orientation)

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self, batch):
        X = []
        y_true = []

        for st, at, rt1, st1, isTerminal in batch:
            ya = rt1
            if not isTerminal:
                res = self.Qnnh.predict(np.atleast_2d(np.array(st1)))
                ya += max(list(res[0]))

            st_np = np.atleast_2d(np.array(st, dtype='float32'))
            X.append(list(st_np[0] * 1.0))
            y_true_i = self.Qnn.predict(st_np)[0]
            y_true_i[at] = ya
            y_true.append(list(y_true_i))

        # print(X)
        # print(y_true)

        self.Qnn.fit(np.array(X), np.array(y_true), verbose=0)

        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self




        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                avg = np.average(self.reward_tots[range(self.episode - 100, self.episode)])
                self.avg_rewards.append(avg)
                print('episode ' + str(self.episode) + '/' + str(self.episode_count) + ' (reward: ', str(avg), ')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];
                if self.episode in saveEpisodes:
                    self.Qnn.save(f'model{self.episode}.h5')

                    with open(f'NNrev{self.episode_count}', 'wb') as f:
                        pickle.dump(self.reward_tots, f)

                    with open(f'NNrevavg{self.episode_count}', 'wb') as f:
                        pickle.dump(self.avg_rewards, f)

                    print(self.reward_tots)
                    # with open(f'Q_{self.episode}.json', 'w') as f:
                    #     f.write(json.dumps(self.Q))
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays

            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            self.old_state = self.state.copy()

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            self.exp_buffer.append((self.old_state.copy(), self.action, reward, self.state.copy(), self.gameboard.gameover))

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.exp_buffer, self.batch_size)
                self.exp_buffer = self.exp_buffer[1:]
                self.fn_reinforce(batch)


                #if self.episode % 10 == 0:
                if self.episode % self.sync_target_episode_count == 0:
                    for i in range(len(self.Qnn.layers)):
                        w = self.Qnn.layers[i].get_weights()
                        self.Qnnh.layers[i].set_weights(w)


                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to copy the current network to the target network

class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()