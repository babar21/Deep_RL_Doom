import numpy as np


class ReplayMemory:

    def __init__(self, max_size, screen_shape, n_variables, n_features, is_DFP = False):
        assert len(screen_shape) == 3
        self.max_size = max_size
        self.screen_shape = screen_shape
        self.n_variables = n_variables
        self.n_features = n_features
        self.cursor = 0
        self.full = False
        self.screens = np.zeros((max_size,) + screen_shape, dtype=np.uint8)
        if n_variables:
            self.variables = np.zeros((max_size, n_variables), dtype=np.int32)
        if n_features:
            self.features = np.zeros((max_size, n_features), dtype=np.int32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.isfinal = np.zeros(max_size, dtype=np.bool)
        self.is_DFP
        #We introduce include_goal and self.goals for the calculation of train_network for DFP
        if is_DFP:
            self.goals = np.zeros((max_size,3), dtype=np.float32)
            
    @property
    def size(self):
        return self.max_size if self.full else self.cursor

    def add(self, screen, variables, features, action, reward, is_final, goal=None):
        assert self.n_variables == 0 or self.n_variables == len(variables)
        assert self.n_features == 0 or self.n_features == len(features)
        self.screens[self.cursor] = screen
        if self.n_variables:
            self.variables[self.cursor] = variables
        if self.n_features:
            self.features[self.cursor] = features
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.isfinal[self.cursor] = is_final

        #for DFP
        if goal is not None:
            self.goals[self.cursor] = goal

        self.cursor += 1
        if self.cursor >= self.max_size:
            self.cursor = 0
            self.full = True
        


    def empty(self):
        self.cursor = 0
        self.full = False

    def get_batch(self, batch_size, hist_size, is_DFP=False):
        """
        Sample a batch of experiences from the replay memory.
        `hist_size` represents the number of observed frames for s_t, so must
        be >= 1
        """
        
        assert self.size > 0, 'replay memory is empty'
        assert hist_size >= 1, 'history is required'

        # idx contains the s_t indices
        idx = np.zeros(batch_size, dtype='int32')
        count = 0

        while count < batch_size:

            # index will be the index of s_t
            index = np.random.randint(hist_size - 1, self.size - 1)

            # check that we are not wrapping over the cursor
            if self.cursor <= index + 1 < self.cursor + hist_size:
                continue

            # s_t should not contain any terminal state, so only
            # its last frame (indexed by index) can be final
            if np.any(self.isfinal[index - (hist_size - 1):index]):
                continue

            idx[count] = index
            count += 1

        all_indices = idx.reshape((-1, 1)) + np.arange(-(hist_size - 1), 2)
        screens = self.screens[all_indices]
        variables = self.variables[all_indices] if self.n_variables else None
        features = self.features[all_indices] if self.n_features else None
        actions = self.actions[all_indices[:, :-1]]
        rewards = self.rewards[all_indices[:, :-1]]
        isfinal = self.isfinal[all_indices[:, :-1]]
        
        #DFP only
        if is_DFP:
            goals = self.goals[all_indices[:, :-1]]
            
            if self.n_features:
                time_steps = [1, 2, 4, 8, 16, 32]            
                future_features = np.zeros((batch_size, hist_size+1, len(time_steps), self.n_features))

                #i_1 tracks index of future_features, i_2 tracks index of self.features
                for i_1, i_2 in enumerate(all_indices):
                    #% to avoid index errors                
                    future_features[i_1] = [self.features[(i_2+j)%self.max_size] - self.features[i_2] for j in time_steps]
            else:
                future_features = None
        
        # check batch sizes
        assert idx.shape == (batch_size,)
        assert screens.shape == (batch_size, hist_size + 1) + self.screen_shape
        assert (variables is None or variables.shape == (batch_size,
                hist_size + 1, self.n_variables))
        assert (features is None or features.shape == (batch_size,
                hist_size + 1, self.n_features))
        assert actions.shape == (batch_size, hist_size)
        assert rewards.shape == (batch_size, hist_size)
        assert isfinal.shape == (batch_size, hist_size)
        
        #goals are of length 3
        assert goals.shape == (batch_size, 3, hist_size)

        #6 = len(time_steps)
        assert (future_features is None or future_features.shape == (batch_size,
                hist_size + 1, 6, self.n_features))
        return dict(
            screens=screens,
            variables=variables,
            features=features,
            actions=actions,
            rewards=rewards,
            isfinal=isfinal,
            goals=goals if is_DFP else None,
            future_features=future_features if is_DFP else None
        )
