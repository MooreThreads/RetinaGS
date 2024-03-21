import torch
import numpy as np
import collections

class NaiveWorkloadBalancer():
    def __init__(self, num_rank:int, model2rank:dict) -> None:
        """
        make sure every GPU responses to no more than max_task 
        """
        self.num_rank:int = num_rank
        self.model2rank:dict = model2rank

    def load_to_rank(self, load_per_model):
        load_of_rank = np.zeros(self.num_rank)
        for model_id, load in enumerate(load_per_model):
            load_of_rank[self.model2rank[model_id]] += load
        return load_of_rank                 

    def get_groups(self, relation_matrix:np.ndarray, shuffled_indices:np.ndarray, max_task:int, max_batch:int):
        """
        relation_matrix: (num_sample, num_model) np.ndarray|tensor('cpu'), -1 stands for no relation 
        """
        groups = []
        shuffled_indices:collections.deque = collections.deque(shuffled_indices)

        _g, _load = [], np.zeros(self.num_rank)
        while len(shuffled_indices) > 0:
            idx = shuffled_indices.popleft()
            load_per_model = np.array(relation_matrix[idx] >= 0).astype(int)
            load_per_rank = self.load_to_rank(load_per_model)

            if len(_g) == 0:
                # group shall not be empty
                _g.append(idx)
                _load += load_per_rank
            elif len(_g) >= max_batch:
                # save group, create new group, and push back idx
                groups.append(tuple(_g))
                _g, _load = [], np.zeros(self.num_rank)
                shuffled_indices.appendleft(idx)
            else:
                # try to put idx into group
                enlarged_load = _load + load_per_rank
                if np.sum(enlarged_load > max_task) > 0:
                    # save group, create new group, and push back idx
                    groups.append(tuple(_g))
                    _g, _load = [], np.zeros(self.num_rank)
                    shuffled_indices.appendleft(idx)
                else:
                    # update group
                    _g.append(idx)
                    _load = enlarged_load
         
        # save final group if is is not empty 
        if len(_g) > 0:
            groups.append(tuple(_g))

        return tuple(groups)
        
