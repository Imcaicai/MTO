
import time
import numpy as np
from rtree import index # this package is only used for constructing Rtree filter
from partition_tree import PartitionTree


class PartitionAlgorithm:
    '''
    The partition algorithms, inlcuding NORA, QdTree and kd-tree.
    '''
    def __init__(self, data_threshold = 10000):
        self.partition_tree = None
        self.data_threshold = data_threshold
    
    
    # = = = = = public functions (API) = = = = =
    def InitializeWithQDT(self, queries, num_dims, boundary, dataset, data_threshold):
        '''
        # should I also store the candidate cut positions in Partition Node ?
        The dimension of queries should match the dimension of boundary and dataset!
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        self.partition_tree.pt_root.queryset = queries # assume all queries overlap with the boundary
        start_time = time.time()
        self.__QDT(data_threshold)
        end_time = time.time()
        return end_time-start_time
     
    def InitializeWithKDT(self, num_dims, boundary, dataset, data_threshold):
        '''
        num_dims denotes the (first) number of dimension to split, usually it should correspond with the boundary
        rewrite the KDT using PartitionTree data structure
        call the recursive __KDT methods
        '''
        self.partition_tree = PartitionTree(num_dims, boundary)
        self.partition_tree.pt_root.node_size = len(dataset)
        self.partition_tree.pt_root.dataset = dataset
        # start from the first dimension
        start_time = time.time()
        self.__KDT(0, data_threshold, self.partition_tree.pt_root)
        end_time = time.time()
        #print("Build Time (s):", end_time-start_time)
    
    def ContinuePartitionWithKDT(self, existing_partition_tree, data_threshold):
        '''
        pass in a PartitionTree instance
        then keep partition its leaf nodes with KDT, if available
        '''
        self.partition_tree = existing_partition_tree
        leaves = existing_partition_tree.get_leaves()
        for leaf in leaves:
            self.__KDT(0, data_threshold, leaf)
    
    def CreateRtreeFilter(self, data_threshold, capacity_ratio = 0.5):
        '''
        create Rtree MBRs for leaf nodes as a filter layer for skew dataset
        '''
        for leaf in self.partition_tree.get_leaves():
            if leaf.is_irregular_shape:
                continue
            else:
                MBRs = self.__CreateRtreeMBRs(leaf.dataset, data_threshold, capacity_ratio)   
                leaf.rtree_filters = MBRs
    
    def RedundantPartitions(self, redundant_space, queries, dataset, data_threshold, weight = None):
        '''
        create redundant partitions to maximize the cost deduction, the extra space is limited by the redundant space
        this is a typical dynamic programming problem
        '''
        old_costs = self.partition_tree.get_queryset_cost(queries)
        spaces = self.__real_result_size(dataset, queries)
        spaces = [max(s, data_threshold) for s in spaces]
        gains = [old_costs[i]-spaces[i] for i in range(len(queries))]
        
        #print("old cost:",old_costs)
        #print("spaces:", spaces)
        #print("gains:", gains)
        
        if weight is not None: # the expected query amount
            gains = [gains[i]*weight[i] for i in range(len(queries))]
        
        max_total_gain, materialized_queries = self.__RPDP(0, gains, spaces, redundant_space, {})
        
        query_size = len(queries) if weight is None else sum(weight)
        old_query_cost = sum(old_costs)
        old_average_query_cost = old_query_cost / query_size
        new_query_cost = old_query_cost - max_total_gain
        new_average_query_cost = new_query_cost / query_size
        
        #print("max total gain:", max_total_gain)
        #print("old_query_cost:", old_query_cost, "new_query_cost:", new_query_cost)
        #print("old_average_query_cost:", old_average_query_cost, "new_average_query_cost:", new_average_query_cost)
        
        return max_total_gain, materialized_queries
    
    # = = = = = internal functions = = = = =    
    def __RPDP(self, i, gains, spaces, total_space, i_space_dict):
        '''
        i: the current query id to be considered
        total_space: the remaining redundant space
        '''
        key = (i, total_space)
        if key in i_space_dict:
            return i_space_dict[key]
        
        if i >= len(gains): # end
            return (0, [])
        
        gain, Q = None, None
        if total_space > spaces[i]:
            # create RP for this query
            (gain1, Q1) = self.__RPDP(i+1, gains, spaces, total_space-spaces[i], i_space_dict)
            (gain1, Q1) = (gains[i] + gain1, [i] + Q1)
            # do not create RP for this query
            (gain2, Q2) = self.__RPDP(i+1, gains, spaces, total_space, i_space_dict) 
            (gain, Q) = (gain1, Q1) if gain1 >= gain2 else (gain2, Q2)
        else:
            # do not create RP for this query
            (gain, Q) = self.__RPDP(i+1, gains, spaces, total_space, i_space_dict)
        
        i_space_dict[key] = (gain, Q)
        return (gain, Q)
        
    
    def __real_result_size(self, dataset, queries):
        num_dims = dataset.shape[1]
        results = []
        for query in queries:
            constraints = []
            for d in range(num_dims):
                constraint_L = dataset[:,d] >= query[d]
                constraint_U = dataset[:,d] <= query[num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            result_size = np.count_nonzero(constraint)
            results.append(result_size)
        return results

    # 在树中搜索最佳分区方案
    def __QDT(self, data_threshold):
        '''
        the QdTree partition algorithm
        '''
        CanSplit = True
        while CanSplit:
            CanSplit = False           
            
            # for leaf in self.partition_tree.get_leaves():
            leaves = self.partition_tree.get_leaves()
            #print("# number of leaf nodes:",len(leaves))
            for leaf in leaves:
                     
                # print("current leaf node id:",leaf.nid, "leaf node dataset size:",len(leaf.dataset))
                if leaf.node_size < 2 * data_threshold:
                    continue
                
                candidate_cuts = leaf.get_candidate_cuts()
                
                # get best candidate cut position
                skip, max_skip, max_skip_split_dim, max_skip_split_value = 0, -1, 0, 0
                for split_dim, split_value in candidate_cuts:

                    valid,skip,_,_ = leaf.if_split(split_dim, split_value, data_threshold)
                    if valid and skip > max_skip:
                        max_skip = skip
                        max_skip_split_dim = split_dim
                        max_skip_split_value = split_value

                if max_skip > 0:
                    # if the cost become smaller, apply the cut
                    child_node1, child_node2 = self.partition_tree.apply_split(leaf.nid, max_skip_split_dim, max_skip_split_value)
                    # print(" Split on node id:", leaf.nid)
                    CanSplit = True
            
    
    def __KDT(self, current_dim, data_threshold, current_node):
        '''
        Store the dataset in PartitionNode: we can keep it, but only as a tempoary attribute
        '''
        # cannot be further split
        if current_node.node_size < 2 * data_threshold:
            return   
        
        # split the node into equal halves by its current split dimension
        median = np.median(current_node.dataset[:,current_dim])
        
        sub_dataset1_size = np.count_nonzero(current_node.dataset[:,current_dim] < median)
        sub_dataset2_size = len(current_node.dataset) - sub_dataset1_size
        
        if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
            pass
        else:
            child_node1, child_node2 = self.partition_tree.apply_split(current_node.nid, current_dim, median)
            
            # update next split dimension
            current_dim += 1
            if current_dim >= current_node.num_dims:
                current_dim %= current_node.num_dims
    
            # recursive call on sub nodes
            self.__KDT(current_dim, data_threshold, child_node1)
            self.__KDT(current_dim, data_threshold, child_node2)