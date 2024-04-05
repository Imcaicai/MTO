import copy
import numpy as np


class PartitionNode:
    '''
    A partition node, including both the internal and leaf nodes in the partition tree
    '''
    def __init__(self, num_dims = 0, boundary = [], nid = None, pid = None, is_irregular_shape_parent = False,
                 is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = 0):
        
        # print("Initialize PartitionTree Root: num_dims",num_dims,"boundary:",boundary,"children_ids:",children_ids)
        self.num_dims = num_dims # 维度数量
        # the domain, [l1,l2,..,ln, u1,u2,..,un,], for irregular shape partition, one need to exempt its siblings
        self.boundary = boundary # 范围左开右闭?
        self.nid = nid # 节点id
        self.pid = pid # 父节点id
        self.is_irregular_shape_parent = is_irregular_shape_parent # whether the [last] child is an irregular shape partition
        self.is_irregular_shape = is_irregular_shape # an irregular shape partition cannot be further split, and it must be a leaf node
        self.num_children = num_children # number of children, should be 0, 2, or 3?
        self.children_ids = children_ids # if it's the irregular shape parent, then the last child should be the irregular partition
        self.is_leaf = is_leaf  # 是否是叶子节点
        self.node_size = node_size # 节点大小，包含的记录数
        
        # the following attributes will not be serialized
        self.dataset = None # only used in partition algorithms, temporary, should consist records that within this partition
        self.queryset = None # only used in partition algorithms, temporary, should consist queries that overlap this partition
        self.partitionable = True # only used in partition algorithms
        self.query_MBRs = None # only used in partition algorithms, temporary
        self.split_type = None # only used in partition algorithms
        
        # Rtree filters
        self.rtree_filters = None # a collection of MBRs, in the shape of boundary, used to indicate the data distribution
        
        # beam search
        self.depth = 0 # only used in beam search, root node depth is 0
        self.no_valid_partition = False # use to stop, an variant of partitionable

    # 判断给定查询是否与当前分区节点有重叠，可以优化?   
    def is_overlap(self, query):
        '''
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        !query dimension should match the partition dimensions! i.e., all projected or all not projected
        return 0 if no overlap  没有重叠
        return 1 if overlap 有重叠
        return 2 if inside  完全在分区内
        '''
        if len(query) != 2 * self.num_dims:
            return -1 # error
        
        overlap_flag = True
        inside_flag = True
        
        # 区间的开闭问题?
        for i in range(self.num_dims):
            if query[i] >= self.boundary[self.num_dims + i] or query[self.num_dims + i] <= self.boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < self.boundary[i] or query[self.num_dims + i] > self.boundary[self.num_dims + i]:
                inside_flag = False
                
        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0
    
    
    # # is_overlap的numpy版本
    # def is_overlap_np(self, query):
    #     '''
    #     the numpy version of the is_overlap function
    #     the query here and boundary class attribute should in the form of numpy array
    #     '''
    #     if all((boundary[0:self.num_dims] > query[self.num_dims:]) | (boundary[self.num_dims:] <= query[0:self.num_dims])):
    #         return 0 # no overlap
    #     elif all((boundary[0:self.num_dims] >= query[0:self.num_dims]) & (boundary[self.num_dims:] <= query[self.num_dims:])):
    #         return 2 # inside
    #     else:
    #         return 1 # overlap


    # 用于确定一个数据点是否包含在当前的分区节点中
    def is_contain(self, point):
        '''
        used to determine wheter a data point is contained in this node
        point: [dim1_value, dim2_value,...], should has the same dimensions as this node
        '''
        for i in range(self.num_dims):
            if point[i] > self.boundary[self.num_dims + i] or point[i] < self.boundary[i]:
                return False
        return True
    
    # 用于获取候选切割的位置
    def get_candidate_cuts(self, extended = False):
        '''
        get the candidate cut positions
        if extended is set to True, also add medians from all dimensions
        '''
        candidate_cut_pos = []
        for query in self.queryset:
            for dim in range(self.num_dims):
                # check if the cut position is inside the partition, as the queryset are queries overlap this partition
                if query[dim] >= self.boundary[dim] and query[dim] <= self.boundary[self.num_dims+dim]:
                    candidate_cut_pos.append((dim, query[dim]))
                if query[self.num_dims+dim] >= self.boundary[dim] and query[self.num_dims+dim] <= self.boundary[self.num_dims+dim]:
                    candidate_cut_pos.append((dim, query[self.num_dims+dim]))
        
        if extended:
            for dim in range(self.num_dims):
                split_value = np.median(self.dataset[:,dim])
                candidate_cut_pos.append((dim, split_value))
        
        return candidate_cut_pos
    
    # 用于判断给定切割列和值，是否切割，并返回收益和子节点大小
    def if_split(self, split_dim, split_value, data_threshold, test = False): # rename: if_split_get_gain
        '''
        return the skip gain and children partition size if split a node from a given split dimension and split value
        '''
        #print("current_node.nid:", current_node.nid)
        #print("current_node.is_leaf:", current_node.is_leaf)
        #print("current_node.dataset is None:", current_node.dataset is None)
        sub_dataset1_size = np.count_nonzero(self.dataset[:,split_dim] < split_value) # process time: 0.007
        sub_dataset2_size = self.node_size - sub_dataset1_size  # 右子节点数量

        if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
            return False, 0, sub_dataset1_size, sub_dataset2_size
        
        left_part, right_part, mid_part = self.split_queryset(split_dim, split_value)
        num_overlap_child1 = len(left_part) + len(mid_part)
        num_overlap_child2 = len(right_part) + len(mid_part)
        
        if test:
            print("num left part:",len(left_part), "num right part:",len(right_part), "num mid part:",len(mid_part))
            print("left part:", left_part, "right part:", right_part, "mid part:",mid_part)
        
        # temp_child_node1, temp_child_node2 = self.__if_split_get_child(split_dim, split_value)
        # 跳过增益为什么这么算?
        skip_gain = len(self.queryset)*self.node_size - num_overlap_child1*sub_dataset1_size - num_overlap_child2*sub_dataset2_size
        return True, skip_gain, sub_dataset1_size, sub_dataset2_size
    
    def if_bounding_split(self, data_threshold, approximate = False, force_extend = False):
        '''
        # the split node is assumed to be >= 2b
        approximate: whether use approximation (even distribution) to find the number of records within a partition
        force_extend: whether extend the bounding partition to make its size greater than data_threshold, if possible
        return availability, skip gain, and the (possible extended) bound
        '''
        max_bound = self.__max_bound(self.queryset)
        bound_size = self.query_result_size(max_bound, approximate)
        if bound_size is None:
            return False, None, None
        
        extended_bound = copy.deepcopy(max_bound)
        if bound_size < data_threshold: # assume the partition is >= 2b, then we must be able to find the valid extension
            if force_extend:
                side = 0
                for dim in range(self.num_dims):
                    valid, extended_bound, bound_size = self.__try_extend(extended_bound, dim, 0, data_threshold) # lower side
                    if valid:
                        break
                    valid, extended_bound, bound_size = self.__try_extend(extended_bound, dim, 1, data_threshold) # upper side
                    if valid:
                        break
            else:
                return False, None, None   
        
        remaining_size = self.node_size - bound_size
        if remaining_size < data_threshold:
            return False, None, None
        cost_before_split = len(self.queryset) * self.node_size
        cost_bound_split = len(self.queryset) * bound_size
        skip_gain = cost_before_split - cost_bound_split
        
        if force_extend:
            return True, skip_gain, extended_bound
        else:
            return True, skip_gain, max_bound # TODO: should we also return the extended bound? 

    def if_new_bounding_split(self, data_threshold, approximate = False, force_extend = True):
        '''
        Also known as var-bounding split or multi-group split
        In this version, we try to generate a collection of MBR partitions if every MBR satisfy:
        1. its size <= b; or
        2. it contains only 1 query; or
        3. |Q|*Core + b > its size * |Q|
        
        OR (if the above failed) a single bounding partition and an irregular shape partition as the old version
        '''
        if self.query_MBRs is None or len(self.query_MBRs) == 0:
            return False
        
        check_valid = True
        extended_flag = False
        
        # simple pruning
        if len(self.query_MBRs) * data_threshold > self.node_size:
            check_valid = False
        else:
            for MBR in self.query_MBRs:
                if MBR.bound_size <= data_threshold or MBR.num_query == 1 or MBR.check_condition3(data_threshold):
                    pass
                else:
                    check_valid = False
                    break
        
        if check_valid:
            # try extend the MBRs to satisfy b, and check whether the extended MBRs overlap with others
            for MBR in self.query_MBRs:
                if MBR.bound_size < data_threshold:
                    MBR.boundary, MBR.bound_size = self.extend_bound(MBR.boundary, data_threshold)
                    MBR.is_extended = True
                    if MBR.bound_size > 2 * data_threshold:
                        MBR.ill_extended = True # if there are too many same key records
                if MBR.is_extended:
                    extended_flag = True # also for historical extended MBRs !!!
                    
            
        # check if the extended MBRs overlaps each other
        if extended_flag and len(self.query_MBRs) > 1:
            for i in range(len(self.query_MBRs) - 1):
                for j in range(i+1, len(self.query_MBRs)):
                    if self.query_MBRs[i].ill_extended or self.query_MBRs[j].ill_extended or self.__is_overlap(self.query_MBRs[i].boundary, self.query_MBRs[j].boundary):
                        #print("partition",self.nid,"found overlap of extended MBRs:", self.query_MBRs[i].boundary, self.query_MBRs[j].boundary)
                        check_valid = False
                        break
                if not check_valid:
                    break
        
        if len(self.query_MBRs) == 1 and self.query_MBRs[0].ill_extended: # in case there is only 1 MBR
            check_valid = False
        
        # check the remaining partition size, if it's not greater than b, return false
        remaining_size = self.node_size
        for MBR in self.query_MBRs:
            remaining_size -= MBR.bound_size
        if remaining_size < data_threshold:
            check_valid = False
        
        # if the above failed
        if check_valid:
            return True # since this is the optimal, we don't need to return skip
        else:
            # do we need to restore the MBRs?
            # NO, when split cross a MBR, it will be rebuilt on both side
            # In other cases, the extended MBR doesn't matter
            return False  
    
    def if_general_group_split(self, data_threshold):
        '''
        the general group split in PAW (this one use merge and doesn't handle overlap)
        '''
        if self.query_MBRs is None or len(self.query_MBRs) == 0:
            #print("PAW: no MBRs")
            return False
        
        def check_MBR_sizes():
            for MBR in self.query_MBRs:
                if MBR.bound_size < data_threshold:
                    return False
            return True
            
        while not check_MBR_sizes() and len(self.query_MBRs)>= 2: # what if only 1 MBR and its size is less than b
            # merge MBRs
            merged_records = [] # (cost, MBR index1, MBR_index2)
            for i in range(len(self.query_MBRs) - 1):
                for j in range(i+1, len(self.query_MBRs)):
                    merged_MBR = self.__if_merge_2MBRs(self.query_MBRs[i], self.query_MBRs[j])
                    cost = merged_MBR.num_query * merged_MBR.bound_size
                    merged_records.append((cost, i, j, merged_MBR))
            merged_records.sort()
            merged_MBR = merged_records[0][-1]
            i, j = merged_records[0][1], merged_records[0][2]
            del self.query_MBRs[j]
            del self.query_MBRs[i] # i < j
            self.query_MBRs.append(merged_MBR)
            #print("merged MBR size:", merged_MBR.bound_size, "boundary:",merged_MBR.boundary)
        
        # check if every partition size is greater than b
        remaining_size = self.node_size
        for MBR in self.query_MBRs:
            remaining_size -= MBR.bound_size
            if MBR.bound_size < data_threshold:
                #print("PAW: MBR size < b, len(MBRs):",len(self.query_MBRs))
                return False
        if remaining_size < data_threshold:
            #print("PAW: remaining size < b")
            return False
        
        # get the cost
        cost = 0
        for MBR in self.query_MBRs:
            cost += MBR.num_query * MBR.bound_size
        # return cost
        skip = len(self.queryset) * self.node_size - cost
        #print("PAW: skip", skip)
        return skip
    
    def if_general_group_split_2(self, data_threshold):
        '''
        this one use extend() + handle overlap
        '''
        if self.query_MBRs is None or len(self.query_MBRs) == 0:
            #print("PAW: no MBRs")
            return False
        
        def check_MBR_overlap(MBRs):
            '''
            return True if overlaps
            '''
            if len(MBRs) >= 2:
                for i in range(len(MBRs)-1):
                    for j in range(i+1, len(MBRs)):
                        if self.__is_overlap(MBRs[i].boundary, MBRs[j].boundary):
                            return True
            return False
        
        # extend
        for MBR in self.query_MBRs:
            if MBR.bound_size < data_threshold:
                #print("extend MBR", MBR.boundary, "before extend, size:", MBR.bound_size)
                MBR.boundary, MBR.bound_size = self.extend_bound(MBR.boundary, data_threshold)
                #print("MBR", MBR.boundary, "after extend, size:", MBR.bound_size)
                MBR.is_extended = True
            if MBR.bound_size > 2 * data_threshold:
                MBR.ill_extended = True # if there are too many same key records
                
        # check overlap:
        if check_MBR_overlap(self.query_MBRs):
            #print("PAW: overlap exists in (extended) MBRs")
            return False
        else:
            # check remaining size
            cost = 0
            remaining_size = self.node_size
            for MBR in self.query_MBRs:
                if MBR.bound_size < data_threshold:
                    #print("PAW: MBR size < b")
                    return False
                remaining_size -= MBR.bound_size
                cost += MBR.bound_size * MBR.num_query
            if remaining_size < data_threshold:
                #print("PAW: remaining size < b")
                return False
            
            # return skip
            skip = len(self.queryset) * self.node_size - cost
            return skip
    
    
    def if_dual_bounding_split(self, split_dim, split_value, data_threshold, approximate = False):
        '''
        check whether it's available to perform dual bounding split
        return availability and skip gain
        '''
        # split queriese first
        left_part, right_part, mid_part = self.split_queryset(split_dim, split_value)
        max_bound_left = self.__max_bound(left_part)
        max_bound_right = self.__max_bound(right_part)
        
        # Should we only consider the case when left and right cannot be further split? i.e., [b,2b)
        # this check logic is given in the PartitionAlgorithm, not here, as the split action should be general
        naive_left_size = np.count_nonzero(self.dataset[:,split_dim] < split_value)
        naive_right_size = self.node_size - naive_left_size
        
        # get (irregular-shape) sub-partition size
        left_size = self.query_result_size(max_bound_left, approximate)
        if left_size is None: # there is no query within the left 
            left_size = naive_left_size # use the whole left part as its size
        if left_size < data_threshold:
            return False, None
        right_size = self.query_result_size(max_bound_right, approximate)
        if right_size is None: # there is no query within the right
            right_size = naive_right_size # use the whole right part as its size
        if right_size < data_threshold:
            return False, None
        remaining_size = self.node_size - left_size - right_size
        if remaining_size < data_threshold:
            return False, None
        
        # check cost
        cost_before_split = len(self.queryset) * self.node_size
        cost_dual_split = len(left_part) * left_size + len(right_part) * right_size + len(mid_part) * remaining_size
        for query in mid_part:
            # if it overlap left bounding box
            if max_bound_left is None or self.__is_overlap(max_bound_left, query) > 0:
                cost_dual_split += left_size
            # if it overlap right bounding box
            if max_bound_right is None or self.__is_overlap(max_bound_right, query) > 0:
                cost_dual_split += right_size
        skip_gain = cost_before_split - cost_dual_split
        return True, skip_gain
    
    # 给定切割维度和值，返回切割后将跨越多少查询
    def num_query_crossed(self, split_dim, split_value):
        '''
        similar to the split_queryset function, but just return how many queries the intended split will cross
        '''
        count = 0
        if self.queryset is not None:
            for query in self.queryset:
                if query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    count += 1
            return count
        return None
    
    # 根据切割值把查询集分为3部分，并返回
    def split_queryset(self, split_dim, split_value):
        '''
        split the queryset into 3 parts:
        the left part, the right part, and those cross the split value
        '''
        if self.queryset is not None:
            left_part = []
            right_part = []
            mid_part = []
            for query in self.queryset:
                #print("[Split Queryset] query:",query, "split dim:", split_dim, "split value", split_value, "query[split dim]:",query[split_dim])
                if query[split_dim] >= split_value:
                    #print("[Split Queryset] query is right part")
                    right_part.append(query)
                elif query[self.num_dims + split_dim] <= split_value:
                    #print("[Split Queryset] query is left part")
                    left_part.append(query)
                elif query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    #print("[Split Queryset] query is mid part")
                    mid_part.append(query)
                else:
                    #print("[Split Queryset] query is nothing")
                    pass
            #print("[Split Queryset] left part:",len(left_part), "right part:",len(right_part),"mid part:",len(mid_part))
            return left_part, right_part, mid_part
    
    # 获取当前节点上查询结果的大小（分区上的记录数）
    def query_result_size(self, query, approximate = False):
        '''
        get the query result's size on this node
        the approximate parameter is set to True, the use even distribution to approximate
        '''
        if query is None:
            return None
        
        result_size = 0
        if approximate:
            query_volume = 1
            volume = 1
            for d in range(self.num_dims):
                query_volume *= query[self.num_dims + d] - query[d]
                volume *= self.boundary[self.num_dims + d] - self.boundary[d]

            result_size = int(query_volume / volume * self.node_size)
        else:
            constraints = []
            for d in range(self.num_dims):
                constraint_L = self.dataset[:,d] >= query[d]
                constraint_U = self.dataset[:,d] <= query[self.num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            result_size = np.count_nonzero(constraint)
        return result_size

    def extend_bound(self, bound, data_threshold, print_info = False, algorithm = 2):
        '''
        extend a bound to be at least b, assume the bound is within the partition boundary
        algorithm == 1: binary search on each dimension
        algorithm == 2: Ken's extend bound method
        '''
        # safe guard
        current_size = self.query_result_size(bound, approximate = False)
        if current_size >= data_threshold:
            return bound, current_size
        
        if algorithm == 1:
            side = 0
            for dim in range(self.num_dims): # or it cannot adapted to other dataset ! #[2,0,1,4,3,5,6]: reranged by distinct values
                if dim+1 > self.num_dims:
                    continue
                valid, bound, bound_size = self.__try_extend(bound, dim, 0, data_threshold, print_info) # lower side
                if print_info:
                    print("dim:",dim,"current bound:",bound,valid,bound_size)
                if valid:
                    break
                valid, bound, bound_size = self.__try_extend(bound, dim, 1, data_threshold, print_info) # upper side
                if print_info:
                    print("dim:",dim,"current bound:",bound,valid,bound_size)
                if valid:
                    break
            return bound, bound_size
        
        elif algorithm == 2:
            center = [(bound[i] + bound[i + self.num_dims]) / 2 for i in range(self.num_dims)]
            radius = [(bound[i + self.num_dims] - bound[i]) / 2 for i in range(self.num_dims)]
            f_records = []
            for point in self.dataset:
                dist_ratio = [abs(point[i] - center[i]) / radius[i] for i in range(self.num_dims)]
                max_dist_ratio = max(dist_ratio)
                f_records.append(max_dist_ratio)
            f_records.sort()
            threshold_ratio = f_records[data_threshold]
            extend_bound_lower = [center[i] - threshold_ratio*radius[i] for i in range(self.num_dims)]
            extend_bound_upper = [center[i] + threshold_ratio*radius[i] for i in range(self.num_dims)]
            extended_bound = extend_bound_lower + extend_bound_upper
            extended_bound = self.__max_bound_single(extended_bound)
            bound_size = self.query_result_size(extended_bound, approximate = False)
            return extended_bound, bound_size
        
    
    # = = = = = internal functions = = = = =
    
    def __try_extend(self, current_bound, try_dim, side, data_threshold, print_info = False):
        '''
        side = 0: lower side
        side = 1: upper side
        return whether this extend has made bound greater than b, current extended bound, and the size
        '''
        # first try the extreme case
        dim = try_dim
        if side == 1:
            dim += self.num_dims
            
        extended_bound = copy.deepcopy(current_bound)
        extended_bound[dim] = self.boundary[dim]
        
        bound_size = self.query_result_size(extended_bound, approximate = False)
        if bound_size < data_threshold:
            return False, extended_bound, bound_size
        
        # binary search in this extend direction
        L, U = None, None
        if side == 0:
            L, U = self.boundary[dim], current_bound[dim]
        else:
            L, U = current_bound[dim], self.boundary[dim]
        
        if print_info:
            print("L,U:",L,U)
        
        loop_count = 0
        while L < U and loop_count < 30:
            mid = (L+U)/2
            extended_bound[dim] = mid
            bound_size = self.query_result_size(extended_bound, approximate = False)
            if bound_size < data_threshold:
                L = mid
            elif bound_size > data_threshold:
                U = mid
                if U - L < 0.00001:
                    break
            else:
                break
            if print_info:
                print("loop,L:",L,"U:",U,"mid:",mid,"extended_bound:",extended_bound,"size:",bound_size)
            loop_count += 1
            
        return bound_size >= data_threshold, extended_bound, bound_size
        
    
    def __is_overlap(self, boundary, query):
        '''
        the difference between this function and the public is_overlap function lies in the boundary parameter
        '''
        if len(query) != 2 * self.num_dims:
            return -1 # error
        
        overlap_flag = True
        inside_flag = True
        
        for i in range(self.num_dims):
            if query[i] >= boundary[self.num_dims + i] or query[self.num_dims + i] <= boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < boundary[i] or query[self.num_dims + i] > boundary[self.num_dims + i]:
                inside_flag = False
                
        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0
        

    def __max_bound(self, queryset):
        '''
        bound the queries by their maximum bounding rectangle !NOTE it is for a collection of queries!!!
        then constraint the MBR by the node's boundary!
        
        the return bound is in the same form as boundary
        '''
        if len(queryset) == 0:
            return None
        #if len(queryset) == 1:
        #    pass, I don't think there will be shape issue here
        
        max_bound_L = np.amin(np.array(queryset)[:,0:self.num_dims],axis=0).tolist()
        # bound the lower side with the boundary's lower side
        max_bound_L = np.amax(np.array([max_bound_L, self.boundary[0:self.num_dims]]),axis=0).tolist()
        
        max_bound_U = np.amax(np.array(queryset)[:,self.num_dims:],axis=0).tolist()
        # bound the upper side with the boundary's upper side
        max_bound_U = np.amin(np.array([max_bound_U, self.boundary[self.num_dims:]]),axis=0).tolist()
        
        max_bound = max_bound_L + max_bound_U # concat
        return max_bound
    
    def __max_bound_single(self, query, parent_boundary = None):
        '''
        bound anything in the shape of query by the current partition boundary
        '''
        if parent_boundary is None:
            for i in range(self.num_dims):
                query[i] = max(query[i], self.boundary[i])
                query[self.num_dims + i] = min(query[self.num_dims + i], self.boundary[self.num_dims + i])
            return query
        else:
            for i in range(self.num_dims):
                query[i] = max(query[i], parent_boundary[i])
                query[self.num_dims + i] = min(query[self.num_dims + i], parent_boundary[self.num_dims + i])
            return query