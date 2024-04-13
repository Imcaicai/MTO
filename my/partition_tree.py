

import copy
from datetime import time
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
from partition_node import PartitionNode

class PartitionTree:
        '''
        The data structure that represent the partition layout, which also maintain the parent, children relation info
        Designed to provide efficient online query and serialized ability
        
        The node data structure could be checked from the PartitionNode class
        
        '''   
        def __init__(self, num_dims = 0, boundary = []):
            
            # the node id of root should be 0, its pid should be -1
            # note this initialization does not need dataset and does not set node size!

            self.pt_root = PartitionNode(num_dims, boundary, nid = 0, pid = -1, is_irregular_shape_parent = False, 
                                         is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = 0)
            self.nid_node_dict = {0: self.pt_root} # node id to node dictionary
            self.node_count = 1 # the root node


        # = = = = = public functions (API) = = = = =
        def save_tree(self, path):
            node_list = self.__generate_node_list(self.pt_root) # do we really need this step?
            serialized_node_list = self.__serialize(node_list)
            #print(serialized_node_list)
            np.savetxt(path, serialized_node_list, delimiter=',')
            return serialized_node_list
            
        def load_tree(self, path):
            serialized_node_list = np.genfromtxt(path, delimiter=',')
            self.__build_tree_from_serialized_node_list(serialized_node_list)
        
        # 给出与给定查询重叠的叶子节点id
        def query_single(self, query, using_rtree_filter = False, print_info = False, redundant_partitions = None):
            '''
            query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
            return the overlapped leaf partitions ids!
            redundant_partition: [(boundary, size)...]?
            '''
            # used only when redundant_partition is given
            def check_inside(query, partition_boundary):
                num_dims = len(query)//2
                for i in range(num_dims):
                    if query[i] >= partition_boundary[i] and query[num_dims + i] <= partition_boundary[num_dims + i]:
                        pass
                    else:
                        return False
                return True
            
            if redundant_partitions is None:
                partition_ids = self.__find_overlapped_partition(self.pt_root, query, using_rtree_filter, print_info)
                return partition_ids
            else:
                # first check if a query is inside any redundant partition, find the smallest one
                costs = []
                for rp in redundant_partitions:
                    if check_inside(query, rp[0]):
                        costs.append(rp[1])
                if len(costs) == 0:
                    # if not, use regular approach
                    partition_ids = self.__find_overlapped_partition(self.pt_root, query, using_rtree_filter, print_info)
                    return partition_ids
                else:
                    # return the smallest rp size
                    return [-min(costs)] # the minus sign used to differentiate from partition ides
        
        def query_batch(self, queries):
            '''
            to be implemented
            '''
            pass
        
        # 返回每个查询的成本列表
        def get_queryset_cost(self, queries):
            '''
            return the cost array directly
            '''
            costs = []
            for query in queries:
                overlapped_leaf_ids = self.query_single(query)
                cost = 0
                for nid in overlapped_leaf_ids:
                    cost += self.nid_node_dict[nid].node_size
                costs.append(cost)
            return costs
        

        def evaluate_worst_query_cost(self, queries, data_threshold = 10000):
            node_size = 0
            num_nodes = 0
            for nid, node in self.nid_node_dict.items():
                if node is not None and node.is_leaf:
                    node_size += node.node_size
                    num_nodes += node.node_size // data_threshold
            return node_size, node_size * len(queries), num_nodes, num_nodes * len(queries)

        
        # 评估一组查询的逻辑I/O，返回平均查询成本
        def evaluate_query_cost(self, queries, using_rtree_filter = False, redundant_partitions = None, data_threshold = 10000):
            '''
            get the logical IOs of the queris
            return the average query cost
            '''
            total_cost = 0
            total_overlap_cost = 0

            for query in queries:
                cost = 0
                overlapped_leaf_ids = self.query_single(query, using_rtree_filter, False, redundant_partitions)
                for nid in overlapped_leaf_ids:
                    if nid >= 0:
                        cost += self.nid_node_dict[nid].node_size
                        total_overlap_cost += self.nid_node_dict[nid].node_size // data_threshold
                    else:
                        cost += (-nid) # redundant partition cost?
                total_cost += cost
            return total_cost // len(queries), total_cost, total_overlap_cost // len(queries), total_overlap_cost
        

        def get_pid_for_data_point(self, point):
            '''
            get the corresponding leaf partition nid for a data point
            point: [dim1_value, dim2_value...], contains the same dimenions as the partition tree
            '''
            return self.__find_resided_partition(self.pt_root, point)
        
        def add_node(self, parent_id, child_node):
            child_node.nid = self.node_count
            self.node_count += 1
            
            child_node.pid = parent_id
            self.nid_node_dict[child_node.nid] = child_node
            
            child_node.depth = self.nid_node_dict[parent_id].depth + 1
            
            self.nid_node_dict[parent_id].children_ids.append(child_node.nid)
            self.nid_node_dict[parent_id].num_children += 1
            self.nid_node_dict[parent_id].is_leaf = False
        
        
        def apply_split(self, parent_nid, split_dim, split_value, split_type = 0, extended_bound = None, approximate = False,
                        pretend = False):
            '''
            split_type = 0: split a node into 2 sub-nodes by a given dimension and value, distribute dataset
            split_type = 1: split a node by bounding split (will create an irregular shape partition)
            split_type = 2: split a node by daul-bounding split (will create an irregular shape partition)
            split_type = 3: split a node by var-bounding split (multi MBRs), distribute dataset
            extended_bound is only used in split type 1
            approximate: used for measure query result size
            pretend: if pretend is True, return the split result, but do not apply this split
            '''
            parent_node = self.nid_node_dict[parent_nid]
            if pretend: # 是否只模拟切割但不应用
                parent_node = copy.deepcopy(self.nid_node_dict[parent_nid])
            
            child_node1, child_node2 = None, None
            
            if split_type == 0:
                
                #print("[Apply Split] Before split node", parent_nid, "node queryset:", parent_node.queryset, "MBRs:", parent_node.query_MBRs)
            
                # create sub nodes
                child_node1 = copy.deepcopy(parent_node)
                child_node1.boundary[split_dim + child_node1.num_dims] = split_value
                child_node1.children_ids = []

                child_node2 = copy.deepcopy(parent_node)
                child_node2.boundary[split_dim] = split_value
                child_node2.children_ids = []
                
                if parent_node.query_MBRs is not None:
                    MBRs1, MBRs2 = parent_node.split_query_MBRs(split_dim, split_value)
                    child_node1.query_MBRs = MBRs1
                    child_node2.query_MBRs = MBRs2
                    
                # if parent_node.dataset != None: # The truth value of an array with more than one element is ambiguous.
                # https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array
                if parent_node.dataset is not None:
                    child_node1.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] < split_value]
                    child_node1.node_size = len(child_node1.dataset)
                    child_node2.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] >= split_value]
                    child_node2.node_size = len(child_node2.dataset)

                if parent_node.queryset is not None:
                    left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
                    child_node1.queryset = left_part + mid_part
                    child_node2.queryset = right_part + mid_part
                    
                #print("[Apply Split] After split node", parent_nid, "left child queryset:", child_node1.queryset, "MBRs:", child_node1.query_MBRs)
                #print("[Apply Split] After split node", parent_nid, "right child queryset:", child_node2.queryset, "MBRs:", child_node2.query_MBRs)

                # update current node
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.nid_node_dict[parent_nid].split_type = "candidate cut"
            
            elif split_type == 1: # must reach leaf node, hence no need to maintain dataset and queryset any more
                
                child_node1 = copy.deepcopy(parent_node) # the bounding partition
                child_node2 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
                
                child_node1.is_leaf = True
                child_node2.is_leaf = True
                
                child_node1.children_ids = []
                child_node2.children_ids = []
                
                max_bound = None
                if extended_bound is not None:
                    max_bound = extended_bound
                else:
                    max_bound = parent_node._PartitionNode__max_bound(parent_node.queryset)
                child_node1.boundary = max_bound
                child_node2.is_irregular_shape = True
                
                bound_size = parent_node.query_result_size(max_bound, approximate = False)
                remaining_size = parent_node.node_size - bound_size           
                child_node1.node_size = bound_size
                child_node2.node_size = remaining_size
                
                child_node1.partitionable = False
                child_node2.partitionable = False
                
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "sole-bounding split"
            
            elif split_type == 2: # must reach leaf node, hence no need to maintain dataset and queryset any more
                
                child_node1 = copy.deepcopy(parent_node) # the bounding partition 1
                child_node2 = copy.deepcopy(parent_node) # the bounding partition 2
                child_node3 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
                
                child_node1.is_leaf = True
                child_node2.is_leaf = True
                child_node3.is_leaf = True
                
                child_node1.children_ids = []
                child_node2.children_ids = []
                child_node3.children_ids = []
                
                left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
                max_bound_1 = parent_node._PartitionNode__max_bound(left_part)
                max_bound_2 = parent_node._PartitionNode__max_bound(right_part)
                
                child_node1.boundary = max_bound_1
                child_node2.boundary = max_bound_2
                child_node3.is_irregular_shape = True          
                
                # Should we only consider the case when left and right cannot be further split? i.e., [b,2b)
                # this check logic is given in the PartitionAlgorithm, not here, as the split action should be general
                naive_left_size = np.count_nonzero(parent_node.dataset[:,split_dim] < split_value)
                naive_right_size = parent_node.node_size - naive_left_size

                # get (irregular-shape) sub-partition size
                bound_size_1 = parent_node.query_result_size(max_bound_1, approximate)
                if bound_size_1 is None: # there is no query within the left 
                    bound_size_1 = naive_left_size # use the whole left part as its size
               
                bound_size_2 = parent_node.query_result_size(max_bound_2, approximate)
                if bound_size_2 is None: # there is no query within the right
                    bound_size_2 = naive_right_size # use the whole right part as its size
               
                remaining_size = parent_node.node_size - bound_size_1 - bound_size_2
                
                child_node1.node_size = bound_size_1
                child_node2.node_size = bound_size_2
                child_node3.node_size = remaining_size
                
                child_node1.partitionable = False
                child_node2.partitionable = False
                child_node3.partitionable = False
                
                if not pretend:
                    self.add_node(parent_nid, child_node1)
                    self.add_node(parent_nid, child_node2)
                    self.add_node(parent_nid, child_node3)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "dual-bounding split"
            
            elif split_type == 3: # new bounding split, create a collection of MBR partitions
                
                remaining_size = parent_node.node_size
                for MBR in parent_node.query_MBRs:
                    child_node = copy.deepcopy(parent_node)
                    child_node.is_leaf = True
                    child_node.children_ids = []
                    child_node.boundary = MBR.boundary
                    child_node.node_size = MBR.bound_size
                    child_node.partitionable = False
                    remaining_size -= child_node.node_size
                    child_node.dataset = self.__extract_sub_dataset(parent_node.dataset, child_node.boundary)
                    child_node.queryset = MBR.queries # no other queries could overlap this MBR, or it's invalid
                    child_node.query_MBRs = [MBR]
                        
                    if not pretend:
                        self.add_node(parent_nid, child_node)
                
                # the last irregular shape partition, we do not need to consider its dataset
                child_node = copy.deepcopy(parent_node)
                child_node.is_leaf = True
                child_node.children_ids = []
                child_node.is_irregular_shape = True
                child_node.node_size = remaining_size
                child_node.partitionable = False
                child_node.dataset = None
                child_node.queryset = None
                child_node.query_MBRs = None
                
                if not pretend:
                    self.add_node(parent_nid, child_node)
                    self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                    self.nid_node_dict[parent_nid].split_type = "var-bounding split"
            
            else:
                print("Invalid Split Type!")
            
            if not pretend:
                del self.nid_node_dict[parent_nid].dataset
                del self.nid_node_dict[parent_nid].queryset
                #del self.nid_node_dict[parent_nid].query_MBRs
                #self.nid_node_dict[parent_nid] = parent_node
                
            return child_node1, child_node2
        
        def get_leaves(self, use_partitionable = False):
            nodes = []
            if use_partitionable:
                for nid, node in self.nid_node_dict.items():
                    if node.is_leaf and node.partitionable: # 是否只获取可分区的叶子节点
                        nodes.append(node)
            else:
                for nid, node in self.nid_node_dict.items():
                    if node.is_leaf:
                        nodes.append(node)
            return nodes
        
        def visualize(self, dims = [0, 1], queries = [], path = None, focus_region = None, add_text = True, use_sci = False):
            '''
            visualize the partition tree's leaf nodes
            focus_region: in the shape of boundary
            '''
            if len(dims) == 2:
                self.__visualize_2d(dims, queries, path, focus_region, add_text, use_sci)
            else:
                self.__visualize_3d(dims[0:3], queries, path, focus_region)
            
        
        # = = = = = internal functions = = = = =
        
        def __extract_sub_dataset(self, dataset, query):
            constraints = []
            num_dims = self.pt_root.num_dims
            for d in range(num_dims):
                constraint_L = dataset[:,d] >= query[d]
                constraint_U = dataset[:,d] <= query[num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            sub_dataset = dataset[constraint]
            return sub_dataset
        
        # 递归生成给定节点的所有子节点列表（包括该节点）
        def __generate_node_list(self, node):
            '''
            recursively add childrens into the list
            '''
            node_list = [node]
            for nid in node.children_ids:
                node_list += self.__generate_node_list(self.nid_node_dict[nid])
            return node_list
        
        # 将节点列表的节点转换为属性列表保存
        def __serialize(self, node_list):
            '''
            convert object to attributes to save
            '''
            serialized_node_list = []
            for node in node_list:
                # follow the same order of attributes in partition class
                attributes = [node.num_dims]
                #attributes += node.boundary
                if isinstance(node.boundary, list):
                    attributes += node.boundary
                else:
                    attributes += node.boundary.tolist()
                attributes.append(node.nid) # node id = its ow id
                attributes.append(node.pid) # parent id
                attributes.append(1 if node.is_irregular_shape_parent else 0)
                attributes.append(1 if node.is_irregular_shape else 0)
                attributes.append(node.num_children) # number of children
                #attributes += node.children_ids
                attributes.append(1 if node.is_leaf else 0)
                attributes.append(node.node_size)
                
                serialized_node_list.append(attributes)
            return serialized_node_list
        
        def __build_tree_from_serialized_node_list(self, serialized_node_list):
            
            self.pt_root = None
            self.nid_node_dict.clear()
            pid_children_ids_dict = {}  # 保存父节点到子节点的映射关系
            
            for serialized_node in serialized_node_list:
                num_dims = int(serialized_node[0])
                boundary = serialized_node[1: 1+2*num_dims]
                nid = int(serialized_node[1+2*num_dims]) # node id
                pid = int(serialized_node[2+2*num_dims]) # parent id
                is_irregular_shape_parent = False if serialized_node[3+2*num_dims] == 0 else True
                is_irregular_shape = False if serialized_node[4+2*num_dims] == 0 else True
                num_children = int(serialized_node[5+2*num_dims])
                # children_ids = []
                # if num_children != 0:
                #     children_ids = serialized_node[1+5+2*num_dims: 1+num_children+1+5+2*num_dims] # +1 for the end exclusive
                # is_leaf = False if serialized_node[1+num_children+5+2*num_dims] == 0 else True
                # node_size = serialized_node[2+num_children+5+2*num_dims] # don't use -1 in case of match error
                is_leaf = False if serialized_node[6+2*num_dims] == 0 else True
                node_size = int(serialized_node[7+2*num_dims])
                
                node = PartitionNode(num_dims, boundary, nid, pid, is_irregular_shape_parent, 
                                     is_irregular_shape, num_children, [], is_leaf, node_size) # let the children_ids empty
                self.nid_node_dict[nid] = node # update dict
                
                if node.pid in pid_children_ids_dict:
                    pid_children_ids_dict[node.pid].append(node.nid)
                else:
                    pid_children_ids_dict[node.pid] = [node.nid]
            
            # make sure the irregular shape partition is placed at the end of the child list
            for pid, children_ids in pid_children_ids_dict.items():
                if pid == -1:
                    continue
                if self.nid_node_dict[pid].is_irregular_shape_parent and not self.nid_node_dict[children_ids[-1]].is_irregular_shape:
                    # search for the irregular shape partition
                    new_children_ids = []
                    irregular_shape_id = None
                    for nid in children_ids:
                        if self.nid_node_dict[nid].is_irregular_shape:
                            irregular_shape_id = nid
                        else:
                            new_children_ids.append(nid)
                    new_children_ids.append(irregular_shape_id)
                    self.nid_node_dict[pid].children_ids = new_children_ids
                else:
                    self.nid_node_dict[pid].children_ids = children_ids
            
            self.pt_root = self.nid_node_dict[0]
        
        def __bound_query_by_boundary(self, query, boundary):
            '''
            bound the query by a node's boundary
            '''
            bounded_query = query.copy()
            num_dims = self.pt_root.num_dims
            for dim in range(num_dims):
                bounded_query[dim] = max(query[dim], boundary[dim])
                bounded_query[num_dims+dim] = min(query[num_dims+dim], boundary[num_dims+dim])
            return bounded_query
        
        # 查找包含给定数据点的叶子节点ID（唯一
        def __find_resided_partition(self, node, point):
            '''
            for data point only
            '''
            #print("enter function!")
            if node.is_leaf:
                #print("within leaf",node.nid)
                if node.is_contain(point):
                    return node.nid
            
            for nid in node.children_ids:
                if self.nid_node_dict[nid].is_contain(point):
                    #print("within child", nid, "of parent",node.nid)
                    return self.__find_resided_partition(self.nid_node_dict[nid], point)
            
            #print("no children of node",node.nid,"contains point")
            return -1
        
        # 给出分区树中与查询重叠的分区节点
        def __find_overlapped_partition(self, node, query, using_rtree_filter = False, print_info = False):
            
            if print_info:
                print("Enter node", node.nid)
                
            if node.is_leaf:
                if print_info:
                    print("node", node.nid, "is leaf")
                
                if using_rtree_filter and node.rtree_filters is not None:
                    for mbr in node.rtree_filters:
                        if node._PartitionNode__is_overlap(mbr, query) > 0:
                            return [node.nid]
                    return []
                else:
                    if print_info and node.is_overlap(query) > 0:
                        print("node", node.nid, "is added as result")
                    return [node.nid] if node.is_overlap(query) > 0 else []
            
            node_id_list = []
            if node.is_overlap(query) <= 0:
                if print_info:
                    print("node", node.nid, "is not overlap with the query")
                pass
            elif node.is_irregular_shape_parent: # special process for irregular shape partitions!
                if print_info:
                    print("node", node.nid, "is_irregular_shape_parent")
                    
                # bound the query with parent partition's boundary, that's for the inside case determination
                bounded_query = self.__bound_query_by_boundary(query, node.boundary)
                
                overlap_irregular_shape_node_flag = True
                for nid in node.children_ids[0: -1]: # except the last one, should be the irregular shape partition
                    overlap_case = self.nid_node_dict[nid].is_overlap(bounded_query)
                    if overlap_case == 2: # inside
                        node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
                        overlap_irregular_shape_node_flag = False
                        if print_info:
                            print("query within children node", nid, "irregular shape neighbors")
                        break
                    if overlap_case == 1: # overlap
                        node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
                        overlap_irregular_shape_node_flag = True
                        if print_info:
                            print("query overlap children node", nid, "irregular shape neighbors")
                if overlap_irregular_shape_node_flag:
                    if print_info:
                        print("query overlap irregular shape child node", node.children_ids[-1])
                    node_id_list.append(node.children_ids[-1])
            else:
                if print_info:
                    print("searching childrens for node", node.nid)
                for nid in node.children_ids:
                    node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter, print_info)
            return node_id_list
        
        def __visualize_2d(self, dims, queries = [], path = None, focus_region = None, add_text = True, use_sci = False):
            '''
            focus_region: in the shape of boundary
            '''
            fig, ax = plt.subplots(1)
            num_dims = self.pt_root.num_dims
            
            plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
            plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])
            
            
            leaves = self.get_leaves()
            for leaf in leaves: 
                lower1 = leaf.boundary[dims[0]]
                lower2 = leaf.boundary[dims[1]]             
                upper1 = leaf.boundary[dims[0]+num_dims]
                upper2 = leaf.boundary[dims[1]+num_dims]

                rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='g',linewidth=1)
                if add_text:
                    ax.text(lower1, lower2, leaf.nid, fontsize=7)
                ax.add_patch(rect)
            
            case = 0
            for query in queries:
                lower1 = query[dims[0]]
                lower2 = query[dims[1]]  
                upper1 = query[dims[0]+num_dims]
                upper2 = query[dims[1]+num_dims]    

                rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='r',linewidth=1)
                if add_text:
                    ax.text(upper1, upper2, case, color='b',fontsize=7)
                case += 1
                ax.add_patch(rect)

            ax.set_xlabel('dimension 1', fontsize=15)
            ax.set_ylabel('dimension 2', fontsize=15)
            if use_sci:
                plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
            #plt.xticks(np.arange(0, 400001, 100000), fontsize=10)
            #plt.yticks(np.arange(0, 20001, 5000), fontsize=10)

            plt.tight_layout() # preventing clipping the labels when save to pdf
            if focus_region is not None:
                
                # reform focus region into interleaf format
                formated_focus_region = []
                for i in range(2):
                    formated_focus_region.append(focus_region[i])
                    formated_focus_region.append(focus_region[2+i])
                
                plt.axis(formated_focus_region)

            if path is not None:
                fig.savefig(path)

            plt.show()
        
        # %matplotlib notebook
        def __visualize_3d(self, dims, queries = [], path = None, focus_region = None):
            fig = plt.figure()
            ax = Axes(fig)
            
            num_dims = self.pt_root.num_dims
            plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
            plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])
            ax.set_zlim(self.pt_root.boundary[dims[2]], self.pt_root.boundary[dims[2]+num_dims])
            
            leaves = self.get_leaves()
            for leaf in leaves:
                
                L1 = leaf.boundary[dims[0]]
                L2 = leaf.boundary[dims[1]]
                L3 = leaf.boundary[dims[2]]      
                U1 = leaf.boundary[dims[0]+num_dims]
                U2 = leaf.boundary[dims[1]+num_dims]
                U3 = leaf.boundary[dims[2]+num_dims]
                
                # the 12 lines to form a rectangle
                x = [L1, U1]
                y = [L2, L2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="g")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="g")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="g")
                y = [L2, L2]
                ax.plot3D(x,y,z,color="g")

                x = [L1, L1]
                y = [L2, U2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="g")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="g")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="g")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="g")

                x = [L1, L1]
                y = [L2, L2]
                z = [L3, U3]
                ax.plot3D(x,y,z,color="g")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="g")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="g")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="g")
            
            for query in queries:

                L1 = query[dims[0]]
                L2 = query[dims[1]]
                L3 = query[dims[2]]
                U1 = query[dims[0]+num_dims]
                U2 = query[dims[1]+num_dims]
                U3 = query[dims[2]+num_dims]

                # the 12 lines to form a rectangle
                x = [L1, U1]
                y = [L2, L2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="r")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="r")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="r")
                y = [L2, L2]
                ax.plot3D(x,y,z,color="r")

                x = [L1, L1]
                y = [L2, U2]
                z = [L3, L3]
                ax.plot3D(x,y,z,color="r")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="r")
                z = [U3, U3]
                ax.plot3D(x,y,z,color="r")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="r")

                x = [L1, L1]
                y = [L2, L2]
                z = [L3, U3]
                ax.plot3D(x,y,z,color="r")
                x = [U1, U1]
                ax.plot3D(x,y,z,color="r")
                y = [U2, U2]
                ax.plot3D(x,y,z,color="r")
                x = [L1, L1]
                ax.plot3D(x,y,z,color="r")

            if path is not None:
                fig.savefig(path)

            plt.show()