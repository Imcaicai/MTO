import random
import time

import numpy as np
import pandas as pd


class DatasetAndQuerysetHelper:
    '''
    naming:
    dataset: [base_path]/dataset/lineitem_[scale_factor]_[prob_threshold].csv
    domain: [base_path]/dataset/lineitem_[scale_factor]_[prob_threshold]_domains.csv
    queryset: [base_path]/queryset/[prob]/[vary_item]/[vary_val]_[used_dimensions]_[distribution/random].csv
    '''    
    def __init__(self, used_dimensions = None, scale_factor = 100, base_path = './experiments',
                prob_id = 1, vary_id = 0, vary_val = 0, train_percent = 0.5, random_percent = 0):
        
        self.used_dimensions = used_dimensions # i.e., [1,2,3,4]
        self.type = type
        self.total_dims = 16
        self.domain_dims = 12
        
        self.scale_factor = scale_factor
        self.prob_threshold = 1 / self.scale_factor # the probability of an original record being sampled into this dataset
        self.block_size = 1000000 // self.scale_factor # in original file, 1M rows take approximately 128MB
        
        self.base_path = base_path
        
        self.vary_items = ['default', 'alpha', 'num_dims', 'prob_dims', 'num_X']
        self.vary_id = vary_id
        self.vary_val = vary_val
        
        self.prob_id = prob_id
        self.query_base_path = self.base_path + '/queryset/prob' + str(self.prob_id) + '/' + self.vary_items[vary_id] + '/'
        self.query_file_name = str(vary_val) + '_' + str(self.used_dimensions) # dependent on used_dimensions, so change dim first
        
        self.query_distribution_path = self.query_base_path + self.query_file_name + '_distribution.csv'
        self.query_random_path = self.query_base_path + self.query_file_name + '_random.csv'
        
        self.train_percent = train_percent
        
        # the following are default query generation settings
        self.random_percent = random_percent # usef for query generation
        self.cluster_center_amount = 10
        self.maximum_range_percent = 1 # 20% of the corresponding domain
        self.sigma_percent = 0.2 # control the differences in a cluster
        
        self.QDistThreshold_percent = 0.001 # distance threshold, 1% of the corresponding domain
        self.maximum_X = 5 # used for the 1-X train test case
        
    
    # = = = = = public functions (API) = = = = =
    def load_dataset(self, used_dimensions = [], data_path = None, domain_path = None):
        '''
        the priority of the used_dimensions argument in the function is higher than the saved attribute version
        domains: [[L1, U1], [L2, U2],...]
        return the dataset projected on selected dimensions
        '''
        dataset = np.genfromtxt(data_path, delimiter=',') # the sampled subset
        domains = np.genfromtxt(domain_path, delimiter=',') # the domain of that scale
        if used_dimensions != []:
            dataset = dataset[:,used_dimensions]
            domains = domains[used_dimensions]
        elif self.used_dimensions is not None:
            dataset = dataset[:,self.used_dimensions]
            domains = domains[self.used_dimensions]
        return dataset, domains
    
    def load_queryset(self, return_train_test = True, query_distribution_path = None, query_random_path = None):
        '''
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        how about the used dimension?
        return the saved queryset, should be projected on selected dimensions.
        '''
        # embed used_dimension info into query file's name
        # when load, will auto matically check whether used_dimension is matched!!! or load will failed
        
        distribution_query, random_query = None, None
        
        if query_distribution_path is not None and query_random_path is not None:
            distribution_query = np.genfromtxt(query_distribution_path, delimiter=',')
            random_query = np.genfromtxt(query_random_path, delimiter=',')
        else:
            distribution_query = np.genfromtxt(self.query_distribution_path, delimiter=',')
            random_query = np.genfromtxt(self.query_random_path, delimiter=',')
        
        if return_train_test:
            training_set, testing_set = self.__convert_to_train_test(distribution_query, random_query)
            return training_set, testing_set
        else:
            return distribution_query, random_query
    
    def generate_dataset_and_save(self, table_path, data_path, domain_path, chunk_size = 100000):
        '''
        refer to TPCH tools to generate the original dataset (.tbl)
        this function is used to process the .tbl file with given sampling rate to generate a .csv file
        consider the possible table size, this function is implemented in a batch processing manner
        '''
        sampled_subset = []
        domains = [[float('Infinity'), float('-Infinity')] for i in range(self.domain_dims)] # indicate min, max
        
        col_names = ['_c'+str(i) for i in range(self.total_dims)]
        cols = [i for i in range(self.total_dims)]

        start_time = time.time()
        
        batch_count = 0
        for chunk in pd.read_table(table_path, delimiter='|', usecols=cols, names=col_names, chunksize=chunk_size):
            print('current chunk: ', batch_count)
            chunk.apply(lambda row: self.__process_chunk_sampling(row, domains, sampled_subset), axis=1)
            batch_count += 1

        end_time = time.time()
        print('total processing time: ', end_time - start_time)
        
        sampled_subset = np.array(sampled_subset)
        domains = np.array(domains)
        np.savetxt(data_path, sampled_subset, delimiter=',')
        np.savetxt(domain_path, domains, delimiter=',')

    
    def generate_queryset_and_save(self, domain_path, query_amount, queryset_type = 0, dim_prob = [], prob_id = 1, vary_id = 0, vary_val = 0, 
                                   return_train_test = True):
        '''
        generate queryset for given dimensions.
        query_amount: total query amount, including distribution queries and random queries, or training set and testing set
        
        queryset_type = 0: typical (old) queryset generator, generate distribution and random queries
        queryset_type = 1: new NORA queryset generator, generate 1-1 train test which satisfy given distance threshold
        queryset_type = 2: new NORA queryset generator, generate 1-X train test which satisfy given distance threshold
        queryset_type = 3: hybrid workload, based on type 2, but random percent are random queries
        queryset_type = 4: Mixture Gaussian, but satisfy distance threshold
        
        dim_prob: the probability of using a given dimension (in used_dimensions) in a query
        other configurations are stored in class attributes
        REMEMBER to change the used_dimensions first if not using the previous one !!!
        
        the returned queries are not numpy object by default
        '''
        
        domains = np.genfromtxt(domain_path, delimiter=',')[self.used_dimensions]
        if dim_prob == []: # by default, use all the selected dimensions
            dim_prob = [1 for i in range(len(self.used_dimensions))]  
        maximum_range = [(domains[i,1] - domains[i,0]) * self.maximum_range_percent for i in range(len(domains))]
            
        if queryset_type == 0:
        
            num_random_query = int(query_amount * self.random_percent)
            num_distribution_query = query_amount - num_random_query

            distribution_query = self.__generate_distribution_query(num_distribution_query, dim_prob, domains, maximum_range)
            random_query = self.__generate_random_query(num_random_query, dim_prob, domains, maximum_range)

            # # refresh query related class attributes
            # self.vary_id = vary_id
            # self.vary_val = vary_val      
            # self.query_base_path = self.base_path + '/queryset/prob' + str(prob_id) + '/' + self.vary_items[vary_id] + '/'
            # self.query_file_name = str(vary_val) + '_' + str(self.used_dimensions)
            # self.query_distribution_path = self.query_base_path + self.query_file_name + '_distribution.csv'
            # self.query_random_path = self.query_base_path + self.query_file_name + '_random.csv'

            # # save
            # np.savetxt(self.query_distribution_path, distribution_query, delimiter=',')
            # np.savetxt(self.query_random_path, random_query, delimiter=',')

            # print(" = = = distribution query = = = ")
            # print(distribution_query)

            # print(" = = = random query = = = ")
            # print(random_query)

            if return_train_test:
                training_set, testing_set = self.__convert_to_train_test(distribution_query, random_query)
                return training_set, testing_set
            else:
                return distribution_query, random_query
            
        elif queryset_type == 1:
            num_training_query = query_amount // 2
            training_set = self.__generate_new_training_set(num_training_query, domains, maximum_range)
            testing_set = self.__generate_new_testing_set(training_set, domains, maximum_X = 1)
            # TODO save it
            return training_set, testing_set
            
        elif queryset_type == 2: # in this case, the total query amount are not fixed
            num_training_query = query_amount // 2
            training_set = self.__generate_new_training_set(num_training_query, domains, maximum_range)
            testing_set = self.__generate_new_testing_set(training_set, domains, maximum_X = self.maximum_X)
            # TODO save it
            return training_set, testing_set
        
        elif queryset_type == 3: # hybrid workload, include some random queries based on type 2
            num_random_query = int(query_amount * self.random_percent)
            num_distribution_query = query_amount - num_random_query
            
            num_training_query = num_distribution_query // 2
            num_testing_query = num_training_query
            
            random_query = self.__generate_random_query(num_random_query, dim_prob, domains, maximum_range)
            training_set = self.__generate_new_training_set(num_training_query, domains, maximum_range)
            testing_set = self.__generate_new_testing_set(training_set, domains, maximum_X = self.maximum_X)
            
            train_random = random_query[0:int(self.train_percent*len(random_query))]
            test_random = random_query[int(self.train_percent*len(random_query)):]
            
            training_set = training_set + train_random
            testing_set = testing_set + test_random
            
            return training_set, testing_set
        
        elif queryset_type == 4:
            num_training_query = query_amount // 2
            training_set = self.__generate_distribution_query(num_training_query, dim_prob, domains, maximum_range)
            testing_set = self.__generate_new_testing_set(training_set, domains, maximum_X = self.maximum_X)
            return training_set, testing_set
            
        else:
            print("No supported queryset type!")
            return None, None

    def real_result_size(self, dataset, queries):
        num_dims = int(len(self.used_dimensions))
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
    
    def queryset_distance(self, queryset1, queryset2):
        '''
        estimate the single direction Hausdorff distance
        '''
        domains = np.genfromtxt(self.save_path_domain, delimiter=',')[self.used_dimensions]
        num_dims = len(self.used_dimensions)
        if len(queryset1[0]) == len(queryset2[0]):
            num_dims = len(queryset1[0]) // 2
            
        max_dist = 0
        for query1 in queryset1:
            min_dist = float('inf')
            for query2 in queryset2:
                dist = 0
                for k in range(num_dims):
                    dist_L = abs(query1[k]-query2[k])
                    dist_U = abs(query1[k+num_dims]-query2[k+num_dims])
                    dist_L_ratio = dist_L / (domains[k][1] - domains[k][0])
                    dist_U_ratio = dist_U / (domains[k][1] - domains[k][0])
                    dist = max(dist, dist_L_ratio, dist_U_ratio)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > max_dist:
                max_dist = min_dist
        
        return max_dist
    

    def __process_chunk_sampling(self, row, domains, sampled_subset):
        prob = random.uniform(0, 1) # 采样概率
        row_numpy = row.to_numpy()  
        col_idx = [0,1,2,3,4,5,6,7,9,11,13,15]
        row_numpy = row_numpy[col_idx]

        for i in range(len(domains)):
            if row_numpy[i] > domains[i][1]:
                domains[i][1] = row_numpy[i]
            if row_numpy[i] < domains[i][0]:
                domains[i][0] = row_numpy[i]
        if prob <= self.prob_threshold:    
            sampled_subset.append(row_numpy.tolist())


    def __generate_new_testing_set(self, training_set, domains, maximum_X = 1):
        '''
        generate the testing set for 1-1 or 1-X train test scenarios (i.e., new NORA) which satisfy distance threshold
        
        the maximum_X denotes the X for 1-X, if it is 1, it indicate the 1-1 case, else it indicate the 1-X case
        '''
        extended_values = [(domain[1]-domain[0]) * self.QDistThreshold_percent for domain in domains]
        EV = np.array(extended_values)
        
        testing_set = []
        for query in training_set:
            num_X = 1 if maximum_X == 1 else random.randint(0, maximum_X) # inclusive for both end
            for i in range(num_X):
                query_lower, query_upper = [], []
                for k in range(len(domains)):
                    QL = query[k] - random.uniform(0, EV[k])
                    QU = query[len(domains)+k] + random.uniform(0, EV[k])
                    if QL <= domains[k][0]:
                        QL = domains[k][0]
                    if QU >= domains[k][1]:
                        QU = domains[k][1]
                    if QL > QU:
                        QL, QU = QU, QL
                    query_lower.append(QL)
                    query_upper.append(QU)
                testing_set.append(query_lower+query_upper)
        return testing_set
    
    def __generate_new_training_set(self, query_amount, domains, maximum_range):
        '''
        generate the training set for 1-1 or 1-X train test scenarios (i.e., new NORA)
        query_amount: number of training queries to generate
        '''
        # first, generate the centers of each query
        centers = []
        for i in range(query_amount):
            center = [] # [D1, D2,..., Dk]
            for k in range(len(domains)):
                ck = random.uniform(domains[k][0], domains[k][1])
                center.append(ck)
            centers.append(center)
        
        # second, generate expected range for each dimension for each center (of each query)
        centers_ranges = []
        for i in range(query_amount):
            ranges = [] # the range in all dimensions for a given center
            for k in range(len(domains)):
                ran = random.uniform(0, maximum_range[k])
                ranges.append(ran)
            centers_ranges.append(ranges)
            
        # third, build the queries and bound them by the domain
        generated_queries = []
        for i in range(query_amount):
            center = centers[i]
            query_lower, query_upper = [], []
            for k in range(len(domains)):
                query_range = centers_ranges[i][k]
                L = center[k] - query_range/2
                U = center[k] + query_range/2
                if L <= domains[k][0]:
                    L = domains[k][0]
                if U >= domains[k][1]:
                    U = domains[k][1]
                if L > U:
                    L, U = U, L
                query_lower.append(L)
                query_upper.append(U)
            generated_queries.append(query_lower+query_upper)
            
        return generated_queries
    

    def extend_queryset(self, queries, QDistThreshold_percent = None, domains = None):
        '''
        extend the provided queryset with the previous provided query distance threshold
        '''
        extended_queries = []
        
        if QDistThreshold_percent is None:
            QDistThreshold_percent = self.QDistThreshold_percent
                    
        num_dims = len(domains)
            
        extended_values = [(domain[1]-domain[0]) * QDistThreshold_percent for domain in domains]
        EV = np.array(extended_values)
        BL = [domain[0] for domain in domains]
        BU = [domain[1] for domain in domains]
         
        for query in queries:
            
            QL = np.array(query[0:num_dims])
            QU = np.array(query[num_dims:])
            
            QL -= EV # extended_values do not need to be converted to numpy
            QL = np.amax(np.array([QL.tolist(), BL]),axis=0).tolist()# bound it by the domain
            
            QU += EV
            QU = np.amin(np.array([QU.tolist(), BU]),axis=0).tolist() # bound it by the domain
            
            extended_query = QL + QU
            extended_queries.append(extended_query)
               
        return extended_queries


    def __convert_to_train_test(self, distribution_query, random_query):
        train_distribution = distribution_query[0:int(self.train_percent*len(distribution_query))]
        test_distribution = distribution_query[int(self.train_percent*len(distribution_query)):]
        train_random = random_query[0:int(self.train_percent*len(random_query))]
        test_random = random_query[int(self.train_percent*len(random_query)):]
        
        # to deal with the shape issue, 0 items cannot be concated
        if len(distribution_query) == 0 and len(random_query) == 0:
            return [], []
        elif len(distribution_query) == 0:
            return train_random, test_random
        elif len(random_query) == 0:
            return train_distribution, test_distribution
        else:
            training_set = np.concatenate((train_distribution, train_random), axis=0)
            testing_set = np.concatenate((test_distribution, test_random), axis=0)
            return training_set, testing_set
    
    def __generate_distribution_query(self, query_amount, dim_prob, domains, maximum_range):
        '''
        generate clusters of queries
        '''
        # first, generate cluster centers
        centers = []
        for i in range(self.cluster_center_amount):
            center = [] # [D1, D2,..., Dk]
            for k in range(len(domains)):
                ck = random.uniform(domains[k][0], domains[k][1])
                center.append(ck)
            centers.append(center)

        # second, generate expected range for each dimension for each center
        centers_ranges = []
        for i in range(self.cluster_center_amount):
            ranges = [] # the range in all dimensions for a given center
            for k in range(len(domains)):
                ran = random.uniform(0, maximum_range[k])
                ranges.append(ran)
            centers_ranges.append(ranges)

        # third, generate sigma for each dimension for each center
        centers_sigmas = []
        for i in range(self.cluster_center_amount):
            sigmas = []
            for k in range(len(domains)):
                sigma = random.uniform(0, maximum_range[k] * self.sigma_percent)
                sigmas.append(sigma)
            centers_sigmas.append(sigmas)

        # fourth, generate queries
        distribution_query = [] = []
        for i in range(query_amount):
            # choose a center
            center_index = random.randint(0, self.cluster_center_amount-1) # this is inclusive            
            query_lower, query_upper = [], []
            for k in range(len(domains)):
                # consider whether or not to use this dimension
                L, U = None, None
                prob = random.uniform(0, 1)
                if prob > dim_prob[k]:
                    L = domains[k][0]
                    U = domains[k][1]
                else:
                    center = centers[center_index]
                    query_range = centers_ranges[center_index][k]
                    L = center[k] - query_range/2
                    U = center[k] + query_range/2
                    L = random.gauss(L, centers_sigmas[center_index][k])
                    U = random.gauss(U, centers_sigmas[center_index][k])
                    if L <= domains[k][0]:
                        L = domains[k][0]
                    if U >= domains[k][1]:
                        U = domains[k][1]
                    if L > U:
                        L, U = U, L
                query_lower.append(L)
                query_upper.append(U)
            distribution_query.append(query_lower + query_upper)
        return distribution_query
    
    def __generate_random_query(self, query_amount, dim_prob, domains, maximum_range):
        random_query = []
        for i in range(query_amount):
            query_lower, query_upper = [], []
            for k in range(len(domains)):     
                # consider whether or not to use this dimension
                L, U = None, None
                prob = random.uniform(0, 1)
                if prob > dim_prob[k]:
                    L = domains[k][0]
                    U = domains[k][1]
                else:
                    center = random.uniform(domains[k][0], domains[k][1])
                    query_range = random.uniform(0, maximum_range[k])
                    L = center - query_range/2
                    U = center + query_range/2
                    if L <= domains[k][0]:
                        L = domains[k][0]
                    if U >= domains[k][1]:
                        U = domains[k][1]
                query_lower.append(L)
                query_upper.append(U)
            random_query.append(query_lower + query_upper)
        return random_query