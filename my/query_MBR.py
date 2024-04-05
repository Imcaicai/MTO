
# 表示边界覆盖查询得到最小边界矩形
class QueryMBR:
    '''
    the MBR that bound overlapped queries
    '''
    def __init__(self, boundary, added_as_fist_query = True):
        self.num_dims = len(boundary) / 2
        self.boundary = boundary
        self.num_query = 1
        self.queries = []
        self.bound_size = None # number of records this MBR overlaps
        self.total_query_result_size = None # total query results size of all the queries inside this MBR
        self.query_result_size = [] # record each query's result size
        self.is_extended = False
        self.ill_extended = False
        if added_as_fist_query:
            self.queries = [copy.deepcopy(boundary)] # OR the change of boundary will affect the queries!
        
    def check_condition3(self, data_threshold):
        '''
        check whether this MBR satisfy the new bounding split condition 3:
        1. every query size > BP - b
        2. total_query_result_size + b > bound_size * num_query
        '''
        for size in self.query_result_size:
            if size <= self.bound_size - data_threshold:
                return False
        
        if self.total_query_result_size + data_threshold <= self.bound_size * self.num_query:
            return False
        
        return True