

# 用于维护切割信息
class SplitContext:
    '''
    to maintain the split information
    split_type:
    0 = candidate cut 候选切割
    1 = bounding split 边界切割
    2 = dual-bounding split 双边界切割
    3 = new bounding split 新边界切割
    '''
    def __init__(self, split_type, split_dimension, split_value):
        self.split_type = split_type    # 切割类型
        self.split_dimension = split_dimension  # 切割维度
        self.split_value = split_value  # 切割值
        
        self.sub_partitions = []
        self.split_gain = 0 # 切割增益