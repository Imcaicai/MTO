import numpy as np
from numpy import genfromtxt
from dataset_and_queryset_helper import DatasetAndQuerysetHelper
from partition_algorithm import PartitionAlgorithm


def main():
    # 执行初始化操作
    initialize_app()

    # 启动应用程序
    start_app()

def initialize_app():
    # 执行初始化操作，例如加载配置、连接数据库等
    pass

def start_app():
    # 启动应用程序，例如创建 Web 服务器、运行任务等
    pass

if __name__ == "__main__":
    # = = = consider the scale factor 1 TPC-H dataset = = = 
    used_dims = [1,2,3,4]
    block_size = 10000
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor=1) # set the base path
    # helper.generate_dataset_and_save('./experiments/lineitem.tbl')  # generate dataset
    dataset, domains = helper.load_dataset(used_dims)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    print(domains)

    num_dims = 4
    # training_set, testing_set = helper.generate_queryset_and_save(100, queryset_type=2)
    # np.savetxt("./experiments/queryset/prob2/scale1/train.csv", training_set, delimiter=',')
    # np.savetxt("./experiments/queryset/prob2/scale1/test.csv", testing_set, delimiter=',')
    training_set = genfromtxt("./experiments/queryset/prob2/scale1/train.csv", delimiter=',')
    testing_set = genfromtxt("./experiments/queryset/prob2/scale1/test.csv", delimiter=',')
    extended_training_set = helper.extend_queryset(training_set)


    # pa_kd = PartitionAlgorithm()
    # pa_kd.InitializeWithKDT(len(boundary)//2, boundary, dataset, data_threshold = block_size) # 447
    # pa_kd.partition_tree.visualize(queries = training_set)
    # # pa_kd.partition_tree.save_tree('C:/Users/Cloud/iCloudDrive/NORA_experiments/partition_layout/prob2_kdtree_scale10')\
    # pa_kd.partition_tree.evaluate_query_cost(testing_set, True)
        
    pa_qd = PartitionAlgorithm()
    pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
    # pa_qd.partition_tree.visualize(queries = training_set)
    pa_qd.partition_tree.save_tree('./experiments/partition_layout/prob2_qdtree_scale1')
    pa_qd.partition_tree.evaluate_query_cost(testing_set, True)