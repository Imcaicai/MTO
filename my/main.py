import numpy as np
from numpy import genfromtxt
from dataset_and_queryset_helper import DatasetAndQuerysetHelper
from partition_algorithm import PartitionAlgorithm

def tpch_q1(scale):
    used_dims = [5]
    block_size = 10000
    table_path = './experiments/lineitem.tbl'
    data_path = './experiments/tpch/Q1/' + str(scale) +'_data.csv'
    domain_path = './experiments/tpch/Q1/' + str(scale) +'_domain.csv'
    train_path = './experiments/tpch/Q1/' + str(scale) +'_train.csv'
    test_path = './experiments/tpch/Q1/' + str(scale) +'_test.csv'
    tree_path = './experiments/tpch/Q1/' + str(scale) +'qdtree'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    # helper.generate_dataset_and_save(table_path, data_path, domain_path)  # generate dataset
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]

    training_set, testing_set = helper.generate_queryset_and_save(domain_path, 100, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')

    pa_qd = PartitionAlgorithm()
    pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
    pa_qd.partition_tree.save_tree(tree_path)
    pa_qd.partition_tree.evaluate_query_cost(queries = testing_set, print_result = True, cost_path = tree_path + '_cost.txt')


def tpch_q6(scale):
    used_dims = [4, 5, 6]
    block_size = 10000
    table_path = './experiments/lineitem.tbl'
    data_path = './experiments/tpch/Q6/' + str(scale) +'_data.csv'
    domain_path = './experiments/tpch/Q6/' + str(scale) +'_domain.csv'
    train_path = './experiments/tpch/Q6/' + str(scale) +'_train.csv'
    test_path = './experiments/tpch/Q6/' + str(scale) +'_test.csv'
    tree_path = './experiments/tpch/Q6/' + str(scale) +'qdtree'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    # helper.generate_dataset_and_save(table_path, data_path, domain_path)  # generate dataset
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]

    training_set, testing_set = helper.generate_queryset_and_save(domain_path, 100, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')

    pa_qd = PartitionAlgorithm()
    pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
    pa_qd.partition_tree.save_tree(tree_path)
    pa_qd.partition_tree.evaluate_query_cost(queries = testing_set, print_result = True, cost_path = tree_path + '_cost.txt')


def tpch_q4(scale):
    
    block_size = 10000
    table_lineitem_path = './experiments/lineitem.tbl'
    table_orders_path = './experiments/orders.tbl'
    data_orders_path = './experiments/tpch/Q4/' + str(scale) +'_data_orders.csv'
    data_lineitem_path = './experiments/tpch/Q4/' + str(scale) +'_data_lineitem.csv'
    domain_orders_path = './experiments/tpch/Q4/' + str(scale) +'_domain_orders.csv'
    domain_lineitem_path = './experiments/tpch/Q4/' + str(scale) +'_domain_lineitem.csv'

    # mto
    used_dims = [3,7]
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    # helper.generate_dataset_and_save_q4(table_lineitem_path, table_orders_path, data_lineitem_path, domain_lineitem_path)  # generate dataset
    dataset, domains = helper.load_dataset(used_dims, data_lineitem_path, domain_lineitem_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    train_path = './experiments/tpch/Q4/' + str(scale) +'_train.csv'
    test_path = './experiments/tpch/Q4/' + str(scale) +'_test.csv'
    training_set, testing_set = helper.generate_queryset_and_save(domain_lineitem_path, 100, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')

    cost_path = './experiments/tpch/Q4/' + str(scale) +'_cost_mto.txt'
    tree_path = './experiments/tpch/Q4/' + str(scale) +'_mto'
    pa_qd = PartitionAlgorithm()
    build_time = pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
    pa_qd.partition_tree.save_tree(tree_path)
    average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_qd.partition_tree.evaluate_query_cost(queries = testing_set, print_result = True)
    with open(cost_path, 'w') as file:
        file.write("Method build time: {}\n".format(build_time))
        file.write("Average logical IOs: {}\n".format(average_cost))
        file.write("Total logical IOs: {}\n".format(2*total_cost))
        file.write("Average blocks accessed: {}\n".format(average_overlap_cost))
        file.write("Total blocks accessed: {}\n".format(2*total_overlap_cost))

    

    
    # helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    # helper.generate_dataset_and_save_q4(table_lineitem_path, table_order_path, data_path, domain_path)  # generate dataset
    # dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    # boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]

    # training_set, testing_set = helper.generate_queryset_and_save(domain_path, 100, queryset_type=2)
    # np.savetxt(train_path, training_set, delimiter=',')
    # np.savetxt(test_path, testing_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')

    # pa_qd = PartitionAlgorithm()
    # build_time = pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
    # pa_qd.partition_tree.save_tree(tree_path)
    # average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_qd.partition_tree.evaluate_query_cost(queries = testing_set, print_result = True)


if __name__ == "__main__":
    scale = 1
    tpch_q4(scale)