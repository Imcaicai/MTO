import numpy as np
from numpy import genfromtxt
import pandas as pd
import csv
from datetime import datetime
import time
from dataset_and_queryset_helper import DatasetAndQuerysetHelper
from partition_algorithm import PartitionAlgorithm




def convert_date_to_int(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return int(date_obj.strftime('%Y%m%d'))


def extract_columns(tbl_name, column_indices, column_date=None, scale_factor = 1):
    tbl_path = './experiments/TPC-H V3.0.1/dbgen/' + tbl_name + '.tbl'
    outfile_path = './experiments/tpch/' + str(scale_factor) + '/' + tbl_name + '.tbl'
    with open(tbl_path, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter='|')
        with open(outfile_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile, delimiter='|')
            for row in reader:
                extracted_row = [row[i] for i in column_indices]
                if column_date:
                    for date_index in column_date:
                        extracted_row[date_index] = convert_date_to_int(extracted_row[date_index])
                writer.writerow(extracted_row)


def merge_tbl_files(scale_factor):
    # 读取 tbl 文件并命名列名
    lineitem_cols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_quantity', 'l_discount', 'l_shipdate', 'l_commitdate', 'l_receiptdate']
    orders_cols = ['o_orderkey', 'o_orderdate']
    supplier_cols = ['s_suppkey', 's_nationkey']
    nation_cols = ['n_nationkey', 'n_regionkey']
    part_cols = ['p_partkey', 'p_retailprice']

    lineitem_path = './experiments/tpch/' + str(scale_factor) + '/lineitem.tbl' 
    orders_path = './experiments/tpch/' + str(scale_factor) + '/orders.tbl'
    supplier_path = './experiments/tpch/' + str(scale_factor) + '/supplier.tbl'
    nation_path = './experiments/tpch/' + str(scale_factor) + '/nation.tbl'
    part_path = './experiments/tpch/' + str(scale_factor) + '/part.tbl'  
    dataset_path = './experiments/tpch/' + str(scale_factor) + '/dataset.tbl'  
    
    lineitem_df = pd.read_table(lineitem_path, delimiter='|', usecols=range(8), names=lineitem_cols)
    orders_df = pd.read_table(orders_path, delimiter='|', usecols=range(2), names=orders_cols)
    supplier_df = pd.read_table(supplier_path, delimiter='|', usecols=range(2), names=supplier_cols)
    nation_df = pd.read_table(nation_path, delimiter='|', usecols=range(2), names=nation_cols)
    part_df = pd.read_table(part_path, delimiter='|', usecols=range(2), names=part_cols)
    
    # 使用 merge 函数按对应的 key 列连接
    merged_df = pd.merge(supplier_df, nation_df, left_on='s_nationkey', right_on='n_nationkey')
    print("merged: supplier, nation")
    merged_df = pd.merge(lineitem_df, merged_df, left_on='l_suppkey', right_on='s_suppkey')
    print("merged: lineitem, supplier, nation")
    merged_df = pd.merge(merged_df, part_df, left_on='l_partkey', right_on='p_partkey')
    print("merged: lineitem, supplier, nation, part")
    merged_df = pd.merge(merged_df, orders_df, left_on='l_orderkey', right_on='o_orderkey')
    print("merged: lineitem, supplier, nation, part, orders")
    # 将合并后的 DataFrame 存储为新的 tbl 文件
    merged_df.to_csv(dataset_path, sep='|', index=False, header=False)


def execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, type = 0):
    if type == 0 or type == 1:
        pa_qd = PartitionAlgorithm()
        build_time, node_count = pa_qd.InitializeWithQDT(training_set, len(boundary)//2, boundary, dataset, data_threshold = block_size)
        # pa_qd.partition_tree.save_tree(tree_path)
        start_time = time.time()
        if type == 0:   # 单表
            average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_qd.partition_tree.evaluate_query_cost(queries = testing_set, data_threshold = block_size)
        else:           # 多表
            average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_qd.partition_tree.evaluate_worst_query_cost(queries = testing_set, data_threshold = block_size)
    else:   # kd-tree: type = 2, 3
        pa_kd = PartitionAlgorithm()
        build_time, node_count = pa_kd.InitializeWithKDT(len(boundary)//2, boundary, dataset, data_threshold = block_size) 
        # pa_kd.partition_tree.save_tree(tree_path)
        start_time = time.time()
        if type == 2:   # 单表
            average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_kd.partition_tree.evaluate_query_cost(queries = testing_set, data_threshold = block_size)
        else:           # 多表
            average_cost, total_cost, average_overlap_cost, total_overlap_cost = pa_kd.partition_tree.evaluate_worst_query_cost(queries = testing_set, data_threshold = block_size)
    end_time = time.time()
    
    if type == 0 or type == 2:
        query_time = end_time - start_time
    else:
        query_time = (end_time - start_time) * len(testing_set)
    with open(cost_path, 'w') as file:
        file.write("Method build time: {}\n".format(build_time))
        file.write("Node count: {}\n".format(node_count))
        file.write("Query time: {}\n".format(query_time))
        file.write("Average logical IOs: {}\n".format(average_cost))
        file.write("Total logical IOs: {}\n".format(total_cost))
        file.write("Average blocks accessed: {}\n".format(average_overlap_cost))
        file.write("Total blocks accessed: {}\n".format(total_overlap_cost))



def tpch_q1(scale, block_size = 10000, query_amount = 100):
    '''
    table: lineitem
    column: l_shipdate
    '''
    used_dims = [5]
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q1_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q1_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/q1_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q1_cost_qdtree.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')
    # testing_set = helper.extend_queryset(training_set, 0, domains)
    np.savetxt(test_path, testing_set, delimiter=',')

    # qd-tree, mto
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)
    tree_path = './experiments/tpch/' + str(scale) +'/model/q1_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q1_cost_kdtree.txt'
    # kd-tree
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 2)



def tpch_q6(scale, block_size = 10000, query_amount = 100):
    '''
    table: lineitem
    column: l_quantity, l_discount, l_shipdate
    '''
    used_dims = [3,4,5]
    block_size = 10000
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q6_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q6_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/q6_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q6_cost_qdtree.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')

    # qd-tree, mto
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)
    tree_path = './experiments/tpch/' + str(scale) +'/model/q6_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q6_cost_kdtree.txt'
    # kd-tree
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 2)



def tpch_q4(scale, block_size = 10000, query_amount = 100):
    '''
    table: lineitem, orders
    column: l_commitdate, l_receiptdate, o_orderdate
    '''

    # mto
    used_dims = [6,7,8]
    block_size = 10000
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q4_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q4_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/mto'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q4_cost_mto.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)

    # qd-tree
    used_dims = [6,7]
    tree_path = './experiments/tpch/' + str(scale) +'/model/q4_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q4_cost_qdtree.txt'    
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')
    training_set = training_set[:, [0,1,3,4]]
    testing_set = testing_set[:, [0,1,3,4]]
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)

    # kd-tree
    tree_path = './experiments/tpch/' + str(scale) +'/model/q4_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q4_cost_kdtree.txt'
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 2)



def tpch_q5(scale, block_size = 10000, query_amount = 100):
    '''
    table: orders,lineitem,supplier,nation
    column: o_orderdate, n_regionkey
    '''

    # mto
    used_dims = [8, 10]
    block_size = 10000
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q5_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q5_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/q5_mto'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q5_cost_mto.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)


    # qd-tree
    used_dims = [0]
    tree_path = './experiments/tpch/' + str(scale) +'/model/q5_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q5_cost_qdtree.txt'    
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 1)

    # kd-tree
    tree_path = './experiments/tpch/' + str(scale) +'/model/q5_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q5_cost_kdtree.txt'
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 3)



def tpch_q9(scale, block_size = 10000, query_amount = 100):
    '''
    table: lineitem, orders, part
    column: l_discount, o_orderdate, p_retailprice
    '''

    # mto
    used_dims = [4,8,11]
    block_size = 10000
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q9_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q9_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/q9_mto'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q9_cost_mto.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)


    # qd-tree
    used_dims = [4]
    tree_path = './experiments/tpch/' + str(scale) +'/model/q9_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q9_cost_qdtree.txt'    
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')
    training_set = training_set[:, [0,3]]
    testing_set = testing_set[:, [0,3]]
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)

    # kd-tree
    tree_path = './experiments/tpch/' + str(scale) +'/model/q9_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q9_cost_kdtree.txt'
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 2)



def tpch_q14(scale, block_size = 10000, query_amount = 100):
    '''
    table: lineitem,part
    column: l_shipdate, p_retailprice
    '''

    # mto
    used_dims = [5,11]
    block_size = 10000
    data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    train_path = './experiments/tpch/' + str(scale) +'/data/q14_train.csv'
    test_path = './experiments/tpch/' + str(scale) +'/data/q14_test.csv'
    tree_path = './experiments/tpch/' + str(scale) +'/model/q14_mto'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q14_cost_mto.txt'
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set, testing_set = helper.generate_queryset_and_save(domain_path, query_amount, queryset_type=2)
    np.savetxt(train_path, training_set, delimiter=',')
    np.savetxt(test_path, testing_set, delimiter=',')
    # training_set = genfromtxt(train_path, delimiter=',')
    # testing_set = genfromtxt(test_path, delimiter=',')
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)


    # qd-tree
    used_dims = [5]
    tree_path = './experiments/tpch/' + str(scale) +'/model/q14_qdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q14_cost_qdtree.txt'    
    helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]
    training_set = genfromtxt(train_path, delimiter=',')
    testing_set = genfromtxt(test_path, delimiter=',')
    training_set = training_set[:, [0,2]]
    testing_set = testing_set[:, [0,2]]
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 0)

    # kd-tree
    tree_path = './experiments/tpch/' + str(scale) +'/model/q14_kdtree'
    cost_path = './experiments/tpch/' + str(scale) +'/result/q14_cost_kdtree.txt'
    execute_query(training_set, testing_set, dataset, boundary, block_size, tree_path, cost_path, 2)



if __name__ == "__main__":

    # # 1. 简化tbl表格
    # extract_columns('lineitem', [0,1,2,4,6,10,11,12], [5,6,7])
    # extract_columns('orders', [0, 4], [1])
    # extract_columns('supplier', [0, 3])
    # extract_columns('nation', [0, 2])
    # extract_columns('part', [0, 7])
    # # 合并数据集
    # merge_tbl_files(1)



    # # 2. 生成数据集、训练集、测试集
    # used_dims = [0,1,2,3,4,5,6,7,8,9,10,11]
    # block_size = 10000
    # scale = 1
    # table_path = './experiments/tpch/' + str(scale) + '/dataset.tbl'
    # data_path = './experiments/tpch/' + str(scale) +'/dataset.csv'
    # domain_path = './experiments/tpch/' + str(scale) +'/domain.csv'
    # train_path = './experiments/tpch/' + str(scale) +'/train.csv'
    # test_path = './experiments/tpch/' + str(scale) +'/test.csv'
    # helper = DatasetAndQuerysetHelper(base_path = './experiments', used_dimensions = used_dims, scale_factor = scale) # set the base path
    # helper.generate_dataset_and_save(table_path, data_path, domain_path)  # generate dataset
    # dataset, domains = helper.load_dataset(used_dims, data_path, domain_path)
    # boundary = [interval[0] for interval in domains]+[interval[1] for interval in domains]

    # training_set, testing_set = helper.generate_queryset_and_save(domain_path, 100, queryset_type=2)
    # np.savetxt(train_path, training_set, delimiter=',')
    # np.savetxt(test_path, testing_set, delimiter=',')



    # 3. 执行查询
    # tpch_q1(1, 10000)
    # print("=================q1====================")
    tpch_q6(1, 10000)
    print("=================q6====================")
    # tpch_q4(1, 10000)
    # print("=================q4====================")
    # tpch_q5(1, 10000)
    # print("=================q5====================")
    # tpch_q9(1, 10000)
    # print("=================q9====================")
    # tpch_q14(1, 10000)
    # print("=================q14====================")

