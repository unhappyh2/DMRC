
import torch
import numpy as np

class Data_loader():
    def __init__(self, data_name, data_dir = './data' ):
        self.data_dir = data_dir
        self.data_name = self.database_name(data_name)

    def database_name(self,data_name):
        if data_name == 'music':
            return '/last-fm'
        elif data_name == 'book':
            return '/amazon-book'
        elif data_name == 'yelp':
            return '/yelp2018'


    def load_edge_index(self,data_path):
        user_item_edges = []
        user_set = set()
        item_set = set()

        # 读取用户–项目交互
        with open(f"{data_path}/train.txt", "r") as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                if len(tokens) < 2:
                    continue
                user = tokens[0]
                items = tokens[1:]
                user_set.add(user)
                item_set.update(items)
                for item in set(items):  # 去重
                    user_item_edges.append([user, item])

        user_item_edge_index = torch.tensor(user_item_edges, dtype=torch.long).T  # shape [2, num_edges]

        # 读取知识图谱三元组
        kg_edges = []
        relations = []
        entity_set = set()
        with open(f"{data_path}/kg_final.txt", "r") as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                kg_edges.append([h, t])
                relations.append(r)
                entity_set.update([h, t])

        kg_edge_index = torch.tensor(kg_edges, dtype=torch.long).T
        kg_edge_type = torch.tensor(relations, dtype=torch.long)

        # 统计数量
        num_users = max(user_set) + 1
        num_items = max(item_set) + 1
        num_entities = len(entity_set) - num_items
        num_relations = len(relations)
        return {
            'user_item_edge_index': user_item_edge_index,
            'kg_edge_index': kg_edge_index,
            'kg_edge_type': kg_edge_type,
            'num_users': num_users,
            'num_items': num_items,
            'num_entities': num_entities,
            'num_relations': num_relations,
        }
if __name__ == '__main__':
    data_loader = Data_loader('./data', 2)
    data = data_loader.load_edge_index("data/last-fm")
    print(data['user_item_edge_index'][:,0:115])
    print("用户-项目 edge_index:", data['user_item_edge_index'].shape)
    print("KG edge_index:", data['kg_edge_index'].shape)
    print("KG edge_type:", data['kg_edge_type'].shape)
    print("用户数:", data['num_users'])
    print("项目数:", data['num_items'])
    print("实体数:", data['num_entities'])
    print("关系数:", data['num_relations'])
    with open("data/last-fm/train.txt", "r") as f:
        first_line = f.readline()
        print("原始第1行：", first_line.strip())