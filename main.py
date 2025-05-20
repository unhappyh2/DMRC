
import torch
from Data_load import Data_loader
from sklearn.model_selection import train_test_split
from utils.parament import hyperparameters_process
from bprmf import MF

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

if __name__ == '__main__':
    args = hyperparameters_process()
    # 设置随机种子
    torch.manual_seed(args.seed)

    timesteps = args.timesteps
    database = "music"
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    iteraction = args.iteraction
    
    data_loader = Data_loader(database)
    data = data_loader.load_edge_index("data/yelp2018")
    edge_index = data['user_item_edge_index']
    num_edges = data['user_item_edge_index'].shape[1]
    num_users = data['num_users']
    num_items = data['num_items']
    num_entities = data['num_entities']
    num_interactions = data['num_relations']

    print("Number of edges: ", num_edges)
    print("Number of users: ", num_users)
    print("Number of items: ", num_items)
    print("Number of entities: ", num_entities)
    print("Number of interactions: ", num_interactions)

    #划分数据集

    all_indices = [i for i in range(num_edges)]  # 所有索引
    
    train_indices, test_indices = train_test_split(
            all_indices, test_size=0.1, random_state=1)  # 将数据集划分成80:10的训练集:测试集
    # val_indices, test_indices = train_test_split(
    #         test_indices, test_size=0.5, random_state=1)  # 将测试集划分成10:10的验证集:测试集,最后的比例就是80:10:10
    
    train_edges = edge_index[:, train_indices]
    # val_edges = edge_index[:, val_indices]
    test_edges = edge_index[:, test_indices]

    train_nodes = set(train_edges.flatten().cpu().numpy())
    test_nodes = set(test_edges.flatten().cpu().numpy())

    common_nodes = test_nodes.intersection(train_nodes)
    print(f"测试集中的节点有 {len(common_nodes) / len(test_nodes) * 100:.2f}% 出现在训练集中")

    model = MF(device,batch_size, embedding_dim, num_users, num_items).to(device)
    train_edges = train_edges.to(device)
    # val_edges = val_edges.to(device)
    test_edges = test_edges.to(device)

#     print("train_edges:",train_edges[0].max())
#     print("train_edges:",train_edges[1].max())
    
    # 训练模型
    model.Coach(iteraction, train_edges,test_edges)
    # 使用模型
    #model.load_state_dict(torch.load("model_pth/best_model.pth", weights_only=True))
    # 评估模型
    #model.Eval(test_edges,train_edges)