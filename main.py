
import torch
from data_loader import loader
from sklearn.model_selection import train_test_split
from utils.parament import hyperparameters_process
from model import mainModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':
    args = hyperparameters_process()
    # 设置随机种子
    torch.manual_seed(args.seed)

    timesteps = args.timesteps
    database = "movie"
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    iteraction = args.iteraction
    rating_threshold = 4
    
    data_loader = loader(database)
    edge_index, genre_index, edge_arr, num_users, num_items,num_genres, user_mapping, movie_mapping =data_loader.read_data(database,rating_threshold)
    num_node = num_items + num_users
    print("num_users:",num_users)
    print("num_items:",num_items)
    print("num_genres:",num_genres)
    print("num_node:",num_node)
    print("num_edges:",edge_index.shape[1])

    #划分数据集
    num_users, num_items = len(user_mapping), len(movie_mapping)
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]  # 所有索引
    
    train_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=1)  # 将数据集划分成80:10的训练集:测试集
    val_indices, test_indices = train_test_split(
            test_indices, test_size=0.5, random_state=1)  # 将测试集划分成10:10的验证集:测试集,最后的比例就是80:10:10
    
    train_edges = edge_index[:, train_indices]
    val_edges = edge_index[:, val_indices]
    test_edges = edge_index[:, test_indices]

    train_nodes = set(train_edges.flatten().cpu().numpy())
    test_nodes = set(test_edges.flatten().cpu().numpy())

    common_nodes = test_nodes.intersection(train_nodes)
    print(f"测试集中的节点有 {len(common_nodes) / len(test_nodes) * 100:.2f}% 出现在训练集中")

    model = mainModel(device, database,timesteps, batch_size, embedding_dim, num_users, num_items,num_genres=num_genres).to(device)
    train_edges = train_edges.to(device)
    val_edges = val_edges.to(device)
    test_edges = test_edges.to(device)

#     print("train_edges:",train_edges[0].max())
#     print("train_edges:",train_edges[1].max())
    
    # 训练模型
    model.Coach(iteraction, train_edges,val_edges,genres_edge_index=genre_index)
    # 使用模型
    #model.load_state_dict(torch.load("model_pth/best_model.pth", weights_only=True))
    # 评估模型
    #model.Eval(test_edges,train_edges)