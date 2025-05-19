import argparse

def hyperparameters_process():
    parser = argparse.ArgumentParser(description="Train a model with hyperparameters.")
    
    # 添加超参数
    parser.add_argument('--data', type=str, default="book", help='Data for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension for training')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--iteraction', type=int, default=8000, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=10, help='Number of timesteps for training')
    # 解析参数
    args = parser.parse_args()
    return args