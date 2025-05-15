from datetime import datetime

def log_beginning(model, database, k, log_path="./log/train_log.txt"):
    """
    在训练开始时，创建一个新的日志文件，并写入标题。
    """
    with open(log_path, 'a') as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 当前时间戳
        f.write("=" * 10 + f"[{now}]"+"=" * 10+"\n")
        f.write("=" * 5 + f"[{model}] || [{database}]"+"="*2+f"[k = {k}]"+"="*4+"\n")

def logger(metrics: dict, epoch: int, log_path="./log/train_log.txt"):
    """
    将验证指标写入 text 文件中，记录 epoch 和当前时间。
    """
    with open(log_path, 'a') as f:
        f.write(f" Epoch {epoch}:\n")
        for key, value in metrics.items():
            f.write(f"    {key}: {value:.4f}\n")
        f.write("\n")