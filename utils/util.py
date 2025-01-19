from matplotlib import pyplot as plt
from datetime import datetime

def write_log(train_losses, val_losses, path_output):
    N = len(train_losses)
    assert N == len(val_losses), "Train and validation loss lists must be of the same length."
    
    # 打开文件，追加模式写入
    with open(path_output, 'a') as log_file:
        # 写入标题和当前时间
        log_file.write("Training Log\n")
        log_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*40 + "\n")
        
        # 写入表头
        log_file.write(f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}\n")
        log_file.write("-"*40 + "\n")

        # 写入每一轮的损失数据
        for i in range(N):
            log_file.write(f"{i + 1:<10}{train_losses[i]:<15.6f}{val_losses[i]:<15.6f}\n")
        
        # 结束分割线
        log_file.write("="*40 + "\n\n")

def plot_losses(train_losses, val_losses, path_output):
    N = len(train_losses)
    # 绘制折线图
    epochs = list(range(1, N + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Train and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, N + 1, 5))
    plt.legend()
    plt.grid()

    # 保存图像
    plt.savefig(path_output)

def print_progress(i, N):
    """
    打印当前进度。

    Param:
        i: 当前进度（第i次迭代）
        N: 总迭代次数
    """
    if i % 100 == 0 or i == N:  # 每 100 次迭代或最后一次迭代时更新进度
        run = i / N * 100
        print("Runing: {:.2f}% done.".format(run))