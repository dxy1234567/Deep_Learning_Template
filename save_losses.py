import matplotlib.pyplot as plt
import os

# 提取的 train 和 val loss 值
train_losses = [0.01742312, 0.00324323, 0.00257624, 0.00229929, 0.00213698, 0.00204269, 0.00197067, 0.00188603, 0.00180737]
# 验证集损失值
val_losses = [0.00420819, 0.00283320, 0.00247400, 0.00230779, 0.00214046, 0.00206228, 0.00198455, 0.00190537, 0.00181877]

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

saved_path = 'output/loss_plot.png'
os.makedirs('output/', exist_ok=True)
# 保存图像
plt.savefig('output/loss_plot.png')