import torch
import torch.nn as nn
import torch.optim as optim

# 定义简单模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# 初始化模型
net = SimpleNet(input_size=10, num_classes=3)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)  # 将模型移至指定设备

# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    # 假设 x 和 y 是一个批次的数据
    x = torch.randn(32, 10).to(device)  # 输入数据
    y = torch.randint(0, 3, (32,)).to(device)  # 真实标签
    
    # 前向传播
    outputs = net(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")