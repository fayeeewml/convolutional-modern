import torch 
from torch import nn
from d2l import torch as d2l

# LeNet-5 model creat
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10))

X = torch.randn(1, 1, 28, 28)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t',X.shape)

# traning
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print('train_iter:', train_iter)

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        # if type(m) == nn.Linear or type(m) == nn.Conv2d:
        #     nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) # 遍历每层net并在其身上作用
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        net.train()
        sum_l = 0
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            sum_l += l.item()
            # print('l:', l) # tensor(0.6508, device='cuda:0', grad_fn=<NllLossBackward0>)
        train_l = sum_l / len(train_iter)
        print('train_l:', train_l)

if __name__ == '__main__':
    lr, num_epochs = 0.9, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

    print("Training complete.")
    # 训练完成后，保存模型
    torch.save(net.state_dict(), 'lenet.pth')
    print("Model saved as lenet.pth")
    # 加载模型
    loaded_net = net
    loaded_net.load_state_dict(torch.load('lenet.pth'))
    loaded_net.eval()  # 设置为评估模式
    # 测试加载的模型
    sum_l = 0
    loss = nn.CrossEntropyLoss()
    for i, (X, y) in enumerate(test_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        sum_l += l.item()
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {l.item():.4f}")
    train_l = sum_l / len(test_iter)
    print('train_l:', train_l)

