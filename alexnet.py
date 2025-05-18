# from mxnet import np, npx
# from mxnet.gluon import nn


import torch
import torch.nn as nn
# import d2l as d2l
from d2l import torch as d2l
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6) # flatten 这种展平操作通常用于：1将 CNN 提取的空间特征转换为一维特征向量 2作为全连接层的输入（全连接层要求一维输入）
        # print('x view:', x.shape)
        x = self.classifier(x)
        return x
    
    def print_layer_shapes(self, x):
        print('input x shape:', x.shape)
        for layer in self.features:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape:', x.shape)
        for layer in self.classifier:
            x = layer(x)
            print(layer.__class__.__name__, 'output shape: ', x.shape)

    def train_net(self, train_iter, num_epochs, lr, device):
        """用GPU训练模型(在第六章定义)"""
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.apply(init_weights)
        print('training on', device)
        self.to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        print('loss:', loss)
        # return
        for epoch in range(num_epochs):
            self.train()
            sum_l = 0
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad() # 为什么每一步要清空梯度
                X, y = X.to(device), y.to(device)
                y_hat = self.forward(X)
                l = loss(y_hat, y)
                if i > 5:
                    print('l:', l.item()) #    l 是个啥
                    break
                if torch.isnan(l).any():
                    print(f"Epoch {epoch}, Batch {i}: 检测到NaN Loss")
                    print(f"l.item: {l.item()}, y: {y}, y_hat: {y_hat}")
                    # 可选：保存当前模型状态用于调试
                    torch.save(self.state_dict(), 'nan_model.pth')
                    break  # 或使用continue跳过当前批次
                l.backward()
                optimizer.step()
                sum_l += l.item()
                if i % 1 == 0:
                    print('epoch:', epoch, ', sum_l:', sum_l, ', len(train_iter):', len(train_iter), 'train_l:', sum_l / len(train_iter), 'i=', i, 'l:', l)
                    torch.save(self.state_dict(), './alexnet/alexnet' + str(epoch) + str(i) + '.pth')
            print('epoch:', epoch, ', sum_l:', sum_l, ', len(train_iter):', len(train_iter), 'train_l:', sum_l / len(train_iter))
        torch.save(self.state_dict(), './alexnet/alexnet_final' + '.pth')   



def my_train():
    # 1. 读取数据
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize = 224)
    print('train_iter:', train_iter, 'len(train_iter):', len(train_iter))
    print('test_iter:', test_iter, 'len(test_iter):', len(test_iter))
    
    train_features, train_labels = next(iter(train_iter))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    
    # # 2. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AlexNet()
    num_epochs, lr = 1, 0.001
    net.train_net(train_iter, num_epochs, lr, device)


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_net = AlexNet()
    loaded_net.load_state_dict(torch.load('./alexnet/alexnet0100.pth'))
    loaded_net.eval()  # 设置为评估模式
    loaded_net.to(device)
    # 测试加载的模型
    test_iter = d2l.load_data_fashion_mnist(128, resize = 224)[1]
    sum_l = 0
    loss = nn.CrossEntropyLoss() 
    
    for i, (X, y) in enumerate(test_iter):
        X, y = X.to(device), y.to(device)
        y_hat = loaded_net(X)
        l = loss(y_hat, y)
        sum_l += l.item()
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {l.item():.4f}")   

if __name__ == '__main__':
    # my_train()
    test_model()
            
        
        
        