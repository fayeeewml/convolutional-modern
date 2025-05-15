# from mxnet import np, npx
# from mxnet.gluon import nn


import torch
import torch.nn as nn
# import d2l as d2l
from d2l import torch as d2l

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
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
        print('inpur x :', x.shape)
        x = self.features(x)
        # for layer in self.features:
        #     print(layer.__class__.__name__, 'output shape:\t', x.shape)
        print('x features:', x.shape)
        x = x.view(x.size(0), 256 * 6 * 6)
        print('x view:', x.shape)
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
        for epoch in range(num_epochs):
            self.train()
            sum_l = 0
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad() # 为什么每一步要清空梯度
                X, y = X.to(device), y.to(device)
                y_hat = self.forward(X)
                l = loss(y_hat, y)
                if i == 0:
                    print('l:', l) #    l 是个啥
                l.backward()
                optimizer.step()
                sum_l += l.items()
                if i % 1000 == 0:
                    print('epoch:', epoch, ', sum_l:', sum_l, ', len(train_iter):', len(train_iter), 'train_l:', sum_l / len(train_iter), 'i=', i, 'l:', l)
            
            print('epoch:', epoch, ', sum_l:', sum_l, ', len(train_iter):', len(train_iter), 'train_l:', sum_l / len(train_iter))
            

if __name__ == '__main__':
    # 1. 读取数据
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 2. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AlexNet()
    num_epochs, lr = 10, 0.01
    net.train_net(train_iter, num_epochs, lr, device)
            
        
        
        