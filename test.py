import torch
from torch import nn


def test_linear():
    # 测试代码
    print("Testing the code...")
    
    my_ln = nn.Linear(10, 3)
    print("my_ln:", my_ln)
    
    X = torch.randn(11, 2, 3, 10)
    print("X shape:", X.shape)
    X = my_ln(X)
    print("X shape:", X.shape)
    print("Testing completed.")
    
if __name__ == '__main__':
     test_linear()