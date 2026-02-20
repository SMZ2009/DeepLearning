import numpy as np

class Perceptron:
    """
    单层感知机(最简单的前馈神经网络, FNN)
    """

    def __init__(self, input_dim, lr=0.01, max_epoch=100):
        self.lr        = lr
        self.max_epoch = max_epoch

        self.W         = np.zeros(input_dim)
        self.b         = 0

    def sign(self, x):
        """
        激活函数(符号函数)
        """
        return np.where(x >= 0, 1, -1)
    
    def forward(self, x):
        """
        前向传播(加权和 + 激活函数)
        """
        z = np.dot(x, self.W) + self.b
        return self.sign(z)
    
    def fit(self, X, y):
        """
        训练
        """
        for epoch in range(self.max_epoch):
            err_count = 0
            for xi, yi in zip(X, y):
                y_pred = self.forward(xi)
                if y_pred != yi:
                    # 若分类错误则更新参数
                    self.W += self.lr * yi * xi
                    self.b += self.lr * yi
                    err_count += 1
            if err_count == 0:
                # 若分类完全正确则停止
                print(f"Round {epoch + 1}: Completely Correct, Paused!")
                break
        print("End of Training!")

    def predict(self, X):
        """
        预测
        """
        return self.forward(X)