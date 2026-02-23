import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 生成简单二分类数据
# =========================
np.random.seed(42)

num_samples = 200

# 类0
X0 = np.random.randn(num_samples//2, 2) + np.array([-2, -2])
y0 = np.zeros((num_samples//2, 1))

# 类1
X1 = np.random.randn(num_samples//2, 2) + np.array([2, 2])
y1 = np.ones((num_samples//2, 1))

X = np.vstack((X0, X1))
y = np.vstack((y0, y1))

# =========================
# 2. 单层感知机实现
# =========================

class Perceptron:
    def __init__(self, input_dim):
        # 参数初始化
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1,))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.sigmoid(z)
        return y_hat
    
    def compute_loss(self, y_hat, y):
        m = y.shape[0]
        loss = - (1/m) * np.sum(
            y * np.log(y_hat + 1e-8) +
            (1 - y) * np.log(1 - y_hat + 1e-8)
        )
        return loss
    
    def backward(self, X, y_hat, y):
        m = y.shape[0]
        
        dz = y_hat - y
        dW = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        
        return dW, db
    
    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db


# =========================
# 3. 训练
# =========================

model = Perceptron(input_dim=2)

epochs = 1000
lr = 0.1

losses = []

for epoch in range(epochs):
    # 前向传播
    y_hat = model.forward(X)
    
    # 计算损失
    loss = model.compute_loss(y_hat, y)
    losses.append(loss)
    
    # 反向传播
    dW, db = model.backward(X, y_hat, y)
    
    # 更新参数
    model.update(dW, db, lr)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# =========================
# 4. 可视化
# =========================

plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 决策边界
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y.flatten())

# 画分界线
x_min, x_max = X[:,0].min(), X[:,0].max()
xs = np.linspace(x_min, x_max, 100)
ys = -(model.W[0]*xs + model.b)/model.W[1]

plt.plot(xs, ys, 'r')
plt.title("Decision Boundary")
plt.show()