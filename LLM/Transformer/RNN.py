import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1.生成简单的序列数据
# 比如：输入 [1,2,3] 输出 4
def create_dataset():
    X = []
    y = []
    
    for i in range(1000):
        start = np.random.randint(0, 100)
        seq = np.array([start, start+1, start+2, start+3])
        X.append(seq[:3])
        y.append(seq[3])
    
    X = np.array(X)
    y = np.array(y)
    
    # reshape 成 RNN 需要的格式 (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

X, y = create_dataset()

# 2️⃣ 构建最基础的 RNN 模型
model = Sequential([
    SimpleRNN(10, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

# 3️⃣ 编译模型
model.compile(optimizer='adam', loss='mse')

# 4️⃣ 训练模型
model.fit(X, y, epochs=10, batch_size=32)

# 5️⃣ 测试模型
test_input = np.array([[10, 11, 12]])
test_input = test_input.reshape((1, 3, 1))

prediction = model.predict(test_input)
print("预测结果:", prediction[0][0])
print("真实结果:", 13)