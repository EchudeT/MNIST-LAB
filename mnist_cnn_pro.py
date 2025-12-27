"""
纯 NumPy 实现的卷积神经网络（CNN）- 优化版本
包含多种进阶优化策略：Adam优化器、Dropout、BatchNorm、数据增强、学习率调度等
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict
import os
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 TensorFlow 镜像源
os.environ['TF_KERAS_DATASETS_MIRROR'] = 'https://mirrors.aliyun.com/tensorflow/'


def clip_gradient(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    梯度裁剪，防止梯度爆炸

    参数:
        grad: 梯度
        max_norm: 最大范数

    返回:
        裁剪后的梯度
    """
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


class AdamOptimizer:
    """
    Adam优化器实现
    自适应学习率优化算法
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        初始化Adam优化器

        参数:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 数值稳定性常数
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        更新参数

        参数:
            params: 参数字典
            grads: 梯度字典
        """
        self.t += 1

        for key in params.keys():
            # 初始化动量
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # 更新一阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # 更新二阶矩估计
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # 更新参数
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class Conv2D:
    """卷积层实现（包含L2正则化支持）"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He初始化
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                      np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))

        self.dweights = None
        self.dbias = None
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, height, width = x.shape

        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                 (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]

                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(
                        receptive_field * self.weights[k, :, :, :],
                        axis=(1, 2, 3)
                    ) + self.bias[k]

        self.cache = x_padded
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_padded = self.cache
        batch_size, in_channels, padded_height, padded_width = x_padded.shape
        _, out_channels, out_height, out_width = dout.shape

        dx_padded = np.zeros_like(x_padded)
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]

                for k in range(out_channels):
                    self.dweights[k] += np.sum(
                        receptive_field * dout[:, k:k+1, i:i+1, j:j+1],
                        axis=0
                    )
                    self.dbias[k] += np.sum(dout[:, k, i, j])
                    dx_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * dout[:, k:k+1, i:i+1, j:j+1]

        # 应用梯度裁剪
        self.dweights = clip_gradient(self.dweights, max_norm=5.0)
        self.dbias = clip_gradient(self.dbias, max_norm=5.0)

        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        return {'weights': self.weights, 'bias': self.bias}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {'weights': self.dweights, 'bias': self.dbias}


class MaxPool2D:
    """最大池化层"""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(pool_region, axis=(2, 3))

        self.cache = x
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        batch_size, channels, height, width = x.shape
        _, _, out_height, out_width = dout.shape

        dx = np.zeros_like(x)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pool_region = x[:, :, h_start:h_end, w_start:w_end]

                for b in range(batch_size):
                    for c in range(channels):
                        mask = (pool_region[b, c] == np.max(pool_region[b, c]))
                        dx[b, c, h_start:h_end, w_start:w_end] += \
                            mask * dout[b, c, i, j] / np.sum(mask)

        return dx


class AveragePool2D:
    """平均池化层"""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.mean(pool_region, axis=(2, 3))

        self.cache = (batch_size, channels, height, width, out_height, out_width)
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width, out_height, out_width = self.cache

        dx = np.zeros((batch_size, channels, height, width))
        pool_area = self.pool_size * self.pool_size

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                dx[:, :, h_start:h_end, w_start:w_end] += \
                    dout[:, :, i:i+1, j:j+1] / pool_area

        return dx


class BatchNorm:
    """批归一化层"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.dgamma = None
        self.dbeta = None
        self.cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, channels, height, width = x.shape

        if training:
            # 计算批次统计量
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + \
                              (1 - self.momentum) * batch_mean.squeeze()
            self.running_var = self.momentum * self.running_var + \
                             (1 - self.momentum) * batch_var.squeeze()

            # 归一化
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)

            # 保存用于反向传播
            self.cache = (x, x_norm, batch_mean, batch_var)
        else:
            # 测试时使用移动平均
            running_mean = self.running_mean.reshape(1, -1, 1, 1)
            running_var = self.running_var.reshape(1, -1, 1, 1)
            x_norm = (x - running_mean) / np.sqrt(running_var + self.eps)

        # 缩放和平移
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        out = gamma * x_norm + beta

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, x_norm, batch_mean, batch_var = self.cache
        batch_size, channels, height, width = x.shape
        N = batch_size * height * width

        gamma = self.gamma.reshape(1, -1, 1, 1)

        # 计算梯度
        self.dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
        self.dbeta = np.sum(dout, axis=(0, 2, 3))

        dx_norm = dout * gamma

        dvar = np.sum(dx_norm * (x - batch_mean) * -0.5 *
                     np.power(batch_var + self.eps, -1.5), axis=(0, 2, 3), keepdims=True)

        dmean = np.sum(dx_norm * -1.0 / np.sqrt(batch_var + self.eps), axis=(0, 2, 3), keepdims=True) + \
                dvar * np.sum(-2.0 * (x - batch_mean), axis=(0, 2, 3), keepdims=True) / N

        dx = dx_norm / np.sqrt(batch_var + self.eps) + \
             dvar * 2.0 * (x - batch_mean) / N + \
             dmean / N

        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        return {'gamma': self.gamma, 'beta': self.beta}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {'gamma': self.dgamma, 'beta': self.dbeta}


class Dropout:
    """Dropout层，防止过拟合"""

    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return x

        # 生成掩码
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Dense:
    """全连接层（包含L2正则化支持）"""

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

        self.dweights = None
        self.dbias = None
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        output = np.dot(x, self.weights) + self.bias
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache

        self.dweights = np.dot(x.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.weights.T)

        # 应用梯度裁剪
        self.dweights = clip_gradient(self.dweights, max_norm=5.0)
        self.dbias = clip_gradient(self.dbias, max_norm=5.0)

        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        return {'weights': self.weights, 'bias': self.bias}

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {'weights': self.dweights, 'bias': self.dbias}


class ReLU:
    """ReLU激活函数"""

    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache
        dx = dout * (x > 0)
        return dx


class Softmax:
    """Softmax激活函数（数值稳定实现）"""

    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = output
        return output


class CrossEntropyLoss:
    """交叉熵损失函数"""

    def __init__(self):
        self.cache = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        batch_size = predictions.shape[0]

        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        loss = -np.sum(targets * np.log(predictions)) / batch_size

        self.cache = (predictions, targets)
        return loss

    def backward(self) -> np.ndarray:
        predictions, targets = self.cache
        batch_size = predictions.shape[0]

        dx = (predictions - targets) / batch_size
        return dx


class CNNOptimized:
    """
    优化版CNN模型
    包含：Adam优化器、Dropout、BatchNorm、L2正则化、早停等
    """

    def __init__(self, use_dropout: bool = True, use_batchnorm: bool = True,
                 weight_decay: float = 1e-4):
        """
        初始化优化版CNN模型

        参数:
            use_dropout: 是否使用Dropout
            use_batchnorm: 是否使用BatchNorm
            weight_decay: L2正则化系数
        """
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.weight_decay = weight_decay

        # 构建网络结构（更深的网络）
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm(16) if use_batchnorm else None
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm(32) if use_batchnorm else None
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # 32 * 7 * 7 = 1568
        self.fc1 = Dense(input_size=32 * 7 * 7, output_size=256)
        self.relu3 = ReLU()
        self.dropout1 = Dropout(0.5) if use_dropout else None

        self.fc2 = Dense(input_size=256, output_size=128)
        self.relu4 = ReLU()
        self.dropout2 = Dropout(0.5) if use_dropout else None

        self.fc3 = Dense(input_size=128, output_size=10)
        self.softmax = Softmax()

        self.loss_fn = CrossEntropyLoss()

        # 优化器
        self.optimizer = None

        # 训练历史
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.learning_rates = []

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        # 第一个卷积块
        out = self.conv1.forward(x)
        if self.bn1:
            out = self.bn1.forward(out, training)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        # 第二个卷积块
        out = self.conv2.forward(out)
        if self.bn2:
            out = self.bn2.forward(out, training)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        # 展平
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        # 全连接层
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        if self.dropout1:
            out = self.dropout1.forward(out, training)

        out = self.fc2.forward(out)
        out = self.relu4.forward(out)
        if self.dropout2:
            out = self.dropout2.forward(out, training)

        out = self.fc3.forward(out)
        out = self.softmax.forward(out)

        return out

    def backward(self, loss_grad: np.ndarray):
        """反向传播"""
        dout = self.fc3.backward(loss_grad)

        if self.dropout2:
            dout = self.dropout2.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.fc2.backward(dout)

        if self.dropout1:
            dout = self.dropout1.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)

        # 还原形状
        dout = dout.reshape(-1, 32, 7, 7)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        if self.bn2:
            dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        if self.bn1:
            dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

    def get_all_params_and_grads(self) -> Tuple[Dict, Dict]:
        """获取所有参数和梯度"""
        params = {}
        grads = {}

        layers_with_params = [
            ('conv1', self.conv1),
            ('conv2', self.conv2),
            ('fc1', self.fc1),
            ('fc2', self.fc2),
            ('fc3', self.fc3)
        ]

        if self.bn1:
            layers_with_params.append(('bn1', self.bn1))
        if self.bn2:
            layers_with_params.append(('bn2', self.bn2))

        for name, layer in layers_with_params:
            layer_params = layer.get_params()
            layer_grads = layer.get_grads()

            for param_name, param_value in layer_params.items():
                key = f"{name}_{param_name}"
                params[key] = param_value

                # 添加L2正则化梯度（除了bias和BN参数）
                if 'bias' not in param_name and 'beta' not in param_name and 'gamma' not in param_name:
                    grads[key] = layer_grads[param_name] + self.weight_decay * param_value
                else:
                    grads[key] = layer_grads[param_name]

        return params, grads

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   epochs: int = 20, batch_size: int = 64,
                   initial_lr: float = 0.001,
                   use_lr_schedule: bool = True,
                   early_stopping_patience: int = 5):
        """
        训练模型（包含学习率调度和早停）

        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            epochs: 训练轮数
            batch_size: 批量大小
            initial_lr: 初始学习率
            use_lr_schedule: 是否使用学习率调度
            early_stopping_patience: 早停耐心值
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # 初始化Adam优化器
        self.optimizer = AdamOptimizer(learning_rate=initial_lr)

        # 早停相关变量
        best_test_loss = float('inf')
        early_stop_count = 0
        best_params = None

        print(f"\n{'='*80}")
        print(f"开始训练优化版CNN模型")
        print(f"训练集大小: {n_samples}, 测试集大小: {X_test.shape[0]}")
        print(f"优化策略: Adam优化器, {'BatchNorm, ' if self.use_batchnorm else ''}"
              f"{'Dropout, ' if self.use_dropout else ''}L2正则化")
        print(f"{'='*80}\n")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 学习率调度（余弦退火）
            if use_lr_schedule:
                lr = initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2
                self.optimizer.learning_rate = lr
            else:
                lr = initial_lr

            # 随机打乱训练数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0

            # 批量训练
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # 前向传播
                predictions = self.forward(X_batch, training=True)

                # 计算损失
                loss = self.loss_fn.forward(predictions, y_batch)
                epoch_loss += loss * (end_idx - start_idx)

                # 反向传播
                loss_grad = self.loss_fn.backward()
                self.backward(loss_grad)

                # 获取参数和梯度
                params, grads = self.get_all_params_and_grads()

                # 使用Adam更新参数
                self.optimizer.update(params, grads)

                # 显示进度
                if (batch + 1) % 20 == 0 or batch == n_batches - 1:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch+1}/{n_batches}, "
                          f"Loss: {loss:.4f}, LR: {lr:.6f}", end='\r')

            # 计算平均损失
            avg_train_loss = epoch_loss / n_samples

            # 评估
            train_acc = self.evaluate(X_train, y_train, batch_size)
            test_loss, test_acc = self.evaluate(X_test, y_test, batch_size, return_loss=True)

            # 保存历史
            self.train_losses.append(avg_train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(lr)

            epoch_time = time.time() - epoch_start_time

            print(f"\n  Epoch {epoch+1}/{epochs} - "
                  f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2%}, "
                  f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2%}, "
                  f"用时: {epoch_time:.2f}s")

            # 早停检查
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stop_count = 0
                # 保存最佳参数
                best_params, _ = self.get_all_params_and_grads()
                best_params = {k: v.copy() for k, v in best_params.items()}
                print(f"  [*] 新的最佳模型！测试损失: {best_test_loss:.4f}")
            else:
                early_stop_count += 1
                print(f"  [!] 测试损失未改善 ({early_stop_count}/{early_stopping_patience})")

                if early_stop_count >= early_stopping_patience:
                    print(f"\n提前停止训练！在第 {epoch+1} 轮")
                    # 恢复最佳参数
                    if best_params:
                        for key, value in best_params.items():
                            parts = key.split('_', 1)
                            layer_name = parts[0]
                            param_name = parts[1]
                            layer = getattr(self, layer_name)
                            if param_name == 'weights':
                                layer.weights = value
                            elif param_name == 'bias':
                                layer.bias = value
                            elif param_name == 'gamma':
                                layer.gamma = value
                            elif param_name == 'beta':
                                layer.beta = value
                    break

            print("-" * 80)

        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f}s")
        print(f"最佳测试损失: {best_test_loss:.4f}")
        print(f"最终测试准确率: {self.test_accuracies[-1]:.2%}")
        print("=" * 80)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64,
                 return_loss: bool = False):
        """评估模型"""
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        correct = 0
        total_loss = 0.0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            # 前向传播（测试模式）
            predictions = self.forward(X_batch, training=False)

            if return_loss:
                loss = self.loss_fn.forward(predictions, y_batch)
                total_loss += loss * (end_idx - start_idx)

            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct += np.sum(pred_labels == true_labels)

        accuracy = correct / n_samples

        if return_loss:
            avg_loss = total_loss / n_samples
            return avg_loss, accuracy

        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = self.forward(X, training=False)
        return np.argmax(predictions, axis=1)


def augment_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据增强：随机平移和旋转

    参数:
        X: 输入图像
        y: 标签

    返回:
        增强后的数据和标签
    """
    print("正在进行数据增强...")

    augmented_X = []
    augmented_y = []

    for i, (img, label) in enumerate(zip(X, y)):
        # 原始图像
        augmented_X.append(img)
        augmented_y.append(label)

        # 随机平移
        if np.random.rand() > 0.5:
            dx = np.random.choice([-1, 0, 1])
            dy = np.random.choice([-1, 0, 1])
            img_translated = translate_image(img[0], dx, dy)
            augmented_X.append(img_translated[np.newaxis, :, :])
            augmented_y.append(label)

        # 随机旋转
        if np.random.rand() > 0.5:
            angle = np.random.randint(-10, 11)
            img_rotated = rotate_image(img[0], angle)
            augmented_X.append(img_rotated[np.newaxis, :, :])
            augmented_y.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i+1}/{len(X)} 张图像", end='\r')

    print(f"\n数据增强完成！原始: {len(X)}, 增强后: {len(augmented_X)}")

    return np.array(augmented_X), np.array(augmented_y)


def translate_image(image: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """图像平移"""
    h, w = image.shape
    new_image = np.zeros_like(image)

    if dx > 0:
        new_image[:, dx:] = image[:, :w-dx]
    elif dx < 0:
        new_image[:, :w+dx] = image[:, -dx:]
    else:
        new_image = image.copy()

    if dy > 0:
        temp = new_image.copy()
        new_image[dy:, :] = temp[:h-dy, :]
        new_image[:dy, :] = 0
    elif dy < 0:
        temp = new_image.copy()
        new_image[:h+dy, :] = temp[-dy:, :]
        new_image[h+dy:, :] = 0

    return new_image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """图像旋转（简化版）"""
    h, w = image.shape
    center = (w // 2, h // 2)

    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    new_image = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            # 坐标转换
            x = j - center[0]
            y = i - center[1]

            new_x = int(cos_theta * x - sin_theta * y + center[0])
            new_y = int(sin_theta * x + cos_theta * y + center[1])

            if 0 <= new_x < w and 0 <= new_y < h:
                new_image[i, j] = image[new_y, new_x]

    return new_image


def load_mnist_data(reduce_size: bool = False,
                   apply_augmentation: bool = False,
                   standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载MNIST数据集

    参数:
        reduce_size: 是否减少数据量
        apply_augmentation: 是否应用数据增强
        standardize: 是否标准化

    返回:
        (X_train, y_train, X_test, y_test)
    """
    print("正在加载 MNIST 数据集...")

    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if reduce_size:
        X_train = X_train[:5000]
        y_train = y_train[:5000]
        X_test = X_test[:1000]
        y_test = y_test[:1000]

    # 归一化
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 标准化
    if standardize:
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / (std + 1e-8)
        X_test = (X_test - mean) / (std + 1e-8)

    # 添加通道维度
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    # one-hot编码
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    # 数据增强
    if apply_augmentation:
        X_train, y_train = augment_data(X_train, y_train)

    print(f"数据加载完成！")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def visualize_comparison(baseline_model, optimized_model, X_test, y_test):
    """可视化基线模型和优化模型的对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 损失曲线对比
    axes[0, 0].plot(baseline_model.train_losses, 'b-', label='基线-训练', linewidth=2)
    axes[0, 0].plot(baseline_model.test_losses, 'b--', label='基线-测试', linewidth=2)
    axes[0, 0].plot(optimized_model.train_losses, 'r-', label='优化-训练', linewidth=2)
    axes[0, 0].plot(optimized_model.test_losses, 'r--', label='优化-测试', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('损失曲线对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线对比
    axes[0, 1].plot(baseline_model.train_accuracies, 'b-', label='基线-训练', linewidth=2)
    axes[0, 1].plot(baseline_model.test_accuracies, 'b--', label='基线-测试', linewidth=2)
    axes[0, 1].plot(optimized_model.train_accuracies, 'r-', label='优化-训练', linewidth=2)
    axes[0, 1].plot(optimized_model.test_accuracies, 'r--', label='优化-测试', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('准确率曲线对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 学习率曲线（仅优化模型）
    if hasattr(optimized_model, 'learning_rates') and optimized_model.learning_rates:
        axes[0, 2].plot(optimized_model.learning_rates, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('学习率')
        axes[0, 2].set_title('学习率调度')
        axes[0, 2].grid(True, alpha=0.3)

    # 预测样本对比
    num_samples = 6
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    samples = X_test[indices]
    labels = y_test[indices]

    baseline_preds = baseline_model.predict(samples)
    optimized_preds = optimized_model.predict(samples)
    true_labels = np.argmax(labels, axis=1)

    for i in range(num_samples):
        ax = axes[1, i] if i < 3 else axes[1, i]
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')

        baseline_correct = baseline_preds[i] == true_labels[i]
        optimized_correct = optimized_preds[i] == true_labels[i]

        title = f'真实: {true_labels[i]}\n'
        title += f'基线: {baseline_preds[i]} {"[OK]" if baseline_correct else "[X]"}\n'
        title += f'优化: {optimized_preds[i]} {"[OK]" if optimized_correct else "[X]"}'

        color = 'green' if optimized_correct else 'red'
        ax.set_title(title, fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("对比结果已保存到 model_comparison.png")
    plt.show()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("纯 NumPy 实现的 CNN - 优化版本")
    print("="*80)

    np.random.seed(42)

    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data(
        reduce_size=True,
        apply_augmentation=False,  # 数据增强比较耗时，可选
        standardize=True
    )

    print("\n" + "="*80)
    print("训练优化版CNN模型")
    print("="*80)

    # 创建并训练优化模型
    optimized_model = CNNOptimized(
        use_dropout=True,
        use_batchnorm=True,
        weight_decay=1e-4
    )

    optimized_model.train_model(
        X_train, y_train, X_test, y_test,
        epochs=20,
        batch_size=64,
        initial_lr=0.001,
        use_lr_schedule=True,
        early_stopping_patience=5
    )

    print("\n所有任务完成！")


if __name__ == "__main__":
    main()
