"""
纯 NumPy 实现的卷积神经网络（CNN）用于 MNIST 手写数字分类
仅使用 numpy 实现核心逻辑，tensorflow 仅用于加载数据集
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List
import os

# 设置 TensorFlow 镜像源（用于加载 MNIST 数据集）
os.environ['TF_KERAS_DATASETS_MIRROR'] = 'https://mirrors.aliyun.com/tensorflow/'


class Conv2D:
    """
    卷积层实现
    支持自定义输入通道数、输出通道数、卷积核大小、步幅和填充
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        """
        初始化卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步幅
            padding: 填充
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 使用 He 初始化卷积核权重
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                      np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))

        # 用于存储梯度
        self.dweights = None
        self.dbias = None

        # 用于反向传播
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, in_channels, height, width)

        返回:
            输出张量，形状为 (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = x.shape

        # 应用填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                 (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # 执行卷积操作
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # 提取感受野
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]

                # 对每个输出通道进行卷积
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(
                        receptive_field * self.weights[k, :, :, :],
                        axis=(1, 2, 3)
                    ) + self.bias[k]

        # 保存用于反向传播
        self.cache = x_padded

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
            dout: 上游梯度，形状为 (batch_size, out_channels, out_height, out_width)

        返回:
            对输入的梯度，形状为 (batch_size, in_channels, height, width)
        """
        x_padded = self.cache
        batch_size, in_channels, padded_height, padded_width = x_padded.shape
        _, out_channels, out_height, out_width = dout.shape

        # 初始化梯度
        dx_padded = np.zeros_like(x_padded)
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

        # 计算梯度
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # 提取感受野
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]

                for k in range(out_channels):
                    # 权重梯度
                    self.dweights[k] += np.sum(
                        receptive_field * dout[:, k:k+1, i:i+1, j:j+1],
                        axis=0
                    )
                    # 偏置梯度
                    self.dbias[k] += np.sum(dout[:, k, i, j])
                    # 输入梯度
                    dx_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * dout[:, k:k+1, i:i+1, j:j+1]

        # 移除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        return dx

    def update(self, learning_rate: float):
        """更新权重和偏置"""
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias


class MaxPool2D:
    """
    最大池化层实现
    支持自定义池化大小和步幅
    """

    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        初始化池化层

        参数:
            pool_size: 池化窗口大小
            stride: 步幅
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, channels, height, width)

        返回:
            输出张量，形状为 (batch_size, channels, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # 计算输出尺寸
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width))

        # 执行最大池化
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # 提取池化窗口
                pool_region = x[:, :, h_start:h_end, w_start:w_end]

                # 取最大值
                output[:, :, i, j] = np.max(pool_region, axis=(2, 3))

        # 保存用于反向传播
        self.cache = x

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
            dout: 上游梯度，形状为 (batch_size, channels, out_height, out_width)

        返回:
            对输入的梯度，形状为 (batch_size, channels, height, width)
        """
        x = self.cache
        batch_size, channels, height, width = x.shape
        _, _, out_height, out_width = dout.shape

        # 初始化梯度
        dx = np.zeros_like(x)

        # 反向传播梯度
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # 提取池化窗口
                pool_region = x[:, :, h_start:h_end, w_start:w_end]

                # 创建掩码，标记最大值位置
                for b in range(batch_size):
                    for c in range(channels):
                        # 找到最大值位置
                        mask = (pool_region[b, c] == np.max(pool_region[b, c]))
                        # 将梯度分配到最大值位置
                        dx[b, c, h_start:h_end, w_start:w_end] += \
                            mask * dout[b, c, i, j] / np.sum(mask)

        return dx


class Dense:
    """
    全连接层实现
    支持自定义输入和输出维度
    """

    def __init__(self, input_size: int, output_size: int):
        """
        初始化全连接层

        参数:
            input_size: 输入维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.output_size = output_size

        # 使用 He 初始化权重
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

        # 用于存储梯度
        self.dweights = None
        self.dbias = None

        # 用于反向传播
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, input_size)

        返回:
            输出张量，形状为 (batch_size, output_size)
        """
        # 保存输入用于反向传播
        self.cache = x

        # 计算输出
        output = np.dot(x, self.weights) + self.bias

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
            dout: 上游梯度，形状为 (batch_size, output_size)

        返回:
            对输入的梯度，形状为 (batch_size, input_size)
        """
        x = self.cache

        # 计算梯度
        self.dweights = np.dot(x.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.weights.T)

        return dx

    def update(self, learning_rate: float):
        """更新权重和偏置"""
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias


class ReLU:
    """ReLU 激活函数"""

    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量

        返回:
            输出张量
        """
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
            dout: 上游梯度

        返回:
            对输入的梯度
        """
        x = self.cache
        dx = dout * (x > 0)
        return dx


class Softmax:
    """Softmax 激活函数（数值稳定实现）"""

    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, num_classes)

        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        # 数值稳定：减去最大值
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = output
        return output


class CrossEntropyLoss:
    """交叉熵损失函数"""

    def __init__(self):
        self.cache = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        前向传播

        参数:
            predictions: 预测值（经过 softmax），形状为 (batch_size, num_classes)
            targets: 目标值（one-hot 编码），形状为 (batch_size, num_classes)

        返回:
            损失值
        """
        batch_size = predictions.shape[0]

        # 避免 log(0)
        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # 计算交叉熵损失
        loss = -np.sum(targets * np.log(predictions)) / batch_size

        # 保存用于反向传播
        self.cache = (predictions, targets)

        return loss

    def backward(self) -> np.ndarray:
        """
        反向传播

        返回:
            对预测值的梯度
        """
        predictions, targets = self.cache
        batch_size = predictions.shape[0]

        # 计算梯度
        dx = (predictions - targets) / batch_size

        return dx


class CNN:
    """
    卷积神经网络模型
    整合所有层构建完整模型
    """

    def __init__(self):
        """初始化 CNN 模型"""
        # 定义网络结构
        self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # MNIST 图像: 28x28 -> conv1: 28x28 -> pool1: 14x14 -> conv2: 14x14 -> pool2: 7x7
        # 展平后: 16 * 7 * 7 = 784
        self.fc1 = Dense(input_size=16 * 7 * 7, output_size=128)
        self.relu3 = ReLU()

        self.fc2 = Dense(input_size=128, output_size=10)
        self.softmax = Softmax()

        self.loss_fn = CrossEntropyLoss()

        # 用于存储训练历史
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)

        返回:
            预测概率，形状为 (batch_size, 10)
        """
        # 第一个卷积块
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        # 第二个卷积块
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        # 展平
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)

        # 全连接层
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)

        out = self.fc2.forward(out)
        out = self.softmax.forward(out)

        return out

    def backward(self, loss_grad: np.ndarray):
        """
        反向传播

        参数:
            loss_grad: 损失函数的梯度
        """
        # 全连接层反向传播
        dout = self.fc2.backward(loss_grad)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)

        # 还原形状
        dout = dout.reshape(-1, 16, 7, 7)

        # 第二个卷积块反向传播
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)

        # 第一个卷积块反向传播
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)

    def update(self, learning_rate: float):
        """
        更新所有可训练参数

        参数:
            learning_rate: 学习率
        """
        self.conv1.update(learning_rate)
        self.conv2.update(learning_rate)
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.01):
        """
        训练模型

        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        print(f"开始训练，共 {epochs} 轮，每批 {batch_size} 个样本")
        print(f"训练集大小: {n_samples}, 测试集大小: {X_test.shape[0]}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 随机打乱训练数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0

            # 批量训练
            for batch in range(n_batches):
                # 获取当前批次数据
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # 前向传播
                predictions = self.forward(X_batch)

                # 计算损失
                loss = self.loss_fn.forward(predictions, y_batch)
                epoch_loss += loss

                # 反向传播
                loss_grad = self.loss_fn.backward()
                self.backward(loss_grad)

                # 更新参数
                self.update(learning_rate)

                # 显示进度
                if (batch + 1) % 10 == 0 or batch == n_batches - 1:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch+1}/{n_batches}, "
                          f"Loss: {loss:.4f}", end='\r')

            # 计算平均损失
            avg_train_loss = epoch_loss / n_batches

            # 评估训练集和测试集
            train_acc = self.evaluate(X_train, y_train, batch_size)
            test_loss, test_acc = self.evaluate(X_test, y_test, batch_size, return_loss=True)

            # 保存历史
            self.train_losses.append(avg_train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            epoch_time = time.time() - epoch_start_time

            print(f"\n  Epoch {epoch+1}/{epochs} 完成 - "
                  f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2%}, "
                  f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2%}, "
                  f"用时: {epoch_time:.2f}s")
            print("-" * 80)

        total_time = time.time() - start_time
        avg_epoch_time = total_time / epochs

        print(f"\n训练完成！")
        print(f"总用时: {total_time:.2f}s, 平均每轮: {avg_epoch_time:.2f}s")
        print("=" * 80)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 return_loss: bool = False) -> float:
        """
        评估模型

        参数:
            X: 输入数据
            y: 标签
            batch_size: 批量大小
            return_loss: 是否返回损失

        返回:
            准确率（和损失，如果 return_loss=True）
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        correct = 0
        total_loss = 0.0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            # 前向传播
            predictions = self.forward(X_batch)

            # 计算损失
            if return_loss:
                loss = self.loss_fn.forward(predictions, y_batch)
                total_loss += loss * (end_idx - start_idx)

            # 计算准确率
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct += np.sum(pred_labels == true_labels)

        accuracy = correct / n_samples

        if return_loss:
            avg_loss = total_loss / n_samples
            return avg_loss, accuracy

        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        参数:
            X: 输入数据

        返回:
            预测标签
        """
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)


def load_mnist_data(reduce_size: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 MNIST 数据集

    参数:
        reduce_size: 是否减少数据量以便快速测试

    返回:
        (X_train, y_train, X_test, y_test)
    """
    print("正在加载 MNIST 数据集...")

    # 使用 TensorFlow 加载 MNIST 数据集
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 如果需要，减少数据量
    if reduce_size:
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:100]
        y_test = y_test[:100]

    # 归一化到 [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 添加通道维度: (N, H, W) -> (N, 1, H, W)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    # 将标签转换为 one-hot 编码
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    print(f"数据加载完成！")
    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def visualize_results(model: CNN, X_test: np.ndarray, y_test: np.ndarray, num_samples: int = 10):
    """
    可视化测试结果

    参数:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        num_samples: 显示的样本数量
    """
    # 随机选择样本
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    samples = X_test[indices]
    labels = y_test[indices]

    # 预测
    predictions = model.predict(samples)
    true_labels = np.argmax(labels, axis=1)

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(samples[i, 0], cmap='gray')
        axes[i].axis('off')

        # 根据预测是否正确使用不同颜色
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        axes[i].set_title(f'真实: {true_labels[i]}\n预测: {predictions[i]}',
                         color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("测试结果已保存到 test_results.png")
    plt.show()


def visualize_training_curves(model: CNN):
    """
    可视化训练曲线

    参数:
        model: 训练好的模型
    """
    epochs = range(1, len(model.train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(epochs, model.train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, model.test_losses, 'r-', label='测试损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练和测试损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, model.train_accuracies, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, model.test_accuracies, 'r-', label='测试准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('训练和测试准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存到 training_curves.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 80)
    print("纯 NumPy 实现的 CNN - MNIST 手写数字分类")
    print("=" * 80)

    # 设置随机种子以便复现
    np.random.seed(42)

    # 加载数据集（可选择减少数据量以便快速测试）
    X_train, y_train, X_test, y_test = load_mnist_data(reduce_size=True)

    # 创建模型
    print("\n创建 CNN 模型...")
    model = CNN()

    # 训练模型
    print("\n" + "=" * 80)
    model.train(X_train, y_train, X_test, y_test,
                epochs=5, batch_size=32, learning_rate=0.01)

    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(model, X_test, y_test, num_samples=10)
    visualize_training_curves(model)

    print("\n所有任务完成！")


if __name__ == "__main__":
    main()
