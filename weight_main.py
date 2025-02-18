import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 配置项
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_MODELS = 6  # 假设有 6 个模型


# 数据预处理与加载
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets_dict = {
        'MNIST': datasets.MNIST,
        'DTD': datasets.ImageFolder,
        'EuroSAT': datasets.ImageFolder,
        'GTSRB': datasets.ImageFolder,
        'SUN397': datasets.ImageFolder,
        'SVHN': datasets.SVHN
    }

    train_datasets = {
        name: dataset(root='./datasets', train=True, download=True, transform=transform)
        for name, dataset in datasets_dict.items()
    }

    return {
        name: DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for name, dataset in train_datasets.items()
    }


# 任务向量生成
class TaskVector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskVector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 元模型（MetaModel）设计
class MetaModel(nn.Module):
    def __init__(self, num_models, task_vector_dim):
        super(MetaModel, self).__init__()
        self.attention_layer = nn.Linear(num_models * task_vector_dim, num_models)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        weights = self.attention_layer(x)
        weights = self.softmax(weights)
        return weights


# 训练元模型
def train_meta_model(meta_model, train_loaders, task_vectors, model_dict, epochs=EPOCHS):
    optimizer = optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        meta_model.train()
        for dataset_name, loader in train_loaders.items():
            model = model_dict[dataset_name]
            task_vector = task_vectors[dataset_name]

            for images, labels in loader:
                model_output = model(images)
                task_output = task_vector(labels.float())

                # 拼接模型输出和任务向量
                combined_input = torch.cat((model_output, task_output), dim=1)

                # 生成权重
                weights = meta_model(combined_input)

                # 计算加权输出
                weighted_output = torch.sum(weights.unsqueeze(1) * model_output, dim=0)

                # 计算损失
                loss = criterion(weighted_output, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# 推理阶段（融合模型输出）
def inference(meta_model, model_dict, task_vectors, test_loader):
    meta_model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            all_model_outputs = []
            all_task_vectors = []

            # 获取每个模型的输出及任务向量
            for dataset_name, model in model_dict.items():
                model_output = model(images)
                task_vector = task_vectors[dataset_name](labels.float())

                all_model_outputs.append(model_output)
                all_task_vectors.append(task_vector)

            # 将所有模型输出和任务向量拼接
            combined_input = torch.cat(all_model_outputs + all_task_vectors, dim=1)

            # 生成权重
            weights = meta_model(combined_input)

            # 加权输出
            weighted_output = torch.sum(weights.unsqueeze(1) * torch.stack(all_model_outputs), dim=0)

            # 使用 softmax 进行分类
            preds = torch.argmax(weighted_output, dim=1)
            print(f"Predictions: {preds}, Ground Truth: {labels}")


# 主函数 - 初始化与训练
def main():
    # 加载数据
    train_loaders = get_data_loaders()

    # 假设的模型字典（使用预训练模型作为示例）
    model_dict = {
        'MNIST': models.resnet18(pretrained=True),
        'DTD': models.resnet18(pretrained=True),
        'EuroSAT': models.resnet18(pretrained=True),
        'GTSRB': models.resnet18(pretrained=True),
        'SUN397': models.resnet18(pretrained=True),
        'SVHN': models.resnet18(pretrained=True)
    }

    # 创建任务向量
    task_vectors = {name: TaskVector(10, 64) for name in model_dict.keys()}  # 假设输入10维，输出64维

    # 创建元模型
    meta_model = MetaModel(num_models=NUM_MODELS, task_vector_dim=64)

    # 训练元模型
    train_meta_model(meta_model, train_loaders, task_vectors, model_dict)

    # 假设有一个测试数据加载器
    test_loader = DataLoader(datasets.MNIST(root='./datasets', train=False, download=True, transform=transform),
                             batch_size=BATCH_SIZE)

    # 推理阶段
    inference(meta_model, model_dict, task_vectors, test_loader)


if __name__ == "__main__":
    main()
