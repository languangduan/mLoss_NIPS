import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_MODELS = 6  # Assume there are 6 models


# Data preprocessing and loading
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


# Task vector generation
class TaskVector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TaskVector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Meta-model design
class MetaModel(nn.Module):
    def __init__(self, num_models, task_vector_dim):
        super(MetaModel, self).__init__()
        self.attention_layer = nn.Linear(num_models * task_vector_dim, num_models)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        weights = self.attention_layer(x)
        weights = self.softmax(weights)
        return weights


# Meta-model training
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

                # Concatenate model output and task vector
                combined_input = torch.cat((model_output, task_output), dim=1)

                # Generate weights
                weights = meta_model(combined_input)

                # Weighted output
                weighted_output = torch.sum(weights.unsqueeze(1) * model_output, dim=0)

                # Compute loss
                loss = criterion(weighted_output, labels)

                # Backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# Inference phase (merge model outputs)
def inference(meta_model, model_dict, task_vectors, test_loader):
    meta_model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            all_model_outputs = []
            all_task_vectors = []

            # Get outputs and task vectors from each model
            for dataset_name, model in model_dict.items():
                model_output = model(images)
                task_vector = task_vectors[dataset_name](labels.float())

                all_model_outputs.append(model_output)
                all_task_vectors.append(task_vector)

            # Concatenate all model outputs and task vectors
            combined_input = torch.cat(all_model_outputs + all_task_vectors, dim=1)

            # Generate weights
            weights = meta_model(combined_input)

            # Weighted output
            weighted_output = torch.sum(weights.unsqueeze(1) * torch.stack(all_model_outputs), dim=0)

            # Use softmax for classification
            preds = torch.argmax(weighted_output, dim=1)
            print(f"Predictions: {preds}, Ground Truth: {labels}")


# Main function - initialization and training
def main():
    # Load data
    train_loaders = get_data_loaders()

    # Example model dictionary (using pretrained models as example)
    model_dict = {
        'MNIST': models.resnet18(pretrained=True),
        'DTD': models.resnet18(pretrained=True),
        'EuroSAT': models.resnet18(pretrained=True),
        'GTSRB': models.resnet18(pretrained=True),
        'SUN397': models.resnet18(pretrained=True),
        'SVHN': models.resnet18(pretrained=True)
    }

    # Create task vectors
    task_vectors = {name: TaskVector(10, 64) for name in model_dict.keys()}  # Assume input 10-dim, output 64-dim

    # Create meta-model
    meta_model = MetaModel(num_models=NUM_MODELS, task_vector_dim=64)

    # Train meta-model
    train_meta_model(meta_model, train_loaders, task_vectors, model_dict)

    # Assume a test data loader exists
    test_loader = DataLoader(datasets.MNIST(root='./datasets', train=False, download=True, transform=transform),
                             batch_size=BATCH_SIZE)

    # Inference phase
    inference(meta_model, model_dict, task_vectors, test_loader)


if __name__ == "__main__":
    main()
