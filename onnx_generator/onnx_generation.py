import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.linear1 = nn.Linear(784, 180)  # 784 = 28*28
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(180, 10)
        self.softmax = nn.Softmax(dim=1)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Constants
    TRAIN_SAMPLES = 10000
    BATCH_SIZE = 32
    NUM_EPOCHS = 30

    # Simple transform - just convert to tensor, no normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # Create subset
    indices = np.random.choice(len(train_dataset), TRAIN_SAMPLES, replace=False)
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

    # Export untrained model
    model.to("cpu")
    # Create properly shaped input tensor (already flattened)
    dummy_input = torch.randn(1, 784)  # batch_size=1, flattened_input=784

    torch.onnx.export(
        model,
        dummy_input,
        "mnist_untrained.onnx",
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )

    # Training loop
    model.to(device)
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # Flatten the images before passing to model
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Export trained model
    model.to("cpu")
    torch.onnx.export(
        model,
        dummy_input,
        "mnist_trained.onnx",
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    print("Exported trained model")

if __name__ == '__main__':
    train()