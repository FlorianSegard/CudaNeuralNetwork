import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Constants
BATCH_SIZE = 1
INPUT_FEATURES = 784  # 28x28 pixels
HIDDEN_FEATURES_1 = 20  # First hidden layer
OUTPUT_FEATURES = 10  # 10 digits
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 10
EPOCHS = 10
LEARNING_RATE = 0.001
GRAD_CLIP_VALUE = 1.0

# Define the Neural Network class
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_FEATURES, OUTPUT_FEATURES, bias=None)
        #self.fc2 = nn.Linear(HIDDEN_FEATURES_1, OUTPUT_FEATURES, bias=None)
        self.softmax = nn.Softmax(dim=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.softmax(x)
        return x

# Custom Optimizer
class CustomOptimizer:
    def __init__(self, parameters, learning_rate):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    # Clip gradients
                    param.grad.clamp_(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)
                    # Update parameters
                    param -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Subset the dataset to match the TRAIN_SAMPLES and TEST_SAMPLES constants
train_subset = Subset(train_dataset, range(TRAIN_SAMPLES))
test_subset = Subset(test_dataset, range(TEST_SAMPLES))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and custom optimizer
model = MNISTModel()
criterion = nn.MSELoss()
optimizer = CustomOptimizer(model.parameters(), LEARNING_RATE)

# Train the model
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Flatten the images
        images = images.view(-1, INPUT_FEATURES)

        # Convert labels to one-hot encoding
        labels_one_hot = torch.zeros((labels.size(0), OUTPUT_FEATURES))
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels_one_hot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, INPUT_FEATURES)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
