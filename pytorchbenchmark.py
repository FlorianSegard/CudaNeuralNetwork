import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# Hyperparameters
BATCH_SIZE = 32
INPUT_FEATURES = 784  # 28x28
HIDDEN_FEATURES = 20
OUTPUT_FEATURES = 10
EPOCHS = 30
LEARNING_RATE = 0.01
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 1000

# Define the Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_FEATURES, OUTPUT_FEATURES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Accuracy function
def compute_accuracy(predictions, labels):
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == labels).sum().item()
    return correct / labels.size(0)

# Log file setup
log_file_path = "pytorch_training_log.txt"
log_file = open(log_file_path, "w")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor in [0, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])
full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
full_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Select subsets to match the required sample sizes
train_dataset = Subset(full_train_dataset, range(TRAIN_SAMPLES))
test_dataset = Subset(full_test_dataset, range(TEST_SAMPLES))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training Loop
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    batch_count = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        total_loss += loss.item()
        total_accuracy += compute_accuracy(outputs, labels)
        batch_count += 1

    avg_loss = total_loss / batch_count
    avg_accuracy = (total_accuracy / batch_count) * 100.0
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    log_file.write(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%\n")

# Training time
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")
log_file.write(f"Total Training Time: {total_training_time:.2f} seconds\n")

# Evaluation on Test Set
model.eval()
test_loss = 0.0
test_accuracy = 0.0
test_batch_count = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        test_accuracy += compute_accuracy(outputs, labels)
        test_batch_count += 1

final_test_loss = test_loss / test_batch_count
final_test_accuracy = (test_accuracy / test_batch_count) * 100.0

print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.2f}%")
log_file.write("Final Test Results:\n")
log_file.write(f"Loss: {final_test_loss:.4f}\n")
log_file.write(f"Accuracy: {final_test_accuracy:.2f}%\n")

# Close the log file
log_file.close()
