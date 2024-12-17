import matplotlib.pyplot as plt
import pandas as pd
import re

# Path to the log file
log_file_path = "build/training_log.txt"

# Function to parse the log file and extract training metrics
def parse_log_file(file_path):
    epochs = []
    losses = []
    accuracies = []
    total_training_time = None
    final_loss = None
    final_accuracy = None
    
    with open(file_path, "r") as file:
        for line in file:
            # Match training epoch log
            epoch_match = re.match(r"Epoch: (\d+), Loss: ([\d.]+), Accuracy: ([\d.]+)%", line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                accuracy = float(epoch_match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(accuracy)
            
            # Match total training time
            time_match = re.match(r"Total Training Time: ([\d.]+) seconds", line)
            if time_match:
                total_training_time = float(time_match.group(1))
            
            # Match final test results
            final_loss_match = re.match(r"Loss: ([\d.]+)", line)
            final_accuracy_match = re.match(r"Accuracy: ([\d.]+)%", line)
            if final_loss_match:
                final_loss = float(final_loss_match.group(1))
            if final_accuracy_match:
                final_accuracy = float(final_accuracy_match.group(1))
    
    return {
        "epochs": epochs,
        "losses": losses,
        "accuracies": accuracies,
        "total_training_time": total_training_time,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy
    }

# Parse the log file
metrics = parse_log_file(log_file_path)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(metrics["epochs"], metrics["losses"], marker='o', linestyle='-', label='Training Loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("training_loss.png", format="png", dpi=300)  # Save the figure
plt.close()  # Close the figure

# Plot training accuracy
plt.figure(figsize=(10, 6))
plt.plot(metrics["epochs"], metrics["accuracies"], marker='o', linestyle='-', label='Training Accuracy')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()
plt.savefig("training_accuracy.png", format="png", dpi=300)  # Save the figure
plt.close()  # Close the figure

# Print final metrics
print(f"Total Training Time: {metrics['total_training_time']} seconds")
print(f"Final Test Loss: {metrics['final_loss']}")
print(f"Final Test Accuracy: {metrics['final_accuracy']}%")
