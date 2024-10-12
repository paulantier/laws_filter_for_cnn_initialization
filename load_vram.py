import numpy as np
import torchvision
import torchvision.transforms as transforms

# Transformation pour convertir en tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger le dataset CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convertir les datasets en numpy arrays
train_data = np.array([np.array(image) for image, _ in train_dataset])
train_labels = np.array([label for _, label in train_dataset])

test_data = np.array([np.array(image) for image, _ in test_dataset])
test_labels = np.array([label for _, label in test_dataset])

# Sauvegarder dans des fichiers npy
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_labels)

print("Les données CIFAR-10 ont été sauvegardées dans des fichiers npy.")
