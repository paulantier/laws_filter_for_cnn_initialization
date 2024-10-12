import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import datetime


# Charger les données en mémoire
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# Convertir en tensors et transférer en GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torch.tensor(train_data).float().to(device)
train_labels = torch.tensor(train_labels).long().to(device)
test_data = torch.tensor(test_data).float().to(device)
test_labels = torch.tensor(test_labels).long().to(device)

# Architecture LeNet-5 adaptée pour CIFAR-10
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_funky_init(nn.Module):
    def __init__(self):
        super(LeNet5_funky_init, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

        # Initialisation manuelle de 4 noyaux de convolution pour la première couche
        self.initialize_filters()

    def initialize_filters(self):
        # Définition des vecteurs de Laws
        L5 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        E5 = torch.tensor([-1, -2, 0, 2, 1], dtype=torch.float32)
        S5 = torch.tensor([-1, 0, 2, 0, -1], dtype=torch.float32)
        R5 = torch.tensor([1, -4, 6, -4, 1], dtype=torch.float32)
        W5 = torch.tensor([-1, 2, 0, -2, 1], dtype=torch.float32)

        # Création des filtres de Laws (25 filtres 5x5)
        laws_filters = [
            torch.outer(L5, L5),  # L5L5
            torch.outer(L5, E5),  # L5E5
            torch.outer(L5, S5),  # L5S5
            torch.outer(L5, R5),  # L5R5
            torch.outer(L5, W5),  # L5W5
            
            torch.outer(E5, L5),  # E5L5
            torch.outer(E5, E5),  # E5E5
            torch.outer(E5, S5),  # E5S5
            torch.outer(E5, R5),  # E5R5
            torch.outer(E5, W5),  # E5W5
            
            torch.outer(S5, L5),  # S5L5
            torch.outer(S5, E5),  # S5E5
            torch.outer(S5, S5),  # S5S5
            torch.outer(S5, R5),  # S5R5
            torch.outer(S5, W5),  # S5W5
            
            torch.outer(R5, L5),  # R5L5
            torch.outer(R5, E5),  # R5E5
            torch.outer(R5, S5),  # R5S5
            torch.outer(R5, R5),  # R5R5
            torch.outer(R5, W5),  # R5W5
            
            torch.outer(W5, L5),  # W5L5
            torch.outer(W5, E5),  # W5E5
            torch.outer(W5, S5),  # W5S5
            torch.outer(W5, R5),  # W5R5
            torch.outer(W5, W5)   # W5W5
        ]

        # Empiler les filtres pour les utiliser dans les convolutions
        laws_filters = torch.stack(laws_filters)

        # Préparation pour les trois canaux (R, G, B)
        laws_filters = laws_filters.unsqueeze(1)  # (25, 1, 5, 5)
        laws_filters = laws_filters.expand(-1, 3, -1, -1)  # (25, 3, 5, 5)

        # Remplir les poids des filtres de la première couche avec les noyaux de Laws
        with torch.no_grad():
            self.conv1.weight[:25] = laws_filters  # 25 premiers filtres de Laws
            # Initialiser les 39 filtres restants de manière aléatoire
            nn.init.kaiming_normal_(self.conv1.weight[25:], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_funky_fix_init(nn.Module): 
    def __init__(self):
        super(LeNet5_funky_fix_init, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

        # Initialisation manuelle de 4 noyaux de convolution pour la première couche
        self.initialize_filters()

    def initialize_filters(self):
        # Définition des vecteurs de Laws
        L5 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        E5 = torch.tensor([-1, -2, 0, 2, 1], dtype=torch.float32)
        S5 = torch.tensor([-1, 0, 2, 0, -1], dtype=torch.float32)
        R5 = torch.tensor([1, -4, 6, -4, 1], dtype=torch.float32)
        W5 = torch.tensor([-1, 2, 0, -2, 1], dtype=torch.float32)

        # Création des filtres de Laws (25 filtres 5x5)
        laws_filters = [
            torch.outer(L5, L5),  # L5L5
            torch.outer(L5, E5),  # L5E5
            torch.outer(L5, S5),  # L5S5
            torch.outer(L5, R5),  # L5R5
            torch.outer(L5, W5),  # L5W5
            
            torch.outer(E5, L5),  # E5L5
            torch.outer(E5, E5),  # E5E5
            torch.outer(E5, S5),  # E5S5
            torch.outer(E5, R5),  # E5R5
            torch.outer(E5, W5),  # E5W5
            
            torch.outer(S5, L5),  # S5L5
            torch.outer(S5, E5),  # S5E5
            torch.outer(S5, S5),  # S5S5
            torch.outer(S5, R5),  # S5R5
            torch.outer(S5, W5),  # S5W5
            
            torch.outer(R5, L5),  # R5L5
            torch.outer(R5, E5),  # R5E5
            torch.outer(R5, S5),  # R5S5
            torch.outer(R5, R5),  # R5R5
            torch.outer(R5, W5),  # R5W5
            
            torch.outer(W5, L5),  # W5L5
            torch.outer(W5, E5),  # W5E5
            torch.outer(W5, S5),  # W5S5
            torch.outer(W5, R5),  # W5R5
            torch.outer(W5, W5)   # W5W5
        ]

        # Empiler les filtres pour les utiliser dans les convolutions
        laws_filters = torch.stack(laws_filters)

        # Préparation pour les trois canaux (R, G, B)
        laws_filters = laws_filters.unsqueeze(1)  # (25, 1, 5, 5)
        laws_filters = laws_filters.expand(-1, 3, -1, -1)  # (25, 3, 5, 5)

        # Remplir les poids des filtres de la première couche avec les noyaux de Laws
        with torch.no_grad():
            self.conv1.weight[:25] = laws_filters  # 25 premiers filtres de Laws
            self.conv1.weight.requires_grad = False
            # Initialiser les 39 filtres restants de manière aléatoire
            nn.init.kaiming_normal_(self.conv1.weight[25:], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_2layers(nn.Module):
    def __init__(self):
        super(LeNet5_2layers, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(1600, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

        # Initialisation manuelle de 4 noyaux de convolution pour la première couche
        self.initialize_filters()

    def initialize_filters(self):
        # Définition des vecteurs de Laws
        L5 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        E5 = torch.tensor([-1, -2, 0, 2, 1], dtype=torch.float32)
        S5 = torch.tensor([-1, 0, 2, 0, -1], dtype=torch.float32)
        R5 = torch.tensor([1, -4, 6, -4, 1], dtype=torch.float32)
        W5 = torch.tensor([-1, 2, 0, -2, 1], dtype=torch.float32)

        # Création des filtres de Laws (25 filtres 5x5)
        laws_filters = [
            torch.outer(L5, L5),  # L5L5
            torch.outer(L5, E5),  # L5E5
            torch.outer(L5, S5),  # L5S5
            torch.outer(L5, R5),  # L5R5
            torch.outer(L5, W5),  # L5W5
            
            torch.outer(E5, L5),  # E5L5
            torch.outer(E5, E5),  # E5E5
            torch.outer(E5, S5),  # E5S5
            torch.outer(E5, R5),  # E5R5
            torch.outer(E5, W5),  # E5W5
            
            torch.outer(S5, L5),  # S5L5
            torch.outer(S5, E5),  # S5E5
            torch.outer(S5, S5),  # S5S5
            torch.outer(S5, R5),  # S5R5
            torch.outer(S5, W5),  # S5W5
            
            torch.outer(R5, L5),  # R5L5
            torch.outer(R5, E5),  # R5E5
            torch.outer(R5, S5),  # R5S5
            torch.outer(R5, R5),  # R5R5
            torch.outer(R5, W5),  # R5W5
            
            torch.outer(W5, L5),  # W5L5
            torch.outer(W5, E5),  # W5E5
            torch.outer(W5, S5),  # W5S5
            torch.outer(W5, R5),  # W5R5
            torch.outer(W5, W5)   # W5W5
        ]

        # Empiler les filtres pour les utiliser dans les convolutions
        laws_filters = torch.stack(laws_filters)

        # Préparation pour les trois canaux (R, G, B)
        laws_filters = laws_filters.unsqueeze(1)  # (25, 1, 5, 5)
        laws_filters1 = laws_filters.clone()
        laws_filters2 = laws_filters.clone()
        laws_filters1 = laws_filters1.expand(-1, 3, -1, -1)  # (25, 3, 5, 5)
        laws_filters2 = laws_filters2.expand(-1, 64, -1, -1)  # (25, 3, 5, 5)
        # Remplir les poids des filtres de la première couche avec les noyaux de Laws
        with torch.no_grad():
            #self.conv1.weight[:25] = laws_filters1  # 25 premiers filtres de Laws
            self.conv2.weight[:25] = laws_filters2
            # Initialiser les 39 filtres restants de manière aléatoire
            #nn.init.kaiming_normal_(self.conv1.weight[25:], mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight[25:], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Paramètres d'entraînement
batch_size = 64
learning_rate = 0.001
num_epochs = 10
test_every = 0.25
initalize = False
fixed_init = False
test_2layers = True

# Modèle, perte, optimiseur
if initalize:
    if fixed_init :
        model = LeNet5_funky_fix_init().to(device)
    else:
        model = LeNet5_funky_init().to(device)
else:
    model = LeNet5().to(device)

if test_2layers:
    model = LeNet5_2layers().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Fonction pour tester le modèle
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Entraînement du modèle
train_time_start = time.time()

date_time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
accuracy_file = f"accuracy_log_{date_time_now}.txt"

# Ouvrir le fichier en mode écriture
with open(accuracy_file, 'w') as f:
    # Écrire un en-tête dans le fichier
    f.write("Epoch-Batch, Accuracy\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_time_start = time.time()

        # Mélange aléatoire des données pour chaque époque
        indices = torch.randperm(train_data.size(0))
        train_data = train_data[indices]
        train_labels = train_labels[indices]

        # Nombre de tests à effectuer par époque
        num_tests = int(1 / test_every)  # Par exemple, 10 si test_every = 0.1
        no_test=0

        # Mini-batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Époque {epoch + 1}/{num_epochs}"):
            inputs = train_data[i:i + batch_size]
            labels = train_labels[i:i + batch_size]

            # Forward + backward + optimiseur
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if no_test <= num_tests:
                # Effectuer un test à chaque intervalle spécifié
                if (i / len(train_data)) > no_test/num_tests:  # Vérifie si on doit tester
                    no_test+=1
                    accuracy = test_model()
                    print(f"\nPrécision après {epoch + 1} époques, mini-batch {i // batch_size + 1}: {accuracy:.2f}%")
                    
                    # Enregistrer la précision dans le fichier
                    f.write(f"{no_test + epoch * num_tests}, {accuracy:.2f}\n")

        epoch_time_end = time.time()
        epoch_duration = epoch_time_end - epoch_time_start
        print(f'[{epoch + 1}/{num_epochs}] Perte: {running_loss / len(train_data):.4f} - Temps: {epoch_duration:.2f}s')

train_time_end = time.time()
train_duration = train_time_end - train_time_start
print(f'Temps total d\'entraînement: {train_duration:.2f} secondes')