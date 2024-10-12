import matplotlib.pyplot as plt

# Fonction pour lire les données depuis un fichier texte
def read_data_from_file(filename):
    epochs_batches = []
    accuracies = []

    with open(filename, 'r') as file:
        for idx, line in enumerate(file):
            if idx>0:
                epoch_batch, accuracy = map(float, line.strip().split(', '))
                epochs_batches.append(int(epoch_batch))  # Convertir en int
                accuracies.append(accuracy)

    return epochs_batches, accuracies

# Lire les données

plt.figure(figsize=(10, 6))

filename = 'initFalse_fixFalse.txt'  # Nom du fichier contenant les données
epochs_batches, accuracies = read_data_from_file(filename)
plt.plot(epochs_batches, accuracies, marker='x', linestyle='-', color='b')

filename = 'initTrue_fixFalse.txt'  # Nom du fichier contenant les données
epochs_batches, accuracies = read_data_from_file(filename)
plt.plot(epochs_batches, accuracies, marker='x', linestyle='-', color='r')

filename = 'initTrue_fixTrue.txt'  # Nom du fichier contenant les données
epochs_batches, accuracies = read_data_from_file(filename)
plt.plot(epochs_batches, accuracies, marker='x', linestyle='-', color='g')


plt.title('Évolution de l\'accuracy')
plt.xlabel('Nombre de batch')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.ylim(0, 100)  # Ajuste la limite de l'axe y si nécessaire
plt.show()
