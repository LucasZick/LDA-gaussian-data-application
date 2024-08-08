import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar o dataset do arquivo CSV
df = pd.read_csv('dataset.csv')

# Separar dados e rótulos
data = df[['X', 'Y', 'Z']].values
labels = df['Label'].values

# Separar dados das duas classes
class1_data = data[labels == 0]
class2_data = data[labels == 1]

# Criar a figura e o eixo 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar dados da primeira classe
ax.scatter(class1_data[:, 0], class1_data[:, 1], class1_data[:, 2], c='blue', label='Classe 1')

# Plotar dados da segunda classe
ax.scatter(class2_data[:, 0], class2_data[:, 1], class2_data[:, 2], c='red', label='Classe 2')

# Adicionar legendas e rótulos
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
