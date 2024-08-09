import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Carregar o dataset do arquivo CSV
df = pd.read_csv('dataset.csv')

# Separar dados e rótulos
data = df[['X', 'Y', 'Z']].values
labels = df['Label'].values

# Separar dados das duas classes
class1_data = data[labels == 0]
class2_data = data[labels == 1]

# LDA - Linear Discriminant Analysis

# Separar dados das classes
mean_class1 = np.mean(class1_data, axis=0).reshape(3, 1)
mean_class2 = np.mean(class2_data, axis=0).reshape(3, 1)
overall_mean = np.mean(data, axis=0).reshape(3, 1)

# Calcular as matrizes de dispersão (Scatter Matrices)
S_W = np.zeros((3, 3))
S_B = np.zeros((3, 3))

for class_data, mean_class in [(class1_data, mean_class1), (class2_data, mean_class2)]:
    # Scatter matrix dentro das classes
    class_scatter = np.zeros((3, 3))
    for point in class_data:
        point = point.reshape(3, 1)
        class_scatter += (point - mean_class).dot((point - mean_class).T)
    S_W += class_scatter
    
    # Scatter matrix entre as classes
    n = class_data.shape[0]
    mean_diff = mean_class - overall_mean
    S_B += n * (mean_diff).dot(mean_diff.T)

# Calcular autovalores e autovetores da matriz Sw-1Sb
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Ordenar autovetores pela magnitude dos autovalores em ordem decrescente
sorted_indices = np.argsort(np.abs(eigen_values))[::-1]
eigen_vectors = eigen_vectors[:, sorted_indices]

# Selecionar os autovetores correspondentes aos maiores autovalores
W = eigen_vectors[:, :2]  # Projetar em um subespaço 2D

# Projetar os dados no novo espaço
data_lda = data.dot(W)

# Plotar os dados projetados
plt.figure()
plt.scatter(data_lda[labels == 0, 0], data_lda[labels == 0, 1], c='blue', label='Class 1')
plt.scatter(data_lda[labels == 1, 0], data_lda[labels == 1, 1], c='red', label='Class 2')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.title('LDA: Data projection')
plt.show()
