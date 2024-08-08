import numpy as np
import pandas as pd

# Definir a média e a matriz de covariância para a primeira classe
mean_class1 = np.array([0, 0, 0])
cov_class1 = np.array([[0.5, 0, 0],
                       [0, 0.5, 0],
                       [0, 0, 0.01]])

# Gerar 100 exemplos
class1_data = np.random.multivariate_normal(mean_class1, cov_class1, 100)

# Definir a matriz de covariância para a segunda classe
cov_class2 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0.01]])

# Definir as médias para os 8 grupos
a = 6  # Você pode ajustar esse valor conforme necessário
means_class2 = [
    [a, 0, 0],
    [a/2, a/2, 0],
    [0, a, 0],
    [-a/2, a/2, 0],
    [-a, 0, 0],
    [-a/2, -a/2, 0],
    [0, -a, 0],
    [a/2, -a/2, 0]
]

# Gerar os dados para os 8 grupos
class2_data = []
for mean in means_class2:
    group_data = np.random.multivariate_normal(mean, cov_class2, 100)
    class2_data.append(group_data)

# Concatenar todos os grupos
class2_data = np.vstack(class2_data)

# Adicionar rótulos
class1_labels = np.zeros((100, 1))  # Rótulo 0 para a primeira classe
class2_labels = np.ones((800, 1))   # Rótulo 1 para a segunda classe

# Combinar dados e rótulos
data = np.vstack([class1_data, class2_data])
labels = np.vstack([class1_labels, class2_labels])

# Embaralhar os dados
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Criar um DataFrame com os dados e rótulos
df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
df['Label'] = labels

# Salvar o DataFrame em um arquivo CSV
df.to_csv('dataset.csv', index=False)
