import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def generate_clusters(n_samples, n_features, n_clusters):
    # Генерация данных с использованием make_blobs
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return X, y

def save_to_csv(X, y, output_file):
    # Сохранение данных в CSV файл
    df = pd.DataFrame(data=X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['label'] = y
    df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')

def plot_clusters(X, y):
    # Построение графика кластеров
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    plt.title('Clustered Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Cluster generation script')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of samples')
    parser.add_argument('--features', type=int, default=2, help='Number of features')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--output', type=str, default='clustered_data.csv', help='Output CSV file name')

    # Получение аргументов
    args = parser.parse_args()

    # Генерация кластеров
    X, y = generate_clusters(args.samples, args.features, args.clusters)

    # Сохранение в CSV
    save_to_csv(X, y, args.output)

    # Построение графика
    plot_clusters(X, y)

if __name__ == '__main__':
    main()
