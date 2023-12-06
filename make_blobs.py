"""
    python make_blobs.py --samples 500 --features 2 --clusters 4 --cluster_std 1 --output clustered_data.csv
    --output_image blobs_plot.jpg
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_clusters(n_samples, n_features, centers, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True,
                      random_state=None, return_centers=False):
    # Генерация данных с использованием make_blobs
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box,
        shuffle=shuffle,
        random_state=random_state,
        return_centers=return_centers
    )
    return X, y


def save_to_csv(X,output_file):
    # Сохранение данных в CSV файл
    df = pd.DataFrame(data=X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')


def plot_clusters(X, y, output_image):
    # Построение графика кластеров
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    plt.title('Clustered Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(output_image, format='jpeg')
    print(f'Image saved to {output_image}')
    plt.show()


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Cluster generation script')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--features', type=int, default=2, help='Number of features')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--cluster_std', type=float, default=1.0, help='Standard deviation of clusters')
    parser.add_argument('--center_box', type=tuple, default=(-10.0, 10.0),
                        help='The bounding box for each cluster center')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the samples')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility')
    parser.add_argument('--return_centers', type=bool, default=False,
                        help='Return cluster centers in addition to data points')
    parser.add_argument('--output', type=str, default='blobs_data.csv', help='Output CSV file name')
    parser.add_argument('--output_image', type=str, default='blobs_plot.jpg', help='Output image file name')

    # Получение аргументов
    args = parser.parse_args()

    # Генерация кластеров
    X, y = generate_clusters(
        args.samples,
        args.features,
        args.clusters,
        cluster_std=args.cluster_std,
        center_box=args.center_box,
        shuffle=args.shuffle,
        random_state=args.random_state,
        return_centers=args.return_centers
    )

    # Сохранение в CSV
    save_to_csv(X, args.output)

    # Построение графика
    plot_clusters(X, y, args.output_image)


if __name__ == '__main__':
    main()
