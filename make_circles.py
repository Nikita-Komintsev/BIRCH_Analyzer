"""
    python make_circles.py --samples 500 --noise 0.1 --output circles_data.csv --output_image circles_plot.jpg
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def generate_circles(n_samples, shuffle=True, noise=None, random_state=None, factor=0.8):
    # Генерация данных с использованием make_circles
    X, y = make_circles(
        n_samples=n_samples,
        shuffle=shuffle,
        noise=noise,
        random_state=random_state,
        factor=factor
    )
    return X, y


def save_to_csv(X, output_file):
    # Сохранение данных в CSV файл
    df = pd.DataFrame(data=X, columns=['feature_1', 'feature_2'])
    df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')


def plot_circles(X, y, output_image):
    # Построение графика окружностей
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    plt.title('Circles Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(output_image, format='jpeg')
    print(f'Image saved to {output_image}')
    plt.show()


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Circles generation script')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the samples')
    parser.add_argument('--noise', type=float, default=None, help='Standard deviation of Gaussian noise added to the data')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility')
    parser.add_argument('--factor', type=float, default=0.8, help='Scale factor between inner and outer circle for make_circles')
    parser.add_argument('--output', type=str, default='circles_data.csv', help='Output CSV file name')
    parser.add_argument('--output_image', type=str, default='circles_plot.jpg', help='Output image file name')

    # Получение аргументов
    args = parser.parse_args()

    # Генерация окружностей
    X, y = generate_circles(
        args.samples,
        shuffle=args.shuffle,
        noise=args.noise,
        random_state=args.random_state,
        factor=args.factor
    )

    # Сохранение в CSV
    save_to_csv(X, args.output)

    # Построение графика
    plot_circles(X, y, args.output_image)


if __name__ == '__main__':
    main()
