"""
    python make_moons.py --samples 500 --noise 0.1 --output moons_data.csv --output_image moons_plot.jpg
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def generate_moons(n_samples, shuffle=True, noise=None, random_state=None):
    # Генерация данных с использованием make_moons
    X, y = make_moons(
        n_samples=n_samples,
        shuffle=shuffle,
        noise=noise,
        random_state=random_state
    )
    return X, y


def save_to_csv(X, output_file):
    # Сохранение данных в CSV файл
    df = pd.DataFrame(data=X, columns=['feature_1', 'feature_2'])
    df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')


def plot_moons(X, y, output_image):
    # Построение графика полумесяцев
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    plt.title('Moons Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(output_image, format='jpeg')
    print(f'Image saved to {output_image}')
    plt.show()


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Moons generation script')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the samples')
    parser.add_argument('--noise', type=float, default=None, help='Standard deviation of Gaussian noise added to the data')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility')
    parser.add_argument('--output', type=str, default='moons_data.csv', help='Output CSV file name')
    parser.add_argument('--output_image', type=str, default='moons_plot.jpg', help='Output image file name')

    # Получение аргументов
    args = parser.parse_args()

    # Генерация полумесяцев
    X, y = generate_moons(
        args.samples,
        shuffle=args.shuffle,
        noise=args.noise,
        random_state=args.random_state
    )

    # Сохранение в CSV
    save_to_csv(X, args.output)

    # Построение графика и сохранение в JPEG
    plot_moons(X, y, args.output_image)


if __name__ == '__main__':
    main()
