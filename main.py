import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def generate_dataset(n_samples=300, n_features=4, centers=4, cluster_std=1.0, random_state=42):
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)

def cluster_and_evaluate(dataset, methods):
    X, _ = dataset
    birch = Birch()
    labels = birch.fit_predict(X)

    results = {}

    for method in methods:
        if method == 'silhouette':
            score = silhouette_score(X, labels)
        elif method == 'davies_bouldin':
            score = davies_bouldin_score(X, labels)
        elif method == 'calinski_harabasz':
            score = calinski_harabasz_score(X, labels)
        else:
            # Другие методы оценки кластеризации можно добавить здесь
            print(f"Метод {method} не распознан.")
            continue

        results[method] = score

    return results

def main():
    n_features = 4
    dataset = generate_dataset(n_features=n_features)
    print(f"Сгенерированный набор данных:\n{pd.DataFrame(dataset[0], columns=[f'Feature {i}' for i in range(n_features)])}")

    methods = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

    with open('clustering_report.txt', 'w') as report_file:
        for method, score in cluster_and_evaluate(dataset, methods).items():
            report_file.write(f"{method.capitalize()} Score: {score}\n")

    print("Анализ завершен. Оценки качества кластеризации записаны в файл 'clustering_report.txt'.")

if __name__ == "__main__":
    main()
