import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances, completeness_score, homogeneity_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generate_dataset(n_samples=300, n_features=5, centers=4, cluster_std=1.0, random_state=42):
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)

def cluster_and_evaluate(dataset):
    X, _ = dataset
    birch = Birch()
    labels = birch.fit_predict(X)

    methods = {
        'silhouette': silhouette_score,
        'davies_bouldin': davies_bouldin_score,
        'calinski_harabasz': calinski_harabasz_score,
        'pairwise_distances': lambda x: pairwise_distances(x).mean(),
        'completeness_score': completeness_score,
        'homogeneity_score': homogeneity_score,
        'v_measure_score': v_measure_score,
        'adjusted_rand_score': adjusted_rand_score,
        'adjusted_mutual_info_score': adjusted_mutual_info_score,
        'fowlkes_mallows_score': fowlkes_mallows_score,
    }

    results = {}

    for method, metric in methods.items():
        try:
            if method == 'pairwise_distances':
                score = metric(X)
            elif method in {'completeness_score', 'homogeneity_score', 'v_measure_score', 'adjusted_rand_score', 'adjusted_mutual_info_score', 'fowlkes_mallows_score'}:
                score = metric(_, labels)
            else:
                score = metric(X, labels)
            results[method] = score
        except Exception as e:
            print(f"Ошибка при вычислении {method}: {e}")

    return results

def plot_scores(scores):
    methods = list(scores.keys())
    values = list(scores.values())
    print(values)
    plt.figure(figsize=(10, 6))
    plt.bar(methods, values, color='skyblue')
    plt.xlabel('Методы оценки')
    plt.ylabel('Значения оценок')
    plt.title('Оценки качества кластеризации')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('clustering_scores.png')
    plt.show()

def main():
    n_features = 5
    dataset = generate_dataset(n_features=n_features)
    print(f"Сгенерированный набор данных:\n{pd.DataFrame(dataset[0], columns=[f'Feature {i}' for i in range(n_features)])}")

    scores = cluster_and_evaluate(dataset)

    with open('clustering_report.txt', 'w') as report_file:
        for method, score in scores.items():
            report_file.write(f"{method.capitalize()} Score: {score}\n")

    print("Анализ завершен. Оценки качества кластеризации записаны в файл 'clustering_report.txt'.")

    plot_scores(scores)
    print("График оценок качества кластеризации сохранен в файл 'clustering_scores.png'.")

if __name__ == "__main__":
    main()
