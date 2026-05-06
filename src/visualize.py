import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

INPUT_DISTANCES = 'outputs/distances.pkl'
INPUT_PRECISIONS = 'outputs/embeddings_all_precisions.pkl'
OUTPUT_DIR = 'outputs/figures'

precisions = ['float64', 'float32', 'float16', 'int8']

def plot_histograms(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Cosine Distances by Precision Level', fontsize=14)

    for ax, name in zip(axes.flatten(), precisions):
        intra = results[name]['intra']
        inter = results[name]['inter']
        ax.hist(intra, bins=50, alpha=0.6, color='steelblue',
                label=f'Intra-speaker (mean={intra.mean():.4f})')
        ax.hist(inter, bins=50, alpha=0.6, color='tomato',
                label=f'Inter-speaker (mean={inter.mean():.4f})')
        ax.set_title(f'{name}')
        ax.set_xlabel('Cosine Distance')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'histograms.png'), dpi=150)
    plt.close()
    print("Saved histograms.png")

def plot_kde(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KDE of Cosine Distances by Precision Level', fontsize=14)

    for ax, name in zip(axes.flatten(), precisions):
        intra = results[name]['intra']
        inter = results[name]['inter']

        for data, color, label in [(intra, 'steelblue', 'Intra-speaker'),
                                    (inter, 'tomato', 'Inter-speaker')]:
            kde = gaussian_kde(data)
            x = np.linspace(0, 0.7, 300)
            ax.plot(x, kde(x), color=color, label=f'{label} (mean={data.mean():.4f})')
            ax.fill_between(x, kde(x), alpha=0.2, color=color)

        ax.set_title(f'{name}')
        ax.set_xlabel('Cosine Distance')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kde.png'), dpi=150)
    plt.close()
    print("Saved kde.png")

def plot_comparison(results, data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Impact of Precision Reduction on Distance Structure', fontsize=13)

    intra_means = [results[p]['intra'].mean() for p in precisions]
    inter_means = [results[p]['inter'].mean() for p in precisions]
    ratios = [inter_means[i]/intra_means[i] for i in range(4)]

    x = np.arange(4)
    axes[0].bar(x - 0.2, intra_means, 0.4, label='Intra-speaker', color='steelblue')
    axes[0].bar(x + 0.2, inter_means, 0.4, label='Inter-speaker', color='tomato')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(precisions)
    axes[0].set_title('Mean Cosine Distances')
    axes[0].set_ylabel('Cosine Distance')
    axes[0].legend()

    axes[1].plot(precisions, ratios, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1].set_title('Inter/Intra Distance Ratio')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim(1.5, 2.5)

    max_errors = [0,
        np.abs(results['float64']['intra'] - results['float32']['intra']).max(),
        np.abs(results['float64']['intra'] - results['float16']['intra']).max(),
        np.abs(results['float64']['intra'] - results['int8']['intra']).max()]
    axes[2].bar(precisions, max_errors, color='darkorange')
    axes[2].set_title('Max Error vs float64 (intra)')
    axes[2].set_ylabel('Absolute Error')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison.png'), dpi=150)
    plt.close()
    print("Saved comparison.png")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_DISTANCES, 'rb') as f:
        results = pickle.load(f)

    with open(INPUT_PRECISIONS, 'rb') as f:
        data = pickle.load(f)

    plot_histograms(results)
    plot_kde(results)
    plot_comparison(results, data)
    print("All figures saved!")

if __name__ == '__main__':
    main()