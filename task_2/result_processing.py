import numpy as np
import os
from matplotlib import pyplot as plt


def sort_results(results):
    sorted_results = {'base': [], 'label_prop': [], 'semi_supervised': []}
    for base, label, semi in sorted(zip(results['base'], results['label_prop'], results['semi_supervised'])):
        sorted_results['base'].append(base)
        sorted_results['label_prop'].append(label)
        sorted_results['semi_supervised'].append(semi)
    
    return sorted_results


def plot_results(results, neighbours=3):
    old_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    results = sort_results(results.copy())
    plt.plot(results['base'], label='Baseline')
    plt.plot(results['label_prop'], label='Label Propagation')
    plt.plot(results['semi_supervised'], label='Semi-Supervised')
    # Set label for y axis
    plt.ylabel('F2 score')
    plt.xlabel('Iteration Index')

    # set limits for y-axis
    plt.ylim(0.84, 1)

    plt.legend()
    plt.savefig(f'results{neighbours}.png')

    os.chdir(old_dir)


def write_results(results, neighbours=3):
    old_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f'\nF2 baseline: {np.mean(results["base"])}, {np.std(results["base"])}                         ')
    print(f'F2 label propagation: {np.mean(results["label_prop"])}, {np.std(results["label_prop"])}')
    print(f'F2 semi-supervised: {np.mean(results["semi_supervised"])}, {np.std(results["semi_supervised"])}')
    with open(f"results{neighbours}.txt", "w") as f:
        f.write(f'\nF2 baseline: {np.mean(results["base"])}, {np.std(results["base"])}\n')
        f.write(f'F2 label propagation: {np.mean(results["label_prop"])}, {np.std(results["label_prop"])}\n')
        f.write(f'F2 semi-supervised: {np.mean(results["semi_supervised"])}, {np.std(results["semi_supervised"])}\n')
        f.write(f'Baseline\n')
        for result in results['base']:
            f.write(f'{result: .4f}\n')
        f.write(f'\nLabel propagation\n')
        for result in results['label_prop']:
            f.write(f'{result: .4f}\n')
        f.write(f'\nSemi-supervised\n')
        for result in results['semi_supervised']:
            f.write(f'{result: .4f}\n')
    
    os.chdir(old_dir)


