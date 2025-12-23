"""
統計圖表
輸出五張圖：
    1. 各模型在不同 Epsilon 下的準確率折線圖
    2. 各模型在 Epsilon = 0.1 時的準確率柱狀圖
    3. 集成模型相對於單一模型的改善幅度柱狀圖
    4. 各模型在不同 Epsilon 下的準確率熱力圖
    5. 訓練過程的準確率曲線圖（含快照點標記）
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 設定顏色
sns.set_style("whitegrid")
colors = {
    'resnet18': '#c3a77f',
    'resnet34': '#e4a273',
    'ensemble_resnet18': '#b3beaf',
    'ensemble_resnet34': '#96b9b9',
}

class ResultVisualizer:
    def __init__(self, result_file = None, output_dir = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok = True)

        with open(result_file, 'r') as f:
            self.result = json.load(f)

    # 各模型在不同 Epsilon 下的準確率折線圖、各模型在 Epsilon = 0.1 時的準確率柱狀圖
    def plot_robustness_comparison(self):
        # Line chart
        fig, ax = plt.subplots(figsize = (12, 6))

        epsilons = [r['epsilon'] for r in self.result['resnet18']['single_model_results']]
        data = {
            'ResNet-18': [r['adversarial_accuracy'] for r in self.result['resnet18']['single_model_results']],
            'ResNet-34': [r['adversarial_accuracy'] for r in self.result['resnet34']['single_model_results']],
            'Ensemble ResNet-18': [r['ensemble_accuracy'] for r in self.result['ensemble_resnet18']['ensemble_results']],
            'Ensemble ResNet-34': [r['ensemble_accuracy'] for r in self.result['ensemble_resnet34']['ensemble_results']]
        }

        for name, accuracy in data.items():
            color = colors.get(name.lower().replace(' ', '_').replace('-', ''), '#333')
            if 'Ensemble' in name:
                marker = 's'
                linestyle = '-'
            else:
                marker = 'o'
                linestyle = '--'

            ax.plot(epsilons, accuracy, marker = marker, linestyle = linestyle, linewidth = 2,
                    markersize = 8, label = name, color = color)

        ax.set_xlabel('Epsilon', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Accuracy (%)', fontsize = 12, fontweight = 'bold')
        ax.set_title('Adversarial Robustness Comparison', fontsize = 14, fontweight = 'bold')
        ax.legend(fontsize = 10, loc = 'upper right')
        ax.grid(True, alpha = 0.3)
        ax.set_ylim([0, 65])

        plt.tight_layout()
        output_file = self.output_dir / "robustness_comparison_lc.png"
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

        # Bar chart
        fig, ax = plt.subplots(figsize = (12, 6))

        epsilons = [r['epsilon'] for r in self.result['resnet18']['single_model_results']]
        data = {
            'ResNet-18': [r['adversarial_accuracy'] for r in self.result['resnet18']['single_model_results']],
            'ResNet-34': [r['adversarial_accuracy'] for r in self.result['resnet34']['single_model_results']],
            'Ensemble ResNet-18': [r['ensemble_accuracy'] for r in self.result['ensemble_resnet18']['ensemble_results']],
            'Ensemble ResNet-34': [r['ensemble_accuracy'] for r in self.result['ensemble_resnet34']['ensemble_results']]
        }

        max_eps_idx = -1
        max_eps_accuracy = [accuracy[max_eps_idx] for accuracy in data.values()]

        x = np.arange(len(data))
        bars = ax.bar(x, max_eps_accuracy, color = [colors.get(name.lower().replace(' ', '_').replace('-', ''), '#333') for name in data.keys()], edgecolor = 'black', linewidth = 0)

        ax.set_xlabel('Model', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Accuracy (%)', fontsize = 12, fontweight = 'bold')
        ax.set_title(f"Performance at epsilon = {epsilons[max_eps_idx]}", fontsize = 14, fontweight = 'bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace(' ', '\n') for name in data.keys()], fontsize = 9)
        ax.set_ylim([0, 10])
        ax.grid(True, alpha = 0.3, axis = 'y')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                    ha = 'center', va = 'bottom', fontsize = 10)

        plt.tight_layout()
        output_file = self.output_dir / "robustness_comparison_bc.png"
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    # 集成模型相對於單一模型的改善幅度柱狀圖
    def plot_ensemble_improvement(self):
        fig, ax = plt.subplots(figsize = (12, 6))

        epsilons = [r['epsilon'] for r in self.result['resnet18']['single_model_results']]

        # 計算改善幅度
        improvements = {
            'ResNet-18': [],
            'ResNet-34': []
        }
        for i in range(len(epsilons)):
            single_18 = self.result['resnet18']['single_model_results'][i]['adversarial_accuracy']
            ensemble_18 = self.result['ensemble_resnet18']['ensemble_results'][i]['ensemble_accuracy']
            improvements['ResNet-18'].append(ensemble_18 - single_18)

            single_34 = self.result['resnet34']['single_model_results'][i]['adversarial_accuracy']
            ensemble_34 = self.result['ensemble_resnet34']['ensemble_results'][i]['ensemble_accuracy']
            improvements['ResNet-34'].append(ensemble_34 - single_34)

        # 繪製圖表
        x = np.arange(len(epsilons))
        width = 0.35

        bars1 = ax.bar(x - width/2, improvements['ResNet-18'], width, label = 'ResNet-18', color = colors['ensemble_resnet18'], edgecolor = 'black', linewidth = 0)
        bars2 = ax.bar(x + width/2, improvements['ResNet-34'], width, label = 'ResNet-34', color = colors['ensemble_resnet34'], edgecolor = 'black', linewidth = 0)

        ax.set_xlabel('Epsilon', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Improvement (%)', fontsize = 12, fontweight = 'bold')
        ax.set_title('Ensemble Improvement over Single Model', fontsize = 14, fontweight = 'bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{eps:.3f}' for eps in epsilons])
        ax.legend(fontsize = 11)
        ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 1)
        ax.grid(True, alpha = 0.3, axis = 'y')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:+.1f}%', ha = 'center', va = 'bottom' if height > 0 else 'top', fontsize = 9)

        plt.tight_layout()
        output_file = self.output_dir / "ensemble_improvement.png"
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    # 各模型在不同 Epsilon 下的準確率熱力圖
    def plot_accuracy_heatmap(self):
        fig, ax = plt.subplots(figsize = (12, 6))

        epsilons = [r['epsilon'] for r in self.result['resnet18']['single_model_results']]
        model_names = ['ResNet-18', 'ResNet-34', 'Ensemble ResNet-18', 'Ensemble ResNet-34']
        data_matrix = []

        # 載入資料
        for key in ['resnet18', 'resnet34', 'ensemble_resnet18', 'ensemble_resnet34']:
            if 'ensemble' in key:
                accuracy = [r['ensemble_accuracy'] for r in self.result[key]['ensemble_results']]
            else:
                accuracy = [r['adversarial_accuracy'] for r in self.result[key]['single_model_results']]
            data_matrix.append(accuracy)
        data_matrix = np.array(data_matrix)

        # 繪製圖表
        im = ax.imshow(data_matrix, cmap = 'Blues', aspect='auto', vmin = 0, vmax = 100)

        ax.set_xticks(np.arange(len(epsilons)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels([f'{eps:.3f}' for eps in epsilons])
        ax.set_yticklabels(model_names)

        for i in range(len(model_names)):
            for j in range(len(epsilons)):
                text = ax.text(j ,i, f'{data_matrix[i, j]:.1f}', ha = 'center', va = 'center', color = 'black')
        ax.set_xlabel('Epsilon', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Model', fontsize = 12, fontweight = 'bold')
        ax.set_title('Accuracy Heatmap under Different Epsilons', fontsize = 14, fontweight = 'bold')

        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label('Accuracy (%)', rotation = 270, labelpad = 20, fontweight = 'bold')

        plt.tight_layout()
        output_file = self.output_dir / "accuracy_heatmap.png"
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    # 訓練過程的準確率曲線圖（含快照點標記）
    def plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

        for model_name, ax in [('resnet18', ax1), ('resnet34', ax2)]:
            if 'train_history' not in self.result[model_name]:
                print(f'No train history found for {model_name}')
                continue

            history = self.result[model_name]['train_history']
            epochs = range(1, len(history['train_accuracy']) + 1)

            ax.plot(epochs, history['train_accuracy'], label = 'Training Accuracy', color = colors[model_name], linewidth = 2)
            if 'snapshot_epochs' in history:
                for snap_epoch in history['snapshot_epochs']:
                    ax.axvline(x = snap_epoch, color = 'black', linewidth = 1, linestyle = '--', alpha = 0.5)
                    ax.text(snap_epoch - 1, 0.01, f'Snapshot at epoch {snap_epoch}  ', ha = 'right', va = 'bottom', transform = ax.get_xaxis_transform(), fontsize = 10, rotation = 90, alpha = 0.5)

            ax.set_xlabel('Epoch', fontsize = 12, fontweight = 'bold')
            ax.set_ylabel('Training Accuracy (%)', fontsize = 12, fontweight = 'bold')
            ax.set_title(f'{model_name.upper()} Training Curve', fontsize = 14, fontweight = 'bold')
            ax.legend(fontsize = 10)
            ax.grid(True, alpha = 0.3)
            ax.set_ylim(0, 105)

        plt.tight_layout()
        output_file = self.output_dir / "training_curves.png"
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    def generate_plots(self):
        self.plot_robustness_comparison()
        self.plot_ensemble_improvement()
        self.plot_accuracy_heatmap()
        self.plot_training_curves()

def main():
    import argparse

    # 設定參數
    parser = argparse.ArgumentParser(description = 'Generating statical charts.')
    parser.add_argument('--result_file', type = str, default = './result.json', help = 'Path to result file.')
    parser.add_argument('--output_dir', type = str, default = './results', help = 'Path to output directory.')

    args = parser.parse_args()

    # 產生圖表
    print('Start plotting.')
    visualizer = ResultVisualizer(args.result_file, args.output_dir)
    visualizer.generate_plots()
    print('Finished.')

if __name__ == '__main__':
    main()