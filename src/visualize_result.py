"""
一鍵生成所有視覺化圖表
整合統計圖表和對抗樣本案例展示
"""

import argparse
from pathlib import Path
from src.utils.statical_chart import ResultVisualizer
from src.utils.adversarial_example import AdversarialExampleVisualizer
from src.utils.set_device import get_device
from src.utils.data_loader import dataloader

def generate_adversarial_examples(models, snapshot_dir, data_dir, epsilon, num_examples, output_dir, device):
    # 載入測試資料
    _, testloader = dataloader(
        data_dir=data_dir,
        batch_size=100,
        num_workers=0
    )

    success_count = 0
    total_models = len(models)

    print("Start generate adversarial examples:")
    for idx, model_name in enumerate(models, 1):
        # 建立視覺化器
        visualizer = AdversarialExampleVisualizer(
            model_name = model_name,
            snapshot_dir = snapshot_dir,
            device = device
        )

        # 尋找案例
        examples = visualizer.find_successful_defense_examples(
            data_loader = testloader,
            epsilon = epsilon,
            num_examples = num_examples
        )

        if not examples:
            print(f"No defense examples found.")
            continue

        # 儲存影像
        for example_idx, example in enumerate(examples, 1):
            save_path = output_dir / f"example_{model_name}_{example_idx}.png"
            visualizer.visualize_example(example, save_path = save_path)
            success_count += 1

    print("Finished.")

# 統計圖表
def generate_statistical_charts(result_file, output_dir):
    visualizer = ResultVisualizer(
        result_file=result_file,
        output_dir=output_dir
    )

    print("Start generate static charts:")
    print(" [1/5] Plot robustness comparison.")
    visualizer.plot_robustness_comparison()
    print(" [2/5] Plot ensemble improvement.")
    visualizer.plot_ensemble_improvement()
    print(" [3/5] 準確率熱力圖...")
    visualizer.plot_accuracy_heatmap()
    print(" [4/5] Plot accuracy heatmap.")
    visualizer.plot_training_curves()

    print("Finished.")

def main():
    parser = argparse.ArgumentParser(description = 'Generate result visualization.')

    # 統計圖表相關
    parser.add_argument('--result_file', type = str, default = './results/result.json', help = 'Path to the result file.')

    # 對抗樣本相關
    parser.add_argument('--models', type = str, nargs = '+', default = ['resnet18', 'resnet34'], help = 'List of models.')
    parser.add_argument('--snapshot_dir', type = str, default = './snapshots', help = 'Path to the snapshot directory.')
    parser.add_argument('--data_dir', type = str, default = './data', help = 'Path to the data directory.')
    parser.add_argument('--epsilon', type = float, default = 0.03, help = 'Epsilon')
    parser.add_argument('--num_examples', type = int, default = 5, help='Number of examples to find.')

    # 輸出相關
    parser.add_argument('--output_dir', type = str, default = './results', help = 'Path to the output directory.')

    args = parser.parse_args()

    # 開始生成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    print("Start running.")

    # 成功對抗案例
    device = get_device()
    generate_adversarial_examples(
        models = args.models,
        snapshot_dir = args.snapshot_dir,
        data_dir = args.data_dir,
        epsilon = args.epsilon,
        num_examples = args.num_examples,
        output_dir = output_dir,
        device = device
    )

    # 統計圖表
    generate_statistical_charts(args.result_file, output_dir)

if __name__ == "__main__":
    main()