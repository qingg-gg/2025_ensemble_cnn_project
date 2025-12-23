"""
快速實驗測試
用較大的 batch_size、較少的 epochs 驗證整個流程
"""

import sys
import argparse
from cnn import ExperimentRunner
from src.utils.set_device import get_device

sys.path.insert(0, '.')

def quick_test():
    args = argparse.Namespace(
        # 資料
        data_dir = './data',
        batch_size = 256,
        num_workers = 0,

        # 訓練
        epochs = 5,
        lr = 0.1,
        num_snapshots = 3,
        save_dir = './snapshots',

        # 攻擊
        epsilon_list = [0.01, 0.03, 0.05],

        # 執行
        train = True,
        evaluate = True,
        save_results = True,
        output_dir = './results'
    )

    # 執行實驗
    device = get_device()
    print("Starting quick test.")

    runner = ExperimentRunner(args, device)
    results = runner.run()

    print("\nFinished quick test.")

if __name__ == "__main__":
    quick_test()