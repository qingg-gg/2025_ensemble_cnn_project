"""
CNN 主程式，訓練並驗證 ResNet-18、ResNet-34 的單一模型與集成模型
最終匯出 Snapshot（.pth）與 結果（JSON）
"""

import argparse
import time
import json
import torch
from pathlib import Path

from src.model.resnet import ResNet18, ResNet34
from src.train.snapshot import SnapshotEnsembleTrainer
from src.train.ensemble import EnsembleDefense
from src.attack.fgsm import FGSMAttack
from src.evaluation.train_evaluate import TrainEvaluator
from src.evaluation.attack_evaluate import AttackEvaluator
from src.utils.set_device import get_device
from src.utils.data_loader import dataloader

# 統一管理、執行所有模型
class ExperimentRunner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.results = {
            'resnet18': {},
            'resnet34': {},
            'ensemble_resnet18': {},
            'ensemble_resnet34': {}
        }
        self.train_loader, self.test_loader = dataloader(
            data_dir = args.data_dir,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )
        self.train_evaluator = TrainEvaluator(device)
        self.attack_evaluator = AttackEvaluator(device)

    # 訓練單一模型、儲存 Snapshot
    def train_model(self, model_name, model_fn):
        print(f"Start training model {model_name.upper()}.")

        # 建立模型
        model = model_fn(num_classes = 10)
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_parameters}")
        print(f"Trainable parameters: {trainable_parameters}")

        # 建立訓練器
        save_dir = f"{self.args.save_dir}/{model_name}"
        trainer = SnapshotEnsembleTrainer(
            model_fn = lambda: model_fn(num_classes = 10),
            device = self.device,
            save_dir = save_dir,
            num_snapshots = self.args.num_snapshots
        )

        # 訓練並評估結果
        start_time = time.time()
        train_history = trainer.train_with_snapshots(
            train_loader = self.train_loader,
            num_epochs = self.args.epochs,
            lr = self.args.lr,
        )
        training_time = time.time() - start_time
        _, final_accuracy = self.train_evaluator.evaluate_model(trainer.model, self.test_loader, return_loss = True)

        # 儲存結果
        self.results[model_name] = {
            'model_parameters': total_parameters,
            'training_time': training_time,
            'final_accuracy': final_accuracy,
            'train_history': train_history,
            'trainer': trainer
        }
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Final accuracy: {final_accuracy:.2f}%")

        return trainer

    # 驗證單一模型的魯棒性
    def evaluate_single_model(self, model_name, trainer):
        print(f"Start evaluating model {model_name.upper()} (single).")

        # 載入模型
        checkpoint = torch.load(trainer.snapshots[-1], map_location = self.device)
        if model_name == 'resnet18':
            model = ResNet18(num_classes = 10).to(self.device)
        else:
            model = ResNet34(num_classes = 10).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 進行不同 Epsilon 的攻擊
        results = []
        for eps in self.args.epsilon_list:
            attack = FGSMAttack(model, epsilon = eps)
            clean_accuracy, adversarial_accuracy = self.attack_evaluator.evaluate_single_model(model, attack, self.test_loader)
            results.append({
                'epsilon': eps,
                'clean_accuracy': clean_accuracy,
                'adversarial_accuracy': adversarial_accuracy,
                'drop': clean_accuracy - adversarial_accuracy
            })

        # 儲存結果
        self.results[model_name]['single_model_results'] = results
        self.results[model_name]['model'] = model

        return model, results

    # 驗證集成模型的魯棒性
    def evaluate_ensemble_model(self, model_name, trainer):
        print(f"Start evaluating model {model_name.upper()} (ensemble).")

        # 載入模型
        ensemble_models = trainer.load_snapshot()
        ensemble = EnsembleDefense(ensemble_models, self.device)

        # 測試原始資料
        clean_accuracy = self.train_evaluator.evaluate_ensemble(ensemble, self.test_loader)

        # 進行不同 Epsilon 的攻擊
        results = []
        for eps in self.args.epsilon_list:
            attack = FGSMAttack(ensemble_models[0], epsilon = eps)
            ensemble_accuracy = self.attack_evaluator.evaluate_ensemble_models(ensemble, attack, self.test_loader)

            results.append({
                'epsilon': eps,
                'ensemble_accuracy': ensemble_accuracy
            })

        # 儲存結果
        ensemble_key = f"ensemble_{model_name}"
        self.results[ensemble_key] = {
            'clean_accuracy': clean_accuracy,
            'ensemble_results': results,
            'ensemble': ensemble
        }

        return ensemble, results

    def _load_trainer(self, model_name, model_fn):
        save_dir = f"{self.args.save_dir}/{model_name}"
        model = model_fn(num_classes = 10)
        trainer = SnapshotEnsembleTrainer(
            model_fn = lambda: model_fn(num_classes = 10),
            device = self.device,
            save_dir = save_dir,
            num_snapshots = self.args.num_snapshots
        )

        # 載入 Snapshots
        snapshot_dir = Path(save_dir)
        if snapshot_dir.exists():
            snapshots = sorted(snapshot_dir.glob("*.pth"))
            trainer.snapshots = [str(s) for s in snapshots]
        else:
            raise FileNotFoundError(f"Could not find directory: {snapshot_dir}")

        return trainer

    def print_comparison(self):
        print("\n" + "=" * 70)
        print("!!! Result !!!")
        print("=" * 70)

        # 模型資訊
        print("\n1. About models")
        print(f"{'Model':<20} {'Parameter':<15} {'Time':<15} {'Clean':<15}")
        for name in ['resnet18', 'resnet34']:
            if name in self.results and 'model_parameters' in self.results[name]:
                parameters = self.results[name]['model_parameters']
                time = self.results[name]['training_time'] / 60
                accuracy = self.results[name]['final_accuracy']
                print(f"{name.upper():<20} {parameters:<12,} {time:<10.2f}min {accuracy:<12.2f}")

        # 單一模型魯棒性
        print("\n2. Robustness of single model")
        header = f"{'Epsilon':<10}"
        for name in ['resnet18', 'resnet34']:
            header += f"{name.upper():<15}"
        print(header)
        for i, eps in enumerate(self.args.epsilon_list):
            row = f"{eps:<10.3f}"
            for name in ['resnet18', 'resnet34']:
                if 'single_model_results' in self.results[name]:
                    accuracy = self.results[name]['single_model_results'][i]['adversarial_accuracy']
                    row += f"{accuracy:<15.2f}"
            print(row)

        # 集成模型魯棒性
        print("\n3. Robustness of ensemble model")
        header = f"{'Epsilon':<10}"
        for name in ['resnet18', 'resnet34']:
            header += f"{'ENS-' + name.upper():<15}"
        print(header)
        for i, eps in enumerate(self.args.epsilon_list):
            row = f"{eps:<10.3f}"
            for name in ['resnet18', 'resnet34']:
                ensemble_key = f"ensemble_{name}"
                if ensemble_key in self.results:
                    accuracy = self.results[ensemble_key]['ensemble_results'][i]['ensemble_accuracy']
                    row += f"{accuracy:<15.2f}"
            print(row)

        # 集成模型改善幅度
        print("\n4, Improvement")
        header = f"{'Epsilon':<10}"
        for name in ['resnet18', 'resnet34']:
            header += f"{name.upper():<15}"
        print(header)
        for i, eps in enumerate(self.args.epsilon_list):
            row = f"{eps:<10.3f}"
            for name in ['resnet18', 'resnet34']:
                ensemble_key = f"ensemble_{name}"
                if 'single_model_results' in self.results[name] and ensemble_key in self.results:
                    single_accuracy = self.results[name]['single_model_results'][i]['adversarial_accuracy']
                    ensemble_accuracy = self.results[ensemble_key]['ensemble_results'][i]['ensemble_accuracy']
                    improvement = ensemble_accuracy - single_accuracy
                    row += f"{improvement:<15.2f}"
            print(row)

    def save_results(self):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok = True)

        output_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                output_results[key] = {}
                for k, v in value.items():
                    if k not in ['trainer', 'model', 'ensemble']:
                        if k == 'train_history' and 'learning_rate' in v:
                            v = {kk: vv for kk, vv in v.items()}
                        output_results[key][k] = v

        output_file = output_dir / f"result.json"
        with open(output_file, 'w') as f:
            json.dump(output_results, f, indent = 2)

        print(f"Results saved to {output_file}.")

    def run(self):
        print("%% ==================== Start running ==================== %%")
        print("Model: ResNet-18, ResNet-34")
        print(f"Epochs: {self.args.epochs}")
        print(f"Number of Snapshots: {self.args.num_snapshots}")
        print(f"Epsilon list: {self.args.epsilon_list}")

        # 訓練
        if self.args.train:
            trainer_18 = self.train_model('resnet18', ResNet18)
            trainer_34 = self.train_model('resnet34', ResNet34)
        else:
            trainer_18 = self._load_trainer('resnet18', ResNet18)
            trainer_34 = self._load_trainer('resnet34', ResNet34)

        # 評估
        if self.args.evaluate:
            model_18, single_results_18 = self.evaluate_single_model('resnet18', trainer_18)
            model_34, single_results_34 = self.evaluate_single_model('resnet34', trainer_34)
            ensemble_18, ensemble_results_18 = self.evaluate_ensemble_model('resnet18', trainer_18)
            ensemble_34, ensemble_results_34 = self.evaluate_ensemble_model('resnet34', trainer_34)

            self.print_comparison()

        # 儲存結果
        if self.args.save_results:
            self.save_results()

        return self.results

def main():
    # 設定參數
    parser = argparse.ArgumentParser(description = "Snapshot Ensemble Trainer for ResNet-18 and ResNet-34")

    # 資料相關
    parser.add_argument("--data_dir", type = str, default = "./data", help = "Directory to load data from.")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training.")
    parser.add_argument("--num_workers", type = int, default = 0, help = "Number of workers for training.")

    # 訓練相關
    parser.add_argument("--epochs", type = int, default = 180, help = "Number of epochs for training.")
    parser.add_argument("--lr", type = float, default = 0.1, help = "Learning rate for training.")
    parser.add_argument("--num_snapshots", type = int, default = 3, help = "Number of snapshots.")
    parser.add_argument("--save_dir", type = str, default = "./snapshots", help = "Directory to save snapshots.")

    # 攻擊相關
    parser.add_argument("--epsilon_list", type = float, nargs = '+', default = [0.01, 0.03, 0.05, 0.1], help = "Epsilon list for training.")

    # 執行相關
    parser.add_argument("--train", action = 'store_true', help = "Train the model.")
    parser.add_argument("--evaluate", action = 'store_true', help = "Evaluate the model.")
    parser.add_argument("--save_results", action = 'store_true', help = "Save the results as json file.")
    parser.add_argument("--output_dir", type = str, default = "./results", help = "Directory to save results.")

    args = parser.parse_args()

    # 執行
    device = get_device()
    runner = ExperimentRunner(args, device)
    results = runner.run()

    print("\n%% ==================== Finished ==================== %%")

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(["--train", "--evaluate", "--save_results"])

    main()