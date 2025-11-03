# scripts/validate.py

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

def find_best_threshold(y_true, y_pred_probs, metric_func):
    """
    Находит лучший порог для бинарной классификации по заданной метрике.
    """
    best_threshold = 0.0
    best_score = 0.0
    
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = []
    
    for threshold in tqdm(thresholds, desc="Finding best threshold"):
        y_pred_binary = (y_pred_probs > threshold).astype(int)
        score = metric_func(y_true, y_pred_binary)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_score, best_threshold, scores, thresholds


def main():
    # 1. Парсинг аргументов и загрузка конфига
    parser = argparse.ArgumentParser(description="Validate OOF predictions.")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        print("❌ Error: Configuration file is empty or invalid.")
        return

    # 2. Определение путей
    output_dir = os.path.join("outputs", config['general']['experiment_name'])
    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    ground_truth_path = os.path.join(config['data']['path'], config['data']['train_file'])

    if not os.path.exists(oof_path):
        print(f"❌ Error: OOF predictions file not found at {oof_path}")
        return

    # 3. Загрузка данных
    print("Loading ground truth and OOF predictions...")
    df_oof = pd.read_csv(oof_path)
    df_gt = pd.read_csv(ground_truth_path)

    # --- ВАЖНО: Логика мержа может зависеть от задачи ---
    # Для Shopee/Quora нужно будет смержить по ID и затем сгенерировать пары
    # Для простоты здесь мы предположим, что у нас уже есть y_true и y_pred
    # TODO: Студентам нужно будет реализовать правильную логику мержа/создания пар
    # Здесь используем заглушку
    if 'target' not in df_gt.columns:
         print("Warning: 'target' column not found in ground truth. Creating a dummy target.")
         df_gt['target'] = (np.random.rand(len(df_gt)) > 0.5).astype(int)

    # Убедимся, что порядок совпадает
    # df_merged = pd.merge(df_gt, df_oof, on='id')
    y_true = df_gt['target'].values
    y_pred_probs = df_oof['prediction'].values

    # 4. Поиск лучшего порога и расчет CV
    print("\nCalculating CV score and finding best threshold...")
    best_f1, best_thresh, f1_scores, thresholds = find_best_threshold(y_true, y_pred_probs, f1_score)

    print("\n--- VALIDATION RESULTS ---")
    print(f"📈 Best CV F1-Score: {best_f1:.4f}")
    print(f"🔪 at Threshold: {best_thresh:.2f}")
    print("--------------------------")

    # 5. Построение и сохранение графика
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(best_thresh, color='r', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
    plt.title(f"F1 Score vs. Threshold | Best CV F1 = {best_f1:.4f}", fontsize=16)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "f1_vs_threshold.png")
    plt.savefig(plot_path)
    print(f"\n✅ Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()