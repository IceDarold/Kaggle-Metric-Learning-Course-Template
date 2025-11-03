# scripts/train.py

import argparse
import os
import yaml
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# --- ВАЖНО: В реальном проекте эти классы были бы в src/ ---
# Но для простоты старта в Модуле 0 мы оставим их здесь как заглушки.

def seed_everything(seed: int):
    """Устанавливает seed для всех генераторов случайных чисел для воспроизводимости."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DummyDataset(torch.utils.data.Dataset):
    """Простой Dataset-заглушка, который генерирует случайные данные."""
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # В реальном задании здесь была бы загрузка и обработка изображений/текста
        dummy_input = torch.randn(3, 224, 224) # Пример "изображения"
        label = 0 # В реальном задании здесь была бы метка
        return dummy_input, torch.tensor(label, dtype=torch.long)

class DummyModel(torch.nn.Module):
    """Простая Модель-заглушка, которая принимает "изображения" и выдает логиты."""
    def __init__(self, config):
        super().__init__()
        # В реальной модели здесь была бы архитектура типа EfficientNet/BERT
        self.backbone = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(3, 1) # Выход для бинарной классификации

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return self.fc(x)

# --- Основная логика скрипта ---

def main():
    # 1. Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Train a model based on a YAML config file.")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # 2. Загрузка конфигурации из YAML файла
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Настройка для воспроизводимости
    seed_everything(config['general']['seed'])

    # 4. Создание директории для результатов эксперимента
    output_dir = os.path.join("outputs", config['general']['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_dir}")

    # 5. Загрузка данных и фолдов
    # ВАЖНО: предполагается, что файл с фолдами уже создан!
    folds_path = os.path.join(config['data']['path'], f"folds_{config['general']['experiment_name']}.csv")
    if not os.path.exists(folds_path):
        print(f"❌ Error: Folds file not found at {folds_path}")
        print("Please run the fold creation script first (реализуйте его в src/data/folds.py).")
        return

    df_folds = pd.read_csv(folds_path)
    oof_predictions = np.zeros(len(df_folds))

    # 6. Основной цикл обучения по фолдам
    for fold in range(config['data']['n_splits']):
        print(f"\n========== FOLD {fold} / {config['data']['n_splits'] - 1} ==========")

        # Разделение данных на train/validation
        train_df = df_folds[df_folds['fold'] != fold].reset_index(drop=True)
        valid_df = df_folds[df_folds['fold'] == fold].reset_index(drop=True)

        # Создание датасетов и даталоадеров
        train_dataset = DummyDataset(train_df, config)
        valid_dataset = DummyDataset(valid_df, config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['train_params']['batch_size'], shuffle=False)

        # Инициализация модели, лосса и оптимизатора
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DummyModel(config).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train_params']['learning_rate'])

        best_valid_loss = float('inf')

        # Цикл по эпохам
        for epoch in range(config['train_params']['epochs']):
            model.train()
            train_loss = 0
            
            # --- Training Phase ---
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train_params']['epochs']} - Train"):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            # --- Validation Phase ---
            model.eval()
            valid_loss = 0
            fold_preds = []
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config['train_params']['epochs']} - Valid"):
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    fold_preds.append(outputs.sigmoid().cpu().numpy())

            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

            # Сохранение лучшей модели
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_best_fold_{fold}.pth"))
                print(f"✨ Model saved with best validation loss: {best_valid_loss:.4f}")
                
                # Сохраняем предсказания лучшей модели
                oof_predictions[valid_df.index] = np.concatenate(fold_preds).flatten()

    # 7. Сохранение OOF предсказаний
    oof_df = pd.DataFrame({
        'id': df_folds[MATCHING_ID_COLUMN], # Убедитесь, что эта колонка есть в конфиге/данных
        'prediction': oof_predictions
    })
    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\n✅ OOF predictions saved to {oof_path}")


if __name__ == "__main__":
    main()