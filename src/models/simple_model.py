# src/models/simple_model.py

import torch
import torch.nn as nn
import timm # PyTorch Image Models - стандарт де-факто для CV на Kaggle

class SimpleVisionModel(nn.Module):
    """
    Простая модель для классификации изображений.
    
    Состоит из:
    1. Предобученного "бэкбона" (backbone) из библиотеки `timm` для извлечения признаков.
    2. "Головы" (head), которая преобразует признаки в предсказание.
    
    В Модуле 0 мы используем ее как пример правильной структуры.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Конфигурационный файл, содержащий параметры модели.
        """
        super().__init__()
        self.config = config
        
        # 1. Создаем бэкбон (извлекатель признаков)
        # timm.create_model - это мощная функция, которая может создать сотни
        # различных архитектур (EfficientNet, ResNet, ViT и т.д.).
        self.backbone = timm.create_model(
            model_name=config['model']['name'],
            pretrained=config['model']['pretrained'],
            in_chans=3,
            num_classes=0 # num_classes=0 означает, что нам нужен "голый" бэкбон без головы
        )
        
        # 2. Определяем "голову" модели
        # Узнаем, сколько признаков на выходе у нашего бэкбона
        backbone_out_features = self.backbone.num_features
        
        # Наша голова будет состоять из простого линейного слоя, который
        # преобразует признаки в одно число (логит) для бинарной классификации.
        self.head = nn.Linear(backbone_out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Определяет прямой проход данных через модель.
        
        Args:
            x (torch.Tensor): Входной тензор (батч изображений).
        
        Returns:
            torch.Tensor: Выходной тензор (логиты).
        """
        # 1. Пропускаем данные через бэкбон для получения карты признаков
        # Метод .forward_features() удобен тем, что он возвращает выход
        # до последнего классификационного слоя.
        features = self.backbone.forward_features(x)
        
        # 2. Применяем Global Average Pooling, чтобы "схлопнуть" пространственные
        # измерения (высоту и ширину) в один вектор признаков.
        # Форма меняется с (batch_size, channels, height, width) на (batch_size, channels)
        pooled_features = torch.mean(features, dim=[2, 3])
        
        # 3. Пропускаем вектор признаков через голову для получения предсказания
        logits = self.head(pooled_features)
        
        return logits

# --- ПРИМЕЧАНИЕ ---
# Чтобы использовать эти классы в train.py, вам нужно будет:
# 1. Добавить импорты:
#    from src.data.datasets import KaggleCompetitionDataset
#    from src.models.simple_model import SimpleVisionModel
#
# 2. Заменить создание Dummy-классов на реальные:
#    train_dataset = KaggleCompetitionDataset(train_df, config)
#    model = SimpleVisionModel(config).to(device)