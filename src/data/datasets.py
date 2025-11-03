    # src/data/datasets.py

import torch
import pandas as pd
# В будущем здесь будут импорты для работы с изображениями (PIL, cv2) 
# и текстом (transformers.tokenizer)

class KaggleCompetitionDataset(torch.utils.data.Dataset):
    """
    Класс Dataset для соревнований Kaggle.
    
    Отвечает за:
    1. Хранение ссылок на данные (DataFrame).
    2. Загрузку и предобработку ОДНОГО объекта по его индексу (метод __getitem__).
    
    В Модуле 0 мы используем его как заглушку, которая генерирует случайные данные.
    В следующих модулях вы будете наполнять `__getitem__` реальной логикой.
    """
    def __init__(self, df: pd.DataFrame, config: dict, is_train: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame с данными (например, пути к картинкам и метки).
            config (dict): Конфигурационный файл.
            is_train (bool): Флаг, указывающий, является ли это обучающей выборкой.
                             Полезен для применения разных аугментаций.
        """
        self.df = df
        self.config = config
        self.is_train = is_train
        
        # В реальном задании вы бы извлекли пути к файлам и метки
        # self.image_paths = df['image_path'].values
        # self.labels = df['target'].values
        
    def __len__(self) -> int:
        """Возвращает общее количество объектов в датасете."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Загружает, обрабатывает и возвращает один элемент данных по его индексу.
        """
        
        # ===================================================================
        # !!! ЛОГИКА-ЗАГЛУШКА ДЛЯ МОДУЛЯ 0 !!!
        # Этот код просто генерирует случайные "картинки" и "метки".
        # Он позволяет нашему пайплайну работать, даже если у нас еще нет
        # реальной логики загрузки данных.
        # ===================================================================
        
        # Генерируем случайный тензор, имитирующий изображение (3 канала, 224x224 пикселя)
        dummy_image = torch.randn(3, 224, 224)
        
        # Генерируем случайную метку (0 или 1)
        dummy_label = torch.tensor(torch.randint(0, 2, (1,)).item(), dtype=torch.long)
        
        return dummy_image, dummy_label
        
        # ===================================================================
        # !!! ПРИМЕР РЕАЛЬНОЙ ЛОГИКИ В БУДУЩЕМ !!!
        # В последующих модулях вы замените заглушку на что-то вроде этого:
        #
        # # 1. Получаем путь к изображению и метку
        # image_path = self.image_paths[idx]
        # label = self.labels[idx]
        #
        # # 2. Загружаем изображение
        # image = Image.open(image_path).convert("RGB")
        #
        # # 3. Применяем аугментации
        # if self.is_train:
        #     image = self.train_transforms(image=np.array(image))['image']
        # else:
        #     image = self.valid_transforms(image=np.array(image))['image']
        #
        # # 4. Возвращаем тензоры
        # return image, torch.tensor(label, dtype=torch.long)
        # ===================================================================