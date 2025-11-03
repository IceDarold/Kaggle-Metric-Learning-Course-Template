# src/utils/general.py

import os
import random
import logging
import numpy as np
import torch

def seed_everything(seed: int):
    """
    Устанавливает seed для всех основных генераторов случайных чисел, чтобы обеспечить
    воспроизводимость экспериментов.

    Args:
        seed (int): Целое число, которое будет использоваться как seed.
    """
    # Устанавливаем seed для встроенного в Python модуля random
    random.seed(seed)
    
    # Устанавливаем переменную окружения PYTHONHASHSEED.
    # Это важно для воспроизводимости операций, использующих хэширование (например, в словарях).
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Устанавливаем seed для NumPy
    np.random.seed(seed)
    
    # Устанавливаем seed для PyTorch на CPU
    torch.manual_seed(seed)
    
    # Устанавливаем seed для PyTorch на GPU (если доступен)
    torch.cuda.manual_seed(seed)
    
    # Дополнительные настройки для полной детерминированности на GPU.
    # torch.backends.cudnn.deterministic = True гарантирует, что cuDNN будет использовать
    # детерминированные алгоритмы.
    # torch.backends.cudnn.benchmark = False отключает встроенный бенчмарк cuDNN,
    # который может выбирать разные, недетерминированные алгоритмы для ускорения.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} for all random number generators.")


def get_logger(name: str, file_path: str) -> logging.Logger:
    """
    Создает и настраивает логгер для записи информации в файл и вывода в консоль.

    Args:
        name (str): Имя логгера (обычно __name__).
        file_path (str): Путь к файлу, в который будут записываться логи.

    Returns:
        logging.Logger: Настроенный объект логгера.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Предотвращаем дублирование сообщений, если логгер уже был настроен
    if logger.hasHandlers():
        return logger

    # Форматтер для сообщений
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler для записи в файл
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler для вывода в консоль
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

# Пример использования:
# if __name__ == '__main__':
#     # 1. Устанавливаем seed
#     seed_everything(42)
#
#     # 2. Создаем логгер
#     # В реальном скрипте путь к лог-файлу будет браться из конфига
#     logger = get_logger(__name__, 'logs/my_experiment.log')
#     
#     # 3. Используем логгер
#     logger.info("This is an informational message.")
#     a = np.random.rand(5)
#     logger.info(f"Generated random numpy array: {a}")
#     logger.warning("This is a warning message.")