# src/data/folds.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

def create_folds(config):
    """
    Создает и сохраняет фолды для кросс-валидации на основе конфига.
    """
    print("Creating folds...")
    
    # Загрузка основного файла с данными
    df = pd.read_csv(f"{config['data']['path']}/{config['data']['train_file']}")
    
    # Инициализация колонки для фолдов
    df['fold'] = -1

    # ===================================================================
    # !!! ВАШЕ ЗАДАНИЕ: РЕАЛИЗУЙТЕ ЛОГИКУ НИЖЕ !!!
    #
    # 1. Прочитайте из `config` стратегию фолдирования (`fold_strategy`).
    # 2. В зависимости от стратегии ('GroupKFold', 'StratifiedKFold', etc.)
    #    создайте соответствующий объект из scikit-learn.
    #    Не забудьте указать n_splits из конфига.
    # 3. Примените .split() для генерации фолдов. Если это GroupKFold,
    #    не забудьте передать группы (df[config['data']['group_col']]).
    # 4. Заполните колонку 'fold' в DataFrame `df`.
    # 5. Сохраните DataFrame с фолдами в .csv файл.
    #    Название файла должно быть уникальным для эксперимента.
    #
    # Пример для StratifiedKFold:
    # skf = StratifiedKFold(n_splits=config['data']['n_splits'], shuffle=True, random_state=config['general']['seed'])
    # for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_group'])):
    #     df.loc[val_idx, 'fold'] = fold
    # ===================================================================
    
    # TODO: Ваш код здесь
    
    # Пример сохранения
    # output_path = f"folds_{config['general']['experiment_name']}.csv"
    # df.to_csv(output_path, index=False)
    # print(f"Folds saved to {output_path}")

    raise NotImplementedError("Пожалуйста, реализуйте логику создания фолдов в src/data/folds.py")