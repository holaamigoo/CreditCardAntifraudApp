"""
Модуль для обучения и сравнения моделей машинного обучения
для обнаружения мошеннических транзакций.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, \
    confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import joblib

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path, sample_size=None):
    """
    Загружает и подготавливает данные для обучения.

    Args:
        file_path (str): Путь к CSV-файлу с данными.
        sample_size (int, optional): Количество записей для загрузки.
                                     Если None, загружаются все данные.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, cat_cols, num_cols, scale_pos_weight)
            - X_train, X_test: Матрицы признаков для обучения и теста.
            - y_train, y_test: Целевые переменные для обучения и теста.
            - cat_cols (list): Список категориальных колонок.
            - num_cols (list): Список числовых колонок.
            - scale_pos_weight (float): Вес для балансировки классов.
    """
    print("Загрузка данных...")
    df = pd.read_csv(file_path)
    if sample_size:
        df = df.head(sample_size)

    print(f"Распределение классов:\n{df['is_fraud'].value_counts()}")
    print(f"Доля мошеннических транзакций: {df['is_fraud'].mean() * 100:.2f}%")

    # Столбцы, которые не являются признаками
    cols_to_drop = ["transaction_id", "customer_id", "transaction_time", "is_fraud"]

    y = df['is_fraud']
    X = df.drop(columns=cols_to_drop)

    # Определяем категориальные и числовые колонки
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Категориальные признаки: {len(cat_cols)}")
    print(f"Числовые признаки: {len(num_cols)}")

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    # Расчёт веса для балансировки классов
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    return X_train, X_test, y_train, y_test, cat_cols, num_cols, scale_pos_weight


def create_preprocessor(cat_cols, num_cols):
    """
    Создаёт препроцессор для обработки категориальных и числовых признаков.

    Args:
        cat_cols (list): Список категориальных колонок.
        num_cols (list): Список числовых колонок.

    Returns:
        ColumnTransformer: Препроцессор для sklearn Pipeline.
    """
    return ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])


def get_models_config(scale_pos_weight):
    """
    Возвращает конфигурацию моделей для обучения и сравнения.

    Args:
        scale_pos_weight (float): Вес для балансировки классов в XGBoost.

    Returns:
        dict: Словарь с конфигурациями моделей.
    """
    return {
        'XGBoost': {
            'model': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False
            ),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [4, 6, 8],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__scale_pos_weight': [5, 10, 20],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [6, 10, 15],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', 'balanced_subsample']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        }
    }


def train_and_evaluate_model(name, config, preprocessor, X_train, y_train, X_test, y_test):
    """
    Обучает и оценивает модель с использованием GridSearchCV.

    Args:
        name (str): Название модели.
        config (dict): Конфигурация модели (модель и параметры для GridSearch).
        preprocessor (ColumnTransformer): Препроцессор для обработки признаков.
        X_train, y_train: Обучающая выборка.
        X_test, y_test: Тестовая выборка.

    Returns:
        tuple: (best_estimator, metrics)
            - best_estimator: Лучшая модель после GridSearch.
            - metrics (dict): Словарь с метриками (F1, Precision, Recall, ROC-AUC).
    """
    print(f"\n{'=' * 50}")
    print(f"Обучение модели: {name}")
    print('=' * 50)

    # Создаём пайплайн с SMOTE
    if name == 'KNN':
        # Для KNN используем меньшее количество oversampling
        sampler = SMOTE(random_state=42, sampling_strategy=0.3)
    else:
        sampler = SMOTE(random_state=42, sampling_strategy=0.5)

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', sampler),
        ('classifier', config['model'])
    ])

    # GridSearchCV с кросс-валидацией
    grid_search = GridSearchCV(
        pipeline,
        config['params'],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    # Обучение
    print(f"Начало обучения {name}...")
    grid_search.fit(X_train, y_train)

    # Лучшие параметры
    print(f"\nЛучшие параметры для {name}:")
    print(grid_search.best_params_)

    # Предсказания
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]

    # Метрики
    metrics = {
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    print(f"\nРезультаты для {name}:")
    print(f"F1-Score: {metrics['F1']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return grid_search.best_estimator_, metrics


def analyze_feature_importance(model, num_cols, cat_cols, top_n=10):
    """
    Анализирует и выводит важность признаков модели.

    Args:
        model: Обученная модель (должна иметь feature_importances_).
        num_cols (list): Список числовых колонок.
        cat_cols (list): Список категориальных колонок.
        top_n (int, optional): Количество наиболее важных признаков для вывода.

    Returns:
        pd.DataFrame: DataFrame с важностью признаков, отсортированный по убыванию.
    """
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        feature_names = num_cols + cat_cols
        importances = model.named_steps['classifier'].feature_importances_

        indices = np.argsort(importances)[::-1]

        print(f"\nТоп-{top_n} наиболее важных признаков:")
        for i in range(min(top_n, len(feature_names))):
            print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

        # Создаём DataFrame для анализа
        imp_df = pd.DataFrame({
            'Признак': feature_names,
            'Важность': importances
        }).sort_values('Важность', ascending=False)

        return imp_df

    print("Модель не поддерживает feature_importances_")
    return None


def save_model_and_results(best_model, results_df, model_filename='fraud_model_best.pkl',
                           results_filename='model_comparison_results.csv'):
    """
    Сохраняет лучшую модель и результаты сравнения.

    Args:
        best_model: Лучшая обученная модель.
        results_df (pd.DataFrame): DataFrame с метриками всех моделей.
        model_filename (str): Имя файла для сохранения модели.
        results_filename (str): Имя файла для сохранения результатов.
    """
    joblib.dump(best_model, model_filename)
    results_df.to_csv(results_filename)
    print(f"\nМодель сохранена как '{model_filename}'")
    print(f"Результаты сравнения сохранены в '{results_filename}'")


def main():
    """
    Основная функция пайплайна обучения моделей.
    Загружает данные, обучает модели, сравнивает результаты и сохраняет лучшую модель.
    """
    # Загрузка и подготовка данных
    X_train, X_test, y_train, y_test, cat_cols, num_cols, scale_pos_weight = load_and_prepare_data(
        'transactions.csv',
        sample_size=100000
    )

    # Создание препроцессора
    preprocessor = create_preprocessor(cat_cols, num_cols)

    # Получение конфигураций моделей
    models_config = get_models_config(scale_pos_weight)

    # Обучение всех моделей
    results = {}
    best_estimators = {}

    for name, config in models_config.items():
        try:
            best_model, metrics = train_and_evaluate_model(
                name, config, preprocessor, X_train, y_train, X_test, y_test
            )
            results[name] = metrics
            best_estimators[name] = best_model
        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")
            results[name] = {'F1': 0, 'Precision': 0, 'Recall': 0, 'ROC-AUC': 0}

    # Сравнение результатов
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    print(results_df.round(4))

    # Выбор лучшей модели
    best_model_name = results_df['F1'].idxmax()
    best_model = best_estimators[best_model_name]
    best_score = results_df.loc[best_model_name, 'F1']

    print(f"\n🏆 Лучшая модель: {best_model_name} с F1-Score = {best_score:.4f}")

    # Анализ важности признаков
    analyze_feature_importance(best_model, num_cols, cat_cols, top_n=15)

    # Сохранение модели и результатов
    save_model_and_results(best_model, results_df)


if __name__ == "__main__":
    main()
