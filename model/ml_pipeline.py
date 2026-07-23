import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, \
    confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных (замените на ваш путь)
print("Загрузка данных...")
df = pd.read_csv('transactions.csv')  # укажите правильный путь
df = df.head(100000)

# Проверка баланса классов
print(f"Распределение классов:\n{df['is_fraud'].value_counts()}")
print(f"Доля мошеннических транзакций: {df['is_fraud'].mean() * 100:.2f}%")

# Столбцы, которые не должны быть признаками (идентификаторы, время, целевая переменная)
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

# Создаем препроцессор с правильной обработкой категориальных признаков
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Считаем scale_pos_weight как отношение количества отрицательных примеров к положительным в обучающей выборке
neg_count = (y_train == 0).sum()  # количество примеров класса 0
pos_count = (y_train == 1).sum()  # количество примеров класса 1
scale_pos_weight = neg_count / pos_count

# Настройка моделей с учетом дисбаланса
models_config = {
    'XGBoost': {
        'model': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,  # Увеличиваем вес для класса мошенничества
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
            class_weight='balanced',  # Автоматическая балансировка классов
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


# Функция для обучения и оценки моделей
def train_and_evaluate_model(name, config, X_train, y_train, X_test, y_test):
    print(f"\n{'=' * 50}")
    print(f"Обучение модели: {name}")
    print('=' * 50)

    # Создаем пайплайн с SMOTE
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),  # Увеличиваем долю минорного класса до 50%
        ('classifier', config['model'])
    ])

    # Для KNN пробуем без SMOTE или с другим подходом
    if name == 'KNN':
        # KNN плохо работает с SMOTE, пробуем с весами
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),  # Меньше oversampling для KNN
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

    # Дополнительная информация
    print(f"\nРезультаты для {name}:")
    print(f"F1-Score: {metrics['F1']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return grid_search.best_estimator_, metrics


# Обучение всех моделей
results = {}
best_estimators = {}

for name, config in models_config.items():
    try:
        best_model, metrics = train_and_evaluate_model(
            name, config, X_train, y_train, X_test, y_test
        )
        results[name] = metrics
        best_estimators[name] = best_model
    except Exception as e:
        print(f"Ошибка при обучении {name}: {e}")
        results[name] = {'F1': 0, 'Precision': 0, 'Recall': 0, 'ROC-AUC': 0}

# Сравнение результатов
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# Выбор лучшей модели
best_model_name = results_df['F1'].idxmax()
best_model = best_estimators[best_model_name]
best_score = results_df.loc[best_model_name, 'F1']

print(f"\nЛучшая модель: {best_model_name} с F1-Score = {best_score:.4f}")

# Сохранение лучшей модели
import joblib

joblib.dump(best_model, 'fraud_model_best.pkl')

# Сохранение результатов сравнения
results_df.to_csv('model_comparison_results.csv')

# Важность признаков (для моделей, поддерживающих это)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # Получаем имена признаков
    feature_names = num_cols + cat_cols
    importances = best_model.named_steps['classifier'].feature_importances_

    # Сортируем по важности
    indices = np.argsort(importances)[::-1]

    print("\nТоп-10 наиболее важных признаков:")
    for i in range(min(10, len(feature_names))):
        print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
