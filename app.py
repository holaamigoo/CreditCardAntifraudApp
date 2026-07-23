import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.express as px
from datetime import datetime
import json
import os
import pickle
import xgboost as xgb

# Настройка страницы
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Загрузка модели и метаданных
@st.cache_resource
def load_model():
    """
    Загружает обученную модель из файла fraud_model_fix.pkl.

    Returns:
        model: Загруженная модель машинного обучения (XGBClassifier).

    Raises:
        st.error: Если модель не найдена или не может быть загружена.
    """
    model_path = 'fraud_model_fix.pkl'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("Модель загружена через joblib")
            return model
        except Exception as e:
            st.warning(f"Загрузка через joblib не удалась: {e}")

    st.error("Не удалось загрузить модель. Проверьте файл.")
    st.stop()


@st.cache_data
def load_unique_values():
    """
    Загружает уникальные значения категориальных признаков из JSON-файла.

    Returns:
        dict: Словарь {название_колонки: [список_уникальных_значений]}.

    Raises:
        st.error: Если файл unique_values.json не найден.
    """
    json_path = 'unique_values.json'
    if not os.path.exists(json_path):
        st.error("Файл unique_values.json не найден. Запустите generate_unique_values.py.")
        st.stop()
    with open(json_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_columns_and_types():
    """
    Загружает информацию о колонках датасета: список признаков и типы числовых полей.

    Returns:
        tuple: (feature_cols_raw, col_types, cat_cols)
            - feature_cols_raw (list): Список всех признаков (до one-hot кодирования).
            - col_types (dict): Словарь {колонка: тип} для числовых полей ('int' или 'float').
            - cat_cols (list): Список категориальных колонок.
    """
    df_sample = pd.read_csv('transactions.csv')
    exclude_cols = ['transaction_id', 'customer_id', 'transaction_time', 'is_fraud']
    feature_cols_raw = [c for c in df_sample.columns if c not in exclude_cols]
    cat_cols = df_sample.select_dtypes(include=['object', 'string']).columns.tolist()
    cat_cols = [c for c in cat_cols if c in feature_cols_raw]
    numeric_cols = [c for c in feature_cols_raw if c not in cat_cols]
    col_types = {}
    for c in numeric_cols:
        if df_sample[c].dtype in ['int64', 'int32']:
            col_types[c] = 'int'
        else:
            col_types[c] = 'float'
    return feature_cols_raw, col_types, cat_cols


# Работа с БД
def init_db():
    """
    Инициализирует базу данных SQLite.
    Создаёт таблицу predictions, если она ещё не существует.
    """
    conn = sqlite3.connect('fraud_predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            prediction INTEGER,
            probability REAL,
            threshold REAL,
            timestamp TEXT,
            input_data TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction(transaction_id, prediction, probability, threshold, input_data):
    """
    Сохраняет результат предсказания в базу данных.

    Args:
        transaction_id (str): Идентификатор транзакции.
        prediction (int): Предсказанный класс (0 или 1).
        probability (float): Вероятность мошенничества.
        threshold (float): Использованный порог классификации.
        input_data (dict): Входные данные транзакции.
    """
    conn = sqlite3.connect('fraud_predictions.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (transaction_id, prediction, probability, threshold, timestamp, input_data) VALUES (?,?,?,?,?,?)",
        (transaction_id, int(prediction), float(probability), float(threshold), datetime.now().isoformat(),
         json.dumps(input_data))
    )
    conn.commit()
    conn.close()


def load_history(limit=200):
    """
    Загружает историю предсказаний из базы данных.

    Args:
        limit (int, optional): Максимальное количество записей. По умолчанию 200.

    Returns:
        pd.DataFrame: DataFrame с историей предсказаний.
    """
    conn = sqlite3.connect('fraud_predictions.db')
    df = pd.read_sql(f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df


# Функция предсказания
def predict_transactions(model, df, threshold=0.5):
    """
    Выполняет предсказание для набора транзакций.

    Применяет one-hot кодирование к категориальным признакам с drop_first=True,
    добавляет недостающие колонки и выполняет предсказание.

    Args:
        model: Обученная модель машинного обучения.
        df (pd.DataFrame): DataFrame с исходными данными транзакций.
        threshold (float, optional): Порог вероятности для классификации. По умолчанию 0.5.

    Returns:
        pd.DataFrame: Исходный DataFrame с добавленными колонками 'probability' и 'prediction'.

    Raises:
        st.error: Если модель не содержит информацию о признаках.
    """
    # Получаем имена признаков, ожидаемых моделью (после one-hot)
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
    else:
        st.error("Модель не содержит информацию о признаках (feature_names_in_).")
        st.stop()

    # Категориальные колонки (из unique_values.json)
    cat_cols = st.session_state.cat_cols

    # Удаляем неиспользуемые колонки
    X_raw = df.drop(columns=['transaction_id', 'customer_id', 'transaction_time', 'is_fraud'], errors='ignore')

    # One-hot кодирование с drop_first=True (как при обучении)
    X_dummies = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    # Добавляем недостающие колонки (если какая-то категория отсутствует)
    for col in feature_cols:
        if col not in X_dummies.columns:
            X_dummies[col] = 0

    # Оставляем только колонки в правильном порядке
    X_dummies = X_dummies[feature_cols]

    # Предсказание
    probs = model.predict_proba(X_dummies)[:, 1]
    preds = (probs >= threshold).astype(int)

    result_df = df.copy()
    result_df['probability'] = probs
    result_df['prediction'] = preds
    return result_df


# Отображение результатов
def display_results(df, low_color, high_color):
    """
    Отображает результаты предсказания в виде таблицы с цветовой кодировкой и графиков.

    Args:
        df (pd.DataFrame): DataFrame с результатами предсказания (содержит 'probability' и 'prediction').
        low_color (float): Нижняя граница для зелёного цвета.
        high_color (float): Верхняя граница для красного цвета.
    """
    st.subheader("Результаты предсказания")
    df_display = df.reset_index(drop=True)
    df_display.insert(0, '№', df_display.index + 1)

    def get_color(prob):
        """Возвращает цвет в зависимости от вероятности."""
        if prob < low_color:
            return '#90EE90'  # зелёный
        elif prob < high_color:
            return '#FFA500'  # оранжевый
        else:
            return '#FF6B6B'  # красный

    def highlight_rows(row):
        """Применяет цвет к строкам таблицы."""
        prob = row['probability']
        color = get_color(prob)
        styles = []
        for col in row.index:
            if col in ['№', 'probability']:
                styles.append(f'background-color: {color}')
            else:
                styles.append('')
        return styles

    styled = df_display.style.apply(highlight_rows, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Статистика
    total = len(df)
    fraud_count = df['prediction'].sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего транзакций", total)
    with col2:
        st.metric("Мошеннических", fraud_count)
    with col3:
        st.metric("Доля мошеннических", f"{fraud_count / total * 100:.2f}%")

    # Графики
    st.subheader("Визуализация")

    fig_hist = px.histogram(df, x='probability', nbins=30,
                            title="Распределение вероятностей мошенничества",
                            color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_pie = px.pie(names=['Легитимные', 'Мошеннические'],
                     values=[total - fraud_count, fraud_count],
                     title="Доля мошеннических транзакций",
                     color_discrete_sequence=['#2ca02c', '#d62728'])
    st.plotly_chart(fig_pie, use_container_width=True)

    # Дополнительные графики по категориям
    if 'merchant_category' in df.columns:
        fig_bar = px.bar(
            df.groupby('merchant_category')['prediction'].mean().reset_index(),
            x='merchant_category', y='prediction',
            title="Доля мошеннических по категории мерчанта",
            labels={'prediction': 'Доля мошеннических', 'merchant_category': 'Категория'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Важность признаков
    if hasattr(st.session_state, 'model') and hasattr(st.session_state.model, 'feature_importances_'):
        st.subheader("Важность признаков (используется модель XGBClassifier)")
        model = st.session_state.model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_cols = st.session_state.feature_cols
            if len(importances) == len(feature_cols):
                imp_df = pd.DataFrame({'Признак': feature_cols, 'Важность': importances})
                imp_df = imp_df.sort_values('Важность', ascending=False).head(15)
                imp_df = imp_df.iloc[::-1]  # самый важный признак сверху
                fig_imp = px.bar(imp_df, x='Важность', y='Признак', orientation='h',
                                 title="Топ-15 важных признаков")
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Модель не предоставляет важность признаков.")


# Описание признаков
def show_feature_descriptions():
    """
    Выводит таблицу с описанием всех признаков транзакции.
    Используется в интерфейсе для справки пользователя.
    """
    st.subheader("Описание признаков транзакции")
    with st.expander("Показать/скрыть описание признаков", expanded=True):
        descriptions = {
            "transaction_id": "Уникальный идентификатор транзакции",
            "customer_id": "Уникальный идентификатор клиента",
            "transaction_time": "Время совершения транзакции",
            "transaction_amount": "Сумма транзакции",
            "merchant_category": "Категория мерчанта",
            "transaction_type": "Тип транзакции",
            "payment_method": "Способ оплаты",
            "city": "Город транзакции",
            "country": "Страна транзакции",
            "device_type": "Тип устройства",
            "operating_system": "Операционная система",
            "browser": "Браузер",
            "card_type": "Тип карты",
            "card_present": "Флаг физического присутствия карты (1 — да, 0 — нет)",
            "international_transaction": "Флаг международной транзакции (1 — да, 0 — нет)",
            "distance_from_home": "Расстояние от дома до места транзакции",
            "previous_transaction_gap": "Время с последней транзакции (в часах)",
            "daily_transaction_count": "Количество транзакций за текущий день",
            "monthly_spend": "Общая сумма трат за месяц",
            "risk_score": "Скоринговый балл риска (чем выше, тем рискованнее)",
            "customer_age": "Возраст клиента",
            "account_tenure_years": "Срок обслуживания счета (в годах)",
            "merchant_risk_level": "Уровень риска мерчанта (Low, Medium, High)",
            "transaction_status": "Статус транзакции",
            "is_fraud": "Целевая переменная (1 — мошенническая, 0 — легитимная)"
        }
        types = {
            "transaction_id": "object",
            "customer_id": "object",
            "transaction_time": "object",
            "transaction_amount": "float64",
            "merchant_category": "object",
            "transaction_type": "object",
            "payment_method": "object",
            "city": "object",
            "country": "object",
            "device_type": "object",
            "operating_system": "object",
            "browser": "object",
            "card_type": "object",
            "card_present": "int64",
            "international_transaction": "int64",
            "distance_from_home": "float64",
            "previous_transaction_gap": "float64",
            "daily_transaction_count": "int64",
            "monthly_spend": "float64",
            "risk_score": "int64",
            "customer_age": "int64",
            "account_tenure_years": "float64",
            "merchant_risk_level": "object",
            "transaction_status": "object",
            "is_fraud": "int64"
        }

        df_desc = pd.DataFrame({
            "Название столбца": list(descriptions.keys()),
            "Тип данных": [types.get(k, "") for k in descriptions.keys()],
            "Описание": list(descriptions.values())
        })
        st.dataframe(df_desc, use_container_width=True, hide_index=True)


# Основная логика приложения
def main():
    """
    Основная функция приложения Streamlit.
    Управляет режимами работы: ручной ввод, загрузка CSV, история.
    """
    init_db()

    # Загрузка модели
    model = load_model()
    st.session_state.model = model

    # Загрузка метаданных
    unique_vals = load_unique_values()
    st.session_state.unique_vals = unique_vals
    cat_cols = list(unique_vals.keys())
    st.session_state.cat_cols = cat_cols

    # Получение списка признаков модели
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
    else:
        st.error("Модель не содержит информацию о признаках (feature_names_in_).")
        st.stop()
    st.session_state.feature_cols = feature_cols

    # Загрузка исходных признаков и их типов для ручного ввода
    feature_cols_raw, col_types, _ = load_columns_and_types()

    # Заголовок и описание
    st.title("Программа обнаружения мошеннических транзакций")
    show_feature_descriptions()

    # Боковая панель с настройками
    st.sidebar.header("Настройки")
    threshold = st.sidebar.slider(
        "Порог вероятности для классификации",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Транзакции с вероятностью выше порога считаются мошенническими"
    )

    st.sidebar.subheader("Цветовая кодировка")
    low_color = st.sidebar.slider(
        "Нижняя граница зелёного", 0.0, 1.0, 0.3, 0.01,
        help="Вероятность ниже этого значения → зелёный"
    )
    high_color = st.sidebar.slider(
        "Верхняя граница красного", 0.0, 1.0, 0.7, 0.01,
        help="Вероятность выше этого значения → красный"
    )
    if low_color >= high_color:
        st.sidebar.warning("Нижняя граница должна быть меньше верхней. Значения скорректированы.")
        low_color, high_color = min(low_color, high_color), max(low_color, high_color)

    # Выбор режима работы
    mode = st.sidebar.radio("Режим работы", ["Ручной ввод", "Загрузка CSV", "История"])

    # Инициализация состояния
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
    if 'results' not in st.session_state:
        st.session_state.results = None

    # РУЧНОЙ ВВОД
    if mode == "Ручной ввод":
        st.header("Ручной ввод транзакции")
        st.markdown(
            "Заполните поля и нажмите **Добавить транзакцию**. Затем выполните предсказание для всех добавленных.")

        with st.form(key='manual_form'):
            col1, col2, col3 = st.columns(3)
            with col1:
                t_id = st.text_input("transaction_id", value="")
            with col2:
                c_id = st.text_input("customer_id", value="")
            with col3:
                t_time = st.text_input("transaction_time", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            cols = st.columns(4)
            input_data = {}
            input_data['transaction_id'] = t_id if t_id else f"MANUAL_{len(st.session_state.transactions)}"
            input_data['customer_id'] = c_id if c_id else "UNKNOWN"
            input_data['transaction_time'] = t_time

            for idx, col in enumerate(feature_cols_raw):
                with cols[idx % 4]:
                    if col in cat_cols:
                        val = st.selectbox(f"{col}", unique_vals[col], key=f"sel_{col}")
                    else:
                        is_int = col_types.get(col, 'float') == 'int'
                        # Реалистичные значения по умолчанию
                        default_values = {
                            'transaction_amount': 100.0,
                            'distance_from_home': 10.0,
                            'previous_transaction_gap': 24.0,
                            'daily_transaction_count': 1,
                            'monthly_spend': 500.0,
                            'risk_score': 50,
                            'customer_age': 35,
                            'account_tenure_years': 2.0
                        }
                        default_val = default_values.get(col, 0)
                        if is_int:
                            val = st.number_input(f"{col}", min_value=0, step=1, value=default_val, format="%d",
                                                  key=f"num_{col}")
                        else:
                            val = st.number_input(f"{col}", min_value=0.0, step=0.01, value=float(default_val),
                                                  format="%.2f", key=f"num_{col}")
                    input_data[col] = val

            submitted = st.form_submit_button("➕ Добавить транзакцию")
            if submitted:
                if input_data.get('transaction_amount', 0) < 0:
                    st.error("Сумма транзакции не может быть отрицательной.")
                else:
                    st.session_state.transactions.append(input_data)
                    st.success(f"Транзакция добавлена! Всего: {len(st.session_state.transactions)}")
                    st.session_state.results = None

        if st.session_state.transactions:
            st.subheader("Список транзакций для проверки")
            df_manual = pd.DataFrame(st.session_state.transactions)
            st.dataframe(df_manual, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Выполнить предсказание", use_container_width=True):
                    with st.spinner("Выполняется предсказание..."):
                        X_new = pd.DataFrame(st.session_state.transactions)
                        missing = set(feature_cols_raw) - set(X_new.columns)
                        if missing:
                            st.error(f"Отсутствуют колонки: {missing}")
                        else:
                            results = predict_transactions(model, X_new, threshold)
                            for idx, row in results.iterrows():
                                t_id = row.get('transaction_id', f"MANUAL_{idx}")
                                input_dict = {k: v for k, v in row.items() if k not in ['prediction', 'probability']}
                                save_prediction(t_id, row['prediction'], row['probability'], threshold, input_dict)
                            st.session_state.results = results
                            st.success("Предсказание выполнено!")
                            st.rerun()
            with col2:
                if st.button("Очистить список", use_container_width=True):
                    st.session_state.transactions = []
                    st.session_state.results = None
                    st.rerun()

    # ЗАГРУЗКА CSV
    elif mode == "Загрузка CSV":
        st.header("Загрузка CSV-файла")
        uploaded_file = st.file_uploader("Выберите CSV файл с транзакциями", type="csv")

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("**Образец загруженных данных:**")
            st.dataframe(df_upload.head(), use_container_width=True)

            missing_feat = set(feature_cols_raw) - set(df_upload.columns)
            if missing_feat:
                st.error(f"В загруженном файле отсутствуют признаки: {missing_feat}")
            else:
                if st.button("Выполнить предсказание для CSV", use_container_width=True):
                    with st.spinner("Обработка..."):
                        results = predict_transactions(model, df_upload, threshold)
                        for idx, row in results.iterrows():
                            t_id = row.get('transaction_id', f"CSV_{idx}")
                            input_dict = {k: v for k, v in row.items() if k not in ['prediction', 'probability']}
                            save_prediction(t_id, row['prediction'], row['probability'], threshold, input_dict)
                        st.session_state.results = results
                        st.success("Предсказание выполнено!")

                        csv_data = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать результаты CSV",
                            data=csv_data,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

    # ИСТОРИЯ
    elif mode == "История":
        st.header("История предсказаний")
        history_df = load_history(limit=200)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
            if st.button("Очистить историю", use_container_width=True):
                conn = sqlite3.connect('fraud_predictions.db')
                conn.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.success("История очищена.")
                st.rerun()
        else:
            st.info("История предсказаний пуста.")

    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    if st.session_state.results is not None and mode != "История":
        display_results(st.session_state.results, low_color, high_color)


if __name__ == "__main__":
    main()