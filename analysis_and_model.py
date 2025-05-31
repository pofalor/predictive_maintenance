import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    data = None

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены!")
        st.write(f"Записей: {len(data)}")

        # Предобработка данных
        st.subheader("Предобработка данных")
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])

        # Сохраняем кодировщик для формы предсказания
        st.session_state['label_encoder'] = le

        # Проверка на пропуски
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.warning(f"Обнаружены пропуски: {missing_values[missing_values > 0]}")
        else:
            st.success("Пропуски в данных отсутствуют")

        # Разделение данных
        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Сохраняем скейлер для формы предсказания
        st.session_state['scaler'] = scaler

        # Обучение модели
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        st.session_state['model'] = model

        # Оценка модели
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Визуализация результатов
        st.subheader("Результаты обучения модели")
        st.write(f"**Точность модели:** {accuracy:.4f}")

        st.write("**Матрица ошибок:**")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Норма', 'Отказ'],
                    yticklabels=['Норма', 'Отказ'])
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Факт')
        st.pyplot(fig)

        st.write("**Отчет классификации:**")
        st.text(class_report)

        # Анализ признаков
        st.subheader("Важность признаков")
        feature_importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importances.plot(kind='bar', ax=ax)
        ax.set_title("Важность признаков для модели")
        ax.set_ylabel("Важность")
        st.pyplot(fig)

    # Интерфейс для предсказания
    st.header("Предсказание по новым данным")
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")

        col1, col2 = st.columns(2)
        with col1:
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха [K]", value=298.0)
            process_temp = st.number_input("Температура процесса [K]", value=308.0)

        with col2:
            rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=0)

        submit_button = st.form_submit_button("Предсказать")

        if submit_button:
            if 'model' not in st.session_state:
                st.error("Сначала загрузите данные и обучите модель!")
            else:
                # Подготовка входных данных
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })

                # Преобразование категориальной переменной
                input_data['Type'] = st.session_state['label_encoder'].transform(
                    input_data['Type']
                )

                # Масштабирование
                input_data_scaled = st.session_state['scaler'].transform(input_data)

                # Предсказание
                model = st.session_state['model']
                prediction = model.predict(input_data_scaled)
                prediction_proba = model.predict_proba(input_data_scaled)

                # Отображение результатов
                st.subheader("Результат предсказания")
                if prediction[0] == 1:
                    st.error("⚠️ **Прогнозируется отказ оборудования**")
                else:
                    st.success("✅ **Оборудование работает нормально**")

                st.write(f"Вероятность отказа: {prediction_proba[0][1]:.4f}")

                # Визуализация вероятностей
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(['Норма', 'Отказ'], prediction_proba[0], color=['green', 'red'])
                ax.set_xlim(0, 1)
                ax.set_title("Вероятности предсказания")
                st.pyplot(fig)