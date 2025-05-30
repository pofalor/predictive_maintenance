import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Предобработка
        data = data.drop(columns=['UDI', 'Product ID'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Разделение данных
        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Обучение модели
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Оценка
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Визуализация
        st.subheader("Результаты модели")
        st.write(f"Точность: {accuracy:.2f}")
        st.write("Матрица ошибок:")
        st.write(confusion_matrix(y_test, y_pred))

        # Интерфейс предсказаний
        with st.form("prediction_form"):
            st.write("Введите параметры оборудования:")
            air_temp = st.number_input("Температура воздуха [K]", value=300.0)
            process_temp = st.number_input("Температура процесса [K]", value=310.0)
            rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
            torque = st.number_input("Крутящий момент [Nm]", value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", value=0)
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])

            if st.form_submit_button("Предсказать"):
                input_data = pd.DataFrame({
                    'Type': [product_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])
                input_data = scaler.transform(input_data)
                prediction = model.predict(input_data)
                st.success(
                    f"Прогноз: {'Отказ' if prediction[0] == 1 else 'Норма'} (Вероятность: {model.predict_proba(input_data)[0][1]:.2f})")