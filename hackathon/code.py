import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # Добавим этот импорт
import pickle

# Произведем загрузку и стандартизацию данных
file_path = 'C:/Users/user/Desktop/Dataset.csv'
data = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)

# Определение целевых переменных и признаков
fields = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

# Произведем стандартизацию данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop("HeartDiseaseorAttack", axis=1))

# Загрузка лучших параметров из файла
best_params_df = pd.read_csv('C:/Users/user/Desktop/best_params.csv')
best_xgb_params = best_params_df.iloc[0].to_dict()

# Преобразование значений параметров к целочисленному формату
best_xgb_params['n_estimators'] = int(best_xgb_params['n_estimators'])
best_xgb_params['max_depth'] = int(best_xgb_params['max_depth'])

# Создание и обучение модели с лучшими параметрами
best_xgb_model = XGBClassifier(**best_xgb_params, random_state=42)
best_xgb_model.fit(X_scaled, data["HeartDiseaseorAttack"])

# Функция для обработки ввода пользователя и вывода результата
def on_predict_button_click():
    # Получение данных из полей ввода
    user_data = []
    for entry in entry_fields:
        user_data.append(float(entry.get()))

    # Создание DataFrame с введенными данными
    new_patient_data = pd.DataFrame([user_data], columns=fields)

    # Преобразование новых данных
    new_data_scaled = scaler.transform(new_patient_data)

    # Предсказание с использованием обученной модели
    result = best_xgb_model.predict(new_data_scaled)

    # Вывод результата в окно
    if result[0] == 1:
        result_label.config(text="Результат: Высокий риск инфаркта миокарда")
    else:
        result_label.config(text="Результат: Низкий риск инфаркта миокарда")

# Создание GUI
root = tk.Tk()
root.title("Прогноз риска инфаркта")

# Создание меток, полей ввода и кнопки
entry_fields = []
for i, field in enumerate(fields):
    label = ttk.Label(root, text=f"{field}:")
    label.grid(row=i, column=0, padx=10, pady=5)

    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry_fields.append(entry)

# Создание кнопки для предсказания
predict_button = ttk.Button(root, text="Предсказать", command=on_predict_button_click)
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=10)

# Создание метки для вывода результата
result_label = ttk.Label(root, text="")
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=5)

# Запуск основного цикла
root.mainloop()
