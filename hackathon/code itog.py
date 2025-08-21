import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

# Произведем загрузку и стандартизацию данных
file_path = 'C:/Users/user/Desktop/Dataset.csv'
data = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)

# Определение целевых переменных и признаков
fields = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits',
          'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
          'Sex', 'Age']

# Словарь для соответствия вопросов и полей данных
questions_mapping = {
    'HighBP': 'Имеется ли у вас повышенное кровяное давление?',
    'HighChol': 'Имеется ли у вас повышенный уровень холестерина в крови?',
    'CholCheck': 'Проверяли ли вы уровень холестерина в течение последних 5 лет?',
    'BMI': 'Ваш ИМТ (Индекс Массы Тела)',
    'Smoker': 'Курите ли вы?',
    'Stroke': 'Был ли у вас инсульт?',
    'Diabetes': 'Есть ли у вас диабет?',
    'PhysActivity': 'Занимаетесь ли вы хоть какой-то физической активностью?',
    'Fruits': 'Едите ли вы хотя бы 1 фрукт в день?',
    'Veggies': 'Едите ли вы хотя бы 1 овощ в день?',
    'HvyAlcoholConsump': 'Употребяете ли вы алкогольные напитки?',
    'AnyHealthcare': 'Есть ли у вас медицинская страховка?',
    'GenHlth': 'Оцените своё общее состояние от 1 до 5, где 1 - отличное, а 5 - ужасное?',
    'MentHlth': 'Были ли у вас проблемы с психическим здоровьем за последние 30 дней? '
                'Если да, напишите, как много дней назад, если нет, напишите 30',
    'PhysHlth': 'Были ли у вас проблемы с физическим здоровьем за последние 30 дней? '
                'Если да, напишите, как много дней назад, если нет, напишите 30',
    'DiffWalk': 'Испытываете ли вы трудности с ходьбой, бегом, подъёмом по лестницам и т.д?',
    'Sex': 'Ваш пол',
    'Age': 'Сколько вам лет?'
}

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
    for entry_info in ui_elements:
        if entry_info['type'] == 'entry':
            user_data.append(float(entry_info['widget'].get()))
        elif entry_info['type'] == 'radiobutton':
            selected_value = entry_info['widget'][0].get()
            user_data.append(float(selected_value))

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
root.title("Прогноз риска инфаркта миокарда")

# Создание меток, полей ввода и кнопки
ui_elements = []
for i, field in enumerate(fields):
    label_text = questions_mapping.get(field, field)
    label = ttk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5)

    if field in ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "Diabetes", "PhysActivity", "Fruits",
                 "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "DiffWalk"]:
        var = tk.StringVar()
        yes_button = ttk.Radiobutton(root, variable=var, value=1, text="Да")
        yes_button.grid(row=i, column=1, padx=5, pady=5)

        no_button = ttk.Radiobutton(root, variable=var, value=0, text="Нет")
        no_button.grid(row=i, column=2, padx=5, pady=5)

        ui_elements.append({'type': 'radiobutton', 'widget': (var, yes_button, no_button)})
    else:
        entry = ttk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=5)
        ui_elements.append({'type': 'entry', 'widget': entry})

# Создание кнопки для предсказания
predict_button = ttk.Button(root, text="Предсказать", command=on_predict_button_click)
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=10)

# Создание метки для вывода результата
result_label = ttk.Label(root, text="")
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=5)

# Запуск основного цикла
root.mainloop()
