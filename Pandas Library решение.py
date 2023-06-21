import pandas as pd
import numpy as np

df_data = pd.read_csv(r"C:\Users\iriss\Desktop\Програмирование\Анлитика\Д.З\pandas\diabetes.csv")
pd.options.display.expand_frame_repr = False
df_data.rename(columns={'Class': 'diabetes'}, inplace=True )
df_data.columns = map(str.lower, df_data.columns)  # превидение в нижний регистр
df_data.loc[df_data['diabetes'] == "True", 'diabetes'] = 1  # изменила булевое значение на единицу
df_data['diabetes'] = df_data['diabetes'].astype(int)
print(df_data.dtypes['diabetes'])
print(df_data)

print('*' * 50)
print('Задача.1 Посчитайте количество пропусков в каждой из колонок')

missing_values = df_data.isna().sum()
print(missing_values)

print('*' * 50)
print('Задача.2 Замена пропусков в дискретных признаках')

df_discrete = df_data.select_dtypes(include=['int', 'bool']) # выбор всех колонок с типами int и bool
df_discrete = df_discrete.fillna(df_discrete.median()) # замена пропусков медианными значениями
print(df_discrete)

# Замена пропусков в непрерывных признаках
df_continuous = df_data.select_dtypes(include=['float']) # выбор всех колонок с типом float
df_continuous = df_continuous.fillna(df_continuous.mean()) # замена пропусков средними значениями
print(df_continuous)

print('*' * 50)
print('**Задание 3.** Вычислите основные статистики (минимум, максимум, среднее, дисперсию, квантили) для всех столбцов.')
print(df_data.describe().round(2))

print('*' * 50)
print('**Задание 4.** У скольких женщин старше 50 лет обнаружен диабет?')
count = df_data[(df_data['age'] > 50) & (df_data['diabetes'] == 1)].count()
print(count)

print('*' * 50)
print('**Задание 5.** Найдите трех женщин с наибольшим числом беременностей.')
top_female_df = df_data.sort_values(by='pregnancies', ascending=False).head(3)
print(top_female_df)

print('*' * 50)
print('**Задание 6.** Сколько женщин возраста между 30 и 40 успело родить 3 или более детей?')
female_df = df_data.query('age >= 30 and age <= 40 and pregnancies >= 3')
count = len(female_df)
print('У', count, 'женщин 3 или более детей.')

print('*' * 50)
print('**Задание 7.** У какого процента женщин давление нормальное? Нормальное кровяное давление в диапазоне [80-89].')
normal_bp_df = df_data.query('80 <= bloodpressure <= 89')
count_normal_bp = len(normal_bp_df)
count_female = len(df_data)
percent = count_normal_bp / count_female * 100
print(f'У {percent.__round__(1)} женщин нормальное кровяное давление.')

print('*' * 50)
print('**Задание 8.** У скольких женщин с признаками ожирения кровяное давление выше среднего?')
df_obese = df_data[(df_data.bmi >= 30) & (df_data.bloodpressure >= 90)] # df_data[df_data['bmi'] >= 30]
print(f'У {len(df_obese)} женщин.')

print('*' * 50)
print('**Задание 9.** У кого из женщин обнаружен диабет, и тех, у кого его нет?')
# разделение данных на две группы
diabetes = df_data[df_data['diabetes'] == 1]
no_diabetes = df_data[df_data['diabetes'] == 0]

# вычисление средних значений для каждого признака в каждой из двух групп
glucose_mean_diabetes = diabetes['glucose'].mean()
glucose_mean_no_diabetes = no_diabetes['glucose'].mean()

blood_pressure_mean_diabetes = diabetes['bloodpressure'].mean()
blood_pressure_mean_no_diabetes = no_diabetes['bloodpressure'].mean()

insulin_mean_diabetes = diabetes['insulin'].mean()
insulin_mean_no_diabetes = no_diabetes['insulin'].mean()

print('Средние значения для признака Glucose:')
print('У тех, у кого обнаружен диабет:', glucose_mean_diabetes.round(2))
print('У тех, у кого диабета нет:', glucose_mean_no_diabetes.round(2))

print('\nСредние значения для признака BloodPressure:')
print('У тех, у кого обнаружен диабет:', blood_pressure_mean_diabetes.round(2))
print('У тех, у кого диабета нет:', blood_pressure_mean_no_diabetes.round(2))

print('\nСредние значения для признака Insulin:')
print('У тех, у кого обнаружен диабет:', insulin_mean_diabetes.round(2))
print('У тех, у кого диабета нет:', insulin_mean_no_diabetes.round(2))

print('*' * 50)
print('**Задание 10.** Добавьте новый бинарный признак.')
df_data['waspregnant'] = np.where(df_data['pregnancies'] >= 1, '1', '0')
print(df_data)

print('*' * 50)
print('**Задание 11.** Сравните процент больных диабетом среди женщин, которые были беременны и не были.')
# создание двух отдельных DataFrame для женщин, которые были беременны, и для женщин, которые не были беременны
pregnant_women = df_data[df_data['pregnancies'] > 0]
non_pregnant_women = df_data[df_data['pregnancies'] == 0]

# вычисление процента больных диабетом в каждой группе
pregnant_diabetes_percent = (pregnant_women['diabetes'].sum() / len(df_data)) * 100
non_pregnant_diabetes_percent = (non_pregnant_women['diabetes'].sum() / len(df_data)) * 100

print("Процент больных диабетом среди женщин, которые были беременны: {:.2f}%".format(pregnant_diabetes_percent))
print("Процент больных диабетом среди женщин, которые не были беременны: {:.2f}%".format(non_pregnant_diabetes_percent))

print('*' * 50)
print('**Задание 12.** Добавьте новый категориальный признак __bodyType__ на основе столбца BMI')
bins = [0, 18.5, 25, 30, float('inf')]
labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
df_data['bodytype'] = pd.cut(df_data['bmi'], bins=bins, labels=labels)
print(df_data)

print('*' * 50)
print('**Задание 13.** Какой процент "здоровых" женщин больны диабетом? **** НЕ УВЕРЕНА, ЧТО ВЕРНО!!!')
# Создаем новый столбец, указывающий, является ли женщина здоровой или нет
df_data['healthy'] = (df_data['bodytype'] == 'Normal weight') & df_data['bloodpressure'].between(80, 89)

# # Группируем данные по столбцу "health" и считаем количество больных диабетом
healthy_diabetic_women = df_data.groupby('healthy')['diabetes'].sum()

# Вычисляем процент здоровых женщин, больных диабетом
percent = (healthy_diabetic_women[True] / (healthy_diabetic_women[True] + healthy_diabetic_women[False])) * 100
#
print("{:.2f}% здоровых женщин, больных диабетом".format(percent))








