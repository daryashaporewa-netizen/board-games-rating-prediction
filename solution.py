### Практическое задание №4 «Предсказание рейтинга настольных игр»
## Вдовина Д.Н.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# загружаем данные train датасет
train = pd.read_csv('/Users/darasaporeva/Desktop/Python/data/train_data.csv')
train.info() # выводим информацию о данных train датасет
train.describe() # выводим статистическое описание данных train датасет
# загружаем данные test датасет
test = pd.read_csv('/Users/darasaporeva/Desktop/Python/data/test_data.csv')
test.info() # выводим информацию о данных test датасет
# загружаем данные
display(train.head())
display(test.head())
# проверяем количество строк и столбцов, а также названия столбцов в train и test датасетах
print(train.shape)
print(test.shape)
print(train.columns.tolist())
print(test.columns.tolist())
# Найдем разницу между train и test датасетами в признаках
set(train.columns) - set(test.columns)
Мы проверили количество признаков в датасетах и их различия между ними. Два датасета отличаются друг от друга одним признаком Rating Average, который в контектсе нашей задачи будет являться target-ом.
При анализе данных выявлено что в признаке Rating Average (средний рейтинг) (в train) и в Complexity Average (средняя сложность) (и в train и в test) данные формата object. Это связано с тем, что в эти признаки внесены сведения с ипользованием разделительного знака ',' вместо '.' произведем преобразование данных для корректного отражения и последующего анализа. 
# преобразуем данные в признаке Complexity Average для train и test датасетов
for df in [train, test]:
    df['Complexity Average'] = df['Complexity Average'].str.replace(',', '.').astype(float)
    
# проверяем тип данных после преобразования
print(train['Complexity Average'].dtype)
print(test['Complexity Average'].dtype)
# преобразуем данные в признаке Rating Average для train датасета
train['Rating Average'] = train['Rating Average'].str.replace(',', '.').astype(float)

# проверяем тип данных после преобразования
print(train['Rating Average'].dtype)
# заполняем пропуски в признаке Year Published медианой для train и test датасетов
for df in [train, test]:
    df['Year Published'] = df['Year Published'].fillna(df['Year Published'].median())
print(train['Domains'].nunique())  # сколько уникальных значений
# проверяем наличие пропусков в признаке Domains для train датасета
train['Domains'].value_counts()
# проверяем наличие пропусков в признаке Mechanics для train датасета
train['Mechanics'].value_counts()
В признаке Domains пропусков около 50%. Это категориальный признак состоящий из мульти объектов. Прежде чем удалять этот признак мы должны оценить его влияние на рейтинг. Для этого заполним пропуски.

В признаке Mechanics также много пропусков и это также категориальный приизнак из мульти объектов. Алгоритм действий такой же как и с Domains&
# заполняем пропуски в признаках Domains и Mechanics для train и test датасетов
for df in [train, test]:
    df['Domains'] = df['Domains'].fillna('Unknown')
    df['Mechanics'] = df['Mechanics'].fillna('Unknown')
    
# проверяем наличие пропусков после заполнения
print(train['Domains'].isnull().sum())
print(train['Mechanics'].isnull().sum())
В признаке Owned Users таке есть пропуски, заполним пропуски медианным значением расчитывая медиану для каждого датасета отдельно
# заполняем пропуски в признаке Owned Users медианой для train и test датасетов
for df in [train, test]:
    df['Owned Users'] = df['Owned Users'].fillna(df['Owned Users'].median())
Признаки Domains и Mechanics и Name содержат текстовые значения — модель не умеет работать с текстом напрямую. поэтому Признаки Domains и Mechanics нужно преобразовать в числа, а признак Name в связи с тем что не несет полезной информации для обучения модели, мы будем удалять перед обучением модели. 

Для кодирования используем One-Hot Encoding: каждая уникальная категория становится отдельным столбцом со значением 0 или 1. Поскольку в одной ячейке может быть несколько значений через запятую, применяем MultiLabelBinarizer.

Для Domains кодируем все категории — их всего около 9 штук.
Для Mechanics берём только топ-20 самых популярных — остальные сотни механик встречаются редко и не несут полезной информации для модели.
from sklearn.preprocessing import MultiLabelBinarizer
# Признаки Domains и Mechanics содержат текстовые значения — модель не умеет работать с текстом напрямую, поэтому их нужно преобразовать в числа.
# Используем One-Hot Encoding: каждая уникальная категория становится отдельным столбцом со значением 0 или 1. Поскольку в одной ячейке может быть несколько
# Разбиваем строки на списки, заполняем пропуски
for df in [train, test]:
    df['Domains_list'] = df['Domains'].fillna('Unknown').str.split(',').apply(
        lambda x: [i.strip() for i in x]
    )
    df['Mechanics_list'] = df['Mechanics'].fillna('Unknown').str.split(',').apply(
        lambda x: [i.strip() for i in x]
    )

# Кодируем Domains (все уникальные значения) 
mlb_domains = MultiLabelBinarizer()
domains_train = pd.DataFrame(
    mlb_domains.fit_transform(train['Domains_list']),
    columns=mlb_domains.classes_
)
domains_test = pd.DataFrame(
    mlb_domains.transform(test['Domains_list']),
    columns=mlb_domains.classes_
)

# Кодируем Mechanics (топ-20 самых популярных)
top_mechanics = train['Mechanics_list'].explode().value_counts().head(20).index
mlb_mechanics = MultiLabelBinarizer(classes=top_mechanics)
mechanics_train = pd.DataFrame(
    mlb_mechanics.fit_transform(train['Mechanics_list']),
    columns=mlb_mechanics.classes_
)
mechanics_test = pd.DataFrame(
    mlb_mechanics.transform(test['Mechanics_list']),
    columns=mlb_mechanics.classes_
)

print(f'Domains столбцов: {domains_train.shape[1]}')
print(f'Mechanics столбцов: {mechanics_train.shape[1]}')
# Присоединяем закодированные признаки к train и test
train = pd.concat([train.reset_index(drop=True), 
                   domains_train.reset_index(drop=True),
                   mechanics_train.reset_index(drop=True)], axis=1)

test = pd.concat([test.reset_index(drop=True),
                  domains_test.reset_index(drop=True),
                  mechanics_test.reset_index(drop=True)], axis=1)
# проверяем итоговые размеры датасетов после добавления новых признаков
print(f'Train: {train.shape}')
print(f'Test: {test.shape}')
Удалим признаки в которых больше нет необходимости. 
ID, Name мы удаляем потому что это неинформативные признаки в которых нет необходимости при обучении модели
Domains, Mechanics, Domains_list, Mechanics_list - это признаки которые мы уже закодировали и которые уже тоже больше не нужны
# Удаляем ненужные столбцы из train и test
cols_to_drop = ['ID', 'Name', 'Domains', 'Mechanics', 'Domains_list', 'Mechanics_list']

for df in [train, test]:
    df.drop(columns=cols_to_drop, inplace=True)

print(f'Train: {train.shape}')
print(f'Test: {test.shape}')
# посмотрим распределение таргета
plt.figure(figsize=(10, 5))
sns.histplot(train['Rating Average'], bins=50, kde=True, color='steelblue')
plt.title('Распределение таргета Rating Average')
plt.xlabel('Rating Average')
plt.ylabel('Количество игр')
plt.show()
Перед обучением модели проверим каким образом распределен target. В нашей задаче распределение близко к нормальному, пик около 6-6.5. Значительных выбросов в данных не наблюдается. 
С нормальным распределением хорошо работают модели регрессии, дополнительных преобразований таргета не требуется.
# Сформируем корреляционную матрицу с целью выявления зависимостей между признаками и таргетом и степенью влияния на результат
plt.figure(figsize=(20, 16))
sns.heatmap(train.corr(), 
            annot=True, fmt='.2f', 
            cmap='coolwarm',
            annot_kws={'size': 7})
plt.title('Полная корреляционная матрица')
plt.show()
## Вывод по корреляционной матрице

Наибольшее влияние на Rating Average оказывают числовые признаки:
BGG Rank (-0.74) — сильная отрицательная корреляция: чем ниже место в рейтинге BGG, тем выше средний рейтинг игры
Complexity Average (0.47) — положительная корреляция: более сложные игры оцениваются выше
Users Rated и Owned Users (0.18) — слабая положительная корреляция: популярные игры оцениваются чуть выше

Признаки Play Time, Year Published, Max Players имеют корреляцию близкую к нулю — линейной связи с таргетом не обнаружено.

Закодированные признаки Domains и Mechanics также показывают корреляцию близкую к нулю. Однако это не означает что они бесполезны — модели на основе деревьев (Random Forest, Gradient Boosting) умеют находить нелинейные зависимости, поэтому эти признаки оставляем для обучения.
# выделим таргет и признаки для обучения модели
y = train['Rating Average']
X = train.drop(columns=['Rating Average'])

print(f'X: {X.shape}')
print(f'y: {y.shape}')
# разделим данные на обучающую и валидационную выборки
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')
Мы разделили данные на признаки X (38 признаков) и таргет y (Rating Average).

Далее делим X и y на train (80%) и validation (20%). Validation set нужен для оценки качества модели на данных которые она не видела при обучении — это позволяет сравнивать модели между собой и выбирать лучшую до финального предсказания на test.
## Выбор моделей для обучения

Задача — регрессия (предсказание числового значения), поэтому логистическая регрессия не подходит — она используется только для классификации.
SVM (SVR) и KNN не используем — на датасете из 15000 строк и 38 признаков они обучаются медленно и показывают слабые результаты по сравнению с бустингом.

Мы выбрали три модели представляющие разные подходы:
Ridge - линейная модель, простой baseline для сравнения
Random Forest - ансамбль деревьев, находит нелинейные зависимости
Gradient Boosting — последовательный ансамбль


# обучим модели и сравним их качество на валидационной выборке
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

SEED = 42

models = {
    'Ridge': Ridge(alpha=10),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, 
                                          random_state=SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                   max_depth=5, random_state=SEED),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    results[name] = mse
    print(f'{name:25s}  Val MSE = {mse:.5f}')
## Мы выбрали три модели представляющие разные подходы:
Ridge - линейная модель,
Random Forest - ансамбль деревьев,
Gradient Boosting — последовательный ансамбль

Худший результат в нашем случае показала линейная модель с MSE 0.32. Такой результат связан с тем, что линейная модель ищет, ищет линейную зависимость между признаками и таргетом.

Бустинг строит решающие деревья, которые «нарезают» пространство на кусочки, легко подстраиваясь под любые сложные формы и изгибы данных. Учитывая ошибки прошлых алгоритмов обучая модель последовательно. В связи с алгоритмом обучения GradientBoosting показал лучший результат с MSE 0.045, что является хорошим результатом. 

# Перейдем к подбору гиперпараметров чтобы узнать какие именно значения дают лучший результат
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 500],      # количество деревьев
    'learning_rate': [0.01, 0.05, 0.1],     # скорость обучения
    'max_depth': [3, 5, 7],         # глубина деревьев
}
# настраиваем GridSearchCV для GradientBoostingRegressor
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42), # базовая модель для настройки
    param_grid=param_grid, #  передаём словарь с параметрами которые хотим перебрать
    scoring='neg_mean_squared_error', # метрика для оценки качества (MSE, но с отрицательным знаком, так как GridSearchCV максимизирует метрику)
    cv=3, # количество фолдов для кросс-валидации
    n_jobs=-1, # использовать все доступные ядра процессора для ускорения
    verbose=1 # показывать прогресс во время обучения
)

grid_search.fit(X_train, y_train) #  запускаем перебор

print(f'Лучшие параметры: {grid_search.best_params_}')
print(f'Лучший MSE: {-grid_search.best_score_:.5f}')
# после подбора гиперпараметров обучаем модель на всей обучающей выборке и оцениваем на валидации
best_model = grid_search.best_estimator_ # получаем лучшую модель с оптимальными параметрами
val_pred = best_model.predict(X_val) # делаем предсказания на валидационной выборке
mse = mean_squared_error(y_val, val_pred) # вычисляем MSE на валидации
print(f'Финальный Val MSE: {mse:.5f}')
GridSearchCV - это инструмент для автоматического перебора параметров проверяет качество через cross-validation (cv=3), а не на валидационной выборке. Cross-validation даёт чуть более консервативную оценку — она усредняет результат по 3 разным разбивкам данных, поэтому MSE немного выше чем на одном X_val.

Но у cross-validation более надёжная оценка чем один validation set.

При подборе мы получили лучшие параметры: learning_rate (скорость обучения) 0.05, max_depth (глубина деревьев): 5, n_estimators (количество деревьев): 500

По итогам подбора гиперпараментров при проверке результат улучшился с MSE 0.045 до MSE 0.043
# финальное предсказание на test и сохранение submission.csv
test_preds = best_model.predict(test)

# Сохранение submission.csv
submission = pd.DataFrame({
    'index': range(test.shape[0]),
    'Rating Average': test_preds})

submission.to_csv('/Users/darasaporeva/Desktop/submission.csv', index=False)
print('submission.csv сохранён ✓')
submission.head()
