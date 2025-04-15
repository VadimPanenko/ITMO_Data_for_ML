import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import vstack
import matplotlib.pyplot as plt
import seaborn as sns

# Параметры
CLEANED_DATA_FILE = "../task3/cleaned_data.csv"
LOG_FILE = "selected_for_labeling_log.txt"
IMG_LEARNING_CURVE_FILE = "task4_learning_curve.png"
INITIAL_TRAIN_SIZE = 50  # Размер начального обучающего набора
TEST_SIZE = 0.2  # Доля данных для тестового набора
N_ITERATIONS = 10  # Количество итераций активного обучения
BATCH_SIZE = 20  # Сколько примеров выбираем для разметки на каждой итерации
RANDOM_STATE = 42

print(f"Чтение очищенных данных из {CLEANED_DATA_FILE}...")
df = pd.read_csv(CLEANED_DATA_FILE)

# Убираем строки, где не удалось получить final_label (если такие есть)
df = df.dropna(subset=["final_label", "title"])
df["final_label"] = df["final_label"].astype(int)

print(f"Всего доступно {len(df)} записей с метками.")

# Разделение данных
# 1. Отделяем тестовый набор
train_val_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["final_label"]
)

# 2. Из оставшихся данных выделяем начальный обучающий набор
labeled_df, unlabeled_df = train_test_split(
    train_val_df,
    train_size=min(INITIAL_TRAIN_SIZE, len(train_val_df)),
    random_state=RANDOM_STATE,
    stratify=train_val_df["final_label"],
)

# Сохраняем исходные индексы для пула неразмеченных, чтобы правильно их удалять
unlabeled_indices = unlabeled_df.index.tolist()

print(
    f"Размеры наборов: Начальный обучающий = {len(labeled_df)}, Пул неразмеченных = {len(unlabeled_df)}, Тестовый = {len(test_df)}"
)

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)

# Обучаем векторизатор ТОЛЬКО на начальных данных
X_labeled = vectorizer.fit_transform(labeled_df["title"])
y_labeled = labeled_df["final_label"].values

# Трансформируем оставшиеся данные
X_unlabeled = vectorizer.transform(unlabeled_df["title"])
X_test = vectorizer.transform(test_df["title"])
y_test = test_df["final_label"].values

# Модель
model = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced", C=0.5)

# Лог выбранных индексов
selected_indices_log = []
# Списки для хранения метрик и размера обучающего набора
history = {"n_samples": [], "accuracy": [], "f1_score": []}
sns.set_theme(style="whitegrid")

print("\n--- Начало цикла активного обучения ---")
print(
    "Стратегия выбора: Наименьшая уверенность (Least Confidence Uncertainty Sampling)"
)

for i in range(N_ITERATIONS):
    print(f"\nИтерация {i+1}/{N_ITERATIONS}")

    # Перед началом цикла запишем метрики для начального набора
    if i == 0:
        initial_model = LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced", C=0.5
        )
        if X_labeled.shape[0] > 0:
            initial_model.fit(X_labeled, y_labeled)
            y_pred_initial = initial_model.predict(X_test)
            initial_accuracy = accuracy_score(y_test, y_pred_initial)
            initial_f1 = f1_score(y_test, y_pred_initial, average="weighted")
            history["n_samples"].append(X_labeled.shape[0])
            history["accuracy"].append(initial_accuracy)
            history["f1_score"].append(initial_f1)
            print(
                f"  Начальное качество (на {X_labeled.shape[0]} образцах): Accuracy={initial_accuracy:.4f}, F1={initial_f1:.4f}"
            )
        else:
            history["n_samples"].append(0)
            history["accuracy"].append(0)
            history["f1_score"].append(0)

    # 1. Обучение модели
    print(f"  Обучение модели на {X_labeled.shape[0]} примерах...")
    # Проверяем, есть ли данные для обучения
    if X_labeled.shape[0] == 0:
        print("  Нет данных для обучения модели.")
        break
    model.fit(X_labeled, y_labeled)

    # 2. Оценка на тестовом наборе
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="weighted")
    print(
        f"  Текущее качество на тесте: Accuracy = {accuracy:.4f}, F1-score = {f1:.4f}"
    )
    # Сохраняем метрики ПОСЛЕ обучения на текущем X_labeled
    history["n_samples"].append(X_labeled.shape[0])
    history["accuracy"].append(accuracy)
    history["f1_score"].append(f1)

    # 3. Проверка, остались ли неразмеченные данные
    if X_unlabeled.shape[0] == 0:
        print("  Пул неразмеченных данных исчерпан.")
        break

    # 4. Предсказание вероятностей для неразмеченных данных
    probas_unlabeled = model.predict_proba(X_unlabeled)

    # 5. Вычисление неуверенности (Least Confidence)
    uncertainty_scores = 1 - np.max(probas_unlabeled, axis=1)

    # 6. Выбор K наиболее неуверенных примеров
    # Сортируем по неуверенности и берем индексы ОТНОСИТЕЛЬНО ТЕКУЩЕГО X_unlabeled
    num_to_query = min(BATCH_SIZE, X_unlabeled.shape[0])
    query_indices_relative = np.argsort(uncertainty_scores)[::-1][:num_to_query]

    # Получаем соответствующие ИСХОДНЫЕ индексы из unlabeled_indices
    query_indices_original = [unlabeled_indices[i] for i in query_indices_relative]
    selected_indices_log.extend(
        df.loc[query_indices_original, "original_index"].tolist()
    )

    print(
        f"  Выбрано {len(query_indices_original)} примеров для разметки (индексы в исходном файле): {df.loc[query_indices_original, 'original_index'].tolist()[:5]}..."
    )

    # 7. "Разметка" выбранных примеров (получаем данные из исходного df)
    queried_data = df.loc[query_indices_original]
    y_queried = queried_data["final_label"].values
    # Трансформируем ТОЛЬКО тексты выбранных данных
    X_queried = vectorizer.transform(queried_data["title"])

    # 8. Добавление размеченных данных в обучающий набор
    X_labeled = vstack([X_labeled, X_queried])
    y_labeled = np.concatenate([y_labeled, y_queried])

    # 9. Удаление выбранных примеров из пула неразмеченных
    # Обновляем X_unlabeled
    mask_to_keep = np.ones(X_unlabeled.shape[0], dtype=bool)
    mask_to_keep[query_indices_relative] = False
    X_unlabeled = X_unlabeled[mask_to_keep]

    # Обновляем список исходных индексов неразмеченных данных
    unlabeled_indices = [
        idx for i, idx in enumerate(unlabeled_indices) if mask_to_keep[i]
    ]

    print(
        f"  Размер обучающего набора: {X_labeled.shape[0]}, Пул неразмеченных: {X_unlabeled.shape[0]}"
    )

print("\n--- Цикл активного обучения завершен ---")

# Итоговая оценка (если модель обучалась)
if "model" in locals() and hasattr(model, "predict"):
    print("\nИтоговая оценка модели на тестовом наборе:")
    y_pred_test_final = model.predict(X_test)
    accuracy_final = accuracy_score(y_test, y_pred_test_final)
    f1_final = f1_score(y_test, y_pred_test_final, average="weighted")
    print(f"  Accuracy = {accuracy_final:.4f}")
    print(f"  F1-score = {f1_final:.4f}")
else:
    print("\nМодель не была обучена, итоговая оценка невозможна.")

# Построение графика кривой обучения
print(f"\nПостроение графика кривой обучения...")
history_df = pd.DataFrame(history)

plt.figure(figsize=(10, 6))
sns.lineplot(data=history_df, x="n_samples", y="accuracy", marker="o", label="Accuracy")
sns.lineplot(
    data=history_df,
    x="n_samples",
    y="f1_score",
    marker="o",
    label="F1-score (weighted)",
)
plt.title("Кривая обучения модели (Активное обучение)")
plt.xlabel("Количество размеченных примеров")
plt.ylabel("Метрика на тестовом наборе")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(IMG_LEARNING_CURVE_FILE)
print(f"График кривой обучения сохранен в {IMG_LEARNING_CURVE_FILE}")
plt.close()

# Сохранение лога выбранных индексов
print(
    f"\nСохранение оригинальных индексов выбранных для разметки записей в {LOG_FILE}..."
)
with open(LOG_FILE, "w") as f:
    f.write(
        "# Оригинальные индексы записей (из combined_dataset...), выбранных для доразметки на каждой итерации\n"
    )
    iteration_size = BATCH_SIZE
    for i in range(0, len(selected_indices_log), iteration_size):
        batch = selected_indices_log[
            i : min(i + iteration_size, len(selected_indices_log))
        ]
        f.write(f"Iteration {i//iteration_size + 1}: {batch}\n")

print("Скрипт активного обучения завершен.")
