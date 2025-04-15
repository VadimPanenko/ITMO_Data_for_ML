import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from collections import Counter

# Параметры
ANNOTATED_FILE = "annotated_data.csv"
CLEANED_FILE = "cleaned_data.csv"
N_ANNOTATORS = 3


# --- Функции ---
def calculate_pairwise_agreement(df, annotator_cols):
    """Расчет попарного согласия (accuracy и kappa)."""
    agreement = {}
    for i in range(len(annotator_cols)):
        for j in range(i + 1, len(annotator_cols)):
            col1 = annotator_cols[i]
            col2 = annotator_cols[j]
            # Убираем строки, где хотя бы один не разметил
            valid_rows = df[[col1, col2]].dropna()
            if not valid_rows.empty:
                acc = accuracy_score(valid_rows[col1], valid_rows[col2])
                kappa = cohen_kappa_score(valid_rows[col1], valid_rows[col2])
                agreement[f"{col1}_vs_{col2}"] = {"accuracy": acc, "kappa": kappa}
            else:
                agreement[f"{col1}_vs_{col2}"] = {"accuracy": np.nan, "kappa": np.nan}
    return agreement


def get_majority_vote(row, annotator_cols, best_annotator_col):
    """Получение метки большинством голосов."""
    votes = [row[col] for col in annotator_cols if pd.notna(row[col])]
    if not votes:
        return np.nan  # Нет голосов

    counts = Counter(votes)
    max_count = max(counts.values())

    # Находим метки с максимальным количеством голосов
    majority_labels = [label for label, count in counts.items() if count == max_count]

    if len(majority_labels) == 1:
        return majority_labels[0]  # Однозначное большинство
    else:
        # Ничья - возвращаем голос лучшего разметчика
        return row[best_annotator_col]


# --- Основной процесс ---

print(f"Чтение данных из {ANNOTATED_FILE}...")
df = pd.read_csv(ANNOTATED_FILE)

annotator_cols = [f"annotator_{i+1}" for i in range(N_ANNOTATORS)]

# 1. Оценка точности по тестовым кейсам
print("\nОценка точности:")
test_cases_df = df.dropna(subset=["final_label"])
annotator_accuracies = {}
if not test_cases_df.empty:
    for col in annotator_cols:
        accuracy = accuracy_score(test_cases_df["final_label"], test_cases_df[col])
        annotator_accuracies[col] = accuracy
        print(f"  {col}: {accuracy:.4f}")

    # Определяем лучшего разметчика
    best_annotator = max(annotator_accuracies, key=annotator_accuracies.get)
    print(
        f"\nЛучший источник: {best_annotator} (Точность: {annotator_accuracies[best_annotator]:.4f})"
    )
else:
    print(
        "  Данные для сравнения не найдены или пусты. Точность не может быть вычислена."
    )
    best_annotator = annotator_cols[0]
    print(f"\nРазрешение ничьих будет производиться по голосу {best_annotator}.")

# 2. Расчет попарного согласия
print("\nРасчет попарной согласованности (Accuracy / Cohen's Kappa):")
pairwise_agreement = calculate_pairwise_agreement(df, annotator_cols)
for pair, scores in pairwise_agreement.items():
    print(
        f"  {pair}: Accuracy = {scores['accuracy']:.4f}, Kappa = {scores['kappa']:.4f}"
    )

# 3. Агрегация меток
print("\nАгрегация меток методом большинства голосов...")
df["final_label"] = df.apply(
    lambda row: get_majority_vote(row, annotator_cols, best_annotator), axis=1
)

# Анализ расхождений (пример)
disagreements = df[df[annotator_cols].nunique(axis=1, dropna=True) > 1]
print(f"\nНайдено {len(disagreements)} записей с расхождениями.")
if not disagreements.empty:
    print("Примеры расхождений:")
    print(disagreements[["title"] + annotator_cols + ["final_label"]].head())

# 4. Сохранение очищенных данных
print(f"\nСохранение данных с итоговой меткой в {CLEANED_FILE}...")
df.to_csv(CLEANED_FILE, index=False)

print("\nАнализ и обработка завершены.")
