import pandas as pd
import numpy as np
import random
import re

INPUT_FILE = "../data/combined_dataset_20250401_220744.csv"
OUTPUT_FILE = "annotated_data.csv"
N_ANNOTATORS = 3
ANNOTATION_SAMPLE_SIZE = 500
TEST_CASES_SIZE = 50

FINAL_LABELS = {
    0: 1,
    3: 1,
    7: 1,
    14: 1,
    23: 1,
    28: 1,
    31: 1,
    37: 1,
    42: 1,
    49: 1,
    53: 1,
    62: 1,
    64: 1,
    67: 1,
    70: 1,
    73: 1,
    80: 1,
    83: 1,
    88: 1,
    103: 1,
    108: 1,
    115: 1,
    120: 1,
    130: 1,
    136: 1,
    4: 0,
    5: 0,
    10: 0,
    12: 0,
    16: 0,
    20: 0,
    25: 0,
    33: 0,
    40: 0,
    41: 0,
    51: 0,
    54: 0,
    60: 0,
    87: 0,
    93: 0,
    97: 0,
    98: 0,
    100: 0,
    102: 0,
    110: 0,
    113: 0,
    124: 0,
    129: 0,
    131: 0,
}

print(f"Чтение данных из {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Найдено {len(df)} вакансий.")

possible_indices = list(set(range(len(df))) - set(FINAL_LABELS.keys()))
annotation_indices = random.sample(
    possible_indices,
    k=min(ANNOTATION_SAMPLE_SIZE - len(FINAL_LABELS), len(possible_indices)),
)

all_indices_to_annotate = sorted(
    list(set(annotation_indices) | set(FINAL_LABELS.keys()))
)

print(
    f"Подготовлено {len(all_indices_to_annotate)} записей для обработки (включая {len(FINAL_LABELS)} контрольных примеров)."
)

df_subset = df.iloc[all_indices_to_annotate].copy()
df_subset["original_index"] = all_indices_to_annotate

results = []

print("Сбор обработанных данных...")
for index, row in df_subset.iterrows():
    title = row["title"]
    original_idx = row["original_index"]
    record = {
        "original_index": original_idx,
        "id": row["id"],
        "title": title,
        "final_label": FINAL_LABELS.get(original_idx, np.nan),
    }

    results.append(record)

annotated_df = pd.DataFrame(results)

print(f"Сохранение обработанных данных в {OUTPUT_FILE}...")
annotated_df.to_csv(OUTPUT_FILE, index=False)

print("Сбор и сохранение обработанных данных завершены.")
