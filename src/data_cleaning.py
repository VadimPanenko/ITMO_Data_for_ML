import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import glob


def load_data():
    """Загрузка данных из файлов"""
    hh_files = glob.glob("../data/hh_api_data_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    if not hh_files or not avito_files:
        raise FileNotFoundError("Не найдены файлы с данными")

    latest_hh = max(hh_files)
    latest_avito = max(avito_files)

    with open(latest_hh, "r", encoding="utf-8") as f:
        hh_data = json.load(f)
    with open(latest_avito, "r", encoding="utf-8") as f:
        avito_data = json.load(f)

    return hh_data, avito_data


def prepare_dataframe(data):
    """Подготовка DataFrame из данных"""
    df = pd.DataFrame(data)
    df["salary_from"] = df["salary"].apply(lambda x: x.get("from") if x else None)
    df["salary_to"] = df["salary"].apply(lambda x: x.get("to") if x else None)
    df["avg_salary"] = df.apply(
        lambda x: (
            (x["salary_from"] + x["salary_to"]) / 2
            if x["salary_from"] and x["salary_to"]
            else x["salary_from"] or x["salary_to"]
        ),
        axis=1,
    )
    return df


def analyze_missing_values(df, source):
    """Анализ пропущенных значений"""
    missing_stats = df.isnull().sum()
    missing_percentages = (missing_stats / len(df)) * 100

    print(f"\nАнализ пропущенных значений для {source}:")
    print("\nКоличество пропусков:")
    print(missing_stats)
    print("\nПроцент пропусков:")
    print(missing_percentages)

    # Визуализация пропусков
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    plt.title(f"Карта пропущенных значений ({source})")
    plt.tight_layout()
    plt.savefig(f"../analysis/missing_values_{source.lower()}.png")
    plt.close()


def analyze_outliers(df, source):
    """Анализ выбросов в зарплатах"""
    # Метод межквартильного размаха
    Q1 = df["avg_salary"].quantile(0.25)
    Q3 = df["avg_salary"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df["avg_salary"] < lower_bound) | (df["avg_salary"] > upper_bound)]

    print(f"\nАнализ выбросов для {source}:")
    print(f"Количество выбросов: {len(outliers)}")
    print(f"Процент выбросов: {(len(outliers) / len(df)) * 100:.2f}%")

    # Визуализация выбросов
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["avg_salary"])
    plt.title(f"Распределение зарплат с выбросами ({source})")
    plt.tight_layout()
    plt.savefig(f"../analysis/salary_outliers_{source.lower()}.png")
    plt.close()


def handle_missing_values(df, method="mean"):
    """Обработка пропущенных значений"""
    if method == "mean":
        df["avg_salary"] = df["avg_salary"].fillna(df["avg_salary"].mean())
    elif method == "median":
        df["avg_salary"] = df["avg_salary"].fillna(df["avg_salary"].median())
    elif method == "drop":
        df = df.dropna(subset=["avg_salary"])
    return df


def handle_outliers(df, method="iqr"):
    """Обработка выбросов"""
    if method == "iqr":
        Q1 = df["avg_salary"].quantile(0.25)
        Q3 = df["avg_salary"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df["avg_salary"] >= lower_bound) & (df["avg_salary"] <= upper_bound)]
    elif method == "zscore":
        z_scores = np.abs(
            (df["avg_salary"] - df["avg_salary"].mean()) / df["avg_salary"].std()
        )
        df = df[z_scores < 3]
    return df


def process_categorical_features(df):
    """Обработка категориальных признаков"""
    # Кодирование профессий
    df["profession_encoded"] = pd.factorize(df["search_query"])[0]

    # Кодирование регионов
    df["region_encoded"] = pd.factorize(df["region"])[0]

    return df


def main():
    # Создаем директорию для результатов анализа
    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    # Загружаем данные
    hh_data, avito_data = load_data()

    # Подготавливаем DataFrame
    hh_df = prepare_dataframe(hh_data)
    avito_df = prepare_dataframe(avito_data)

    # Анализ пропущенных значений
    analyze_missing_values(hh_df, "HeadHunter")
    analyze_missing_values(avito_df, "Авито")

    # Анализ выбросов
    analyze_outliers(hh_df, "HeadHunter")
    analyze_outliers(avito_df, "Авито")

    # Обработка пропущенных значений
    hh_df_cleaned = handle_missing_values(hh_df.copy(), method="mean")
    avito_df_cleaned = handle_missing_values(avito_df.copy(), method="mean")

    # Обработка выбросов
    hh_df_cleaned = handle_outliers(hh_df_cleaned, method="iqr")
    avito_df_cleaned = handle_outliers(avito_df_cleaned, method="iqr")

    # Обработка категориальных признаков
    hh_df_cleaned = process_categorical_features(hh_df_cleaned)
    avito_df_cleaned = process_categorical_features(avito_df_cleaned)

    # Сохраняем очищенные данные
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hh_df_cleaned.to_csv(f"../data/cleaned_hh_data_{timestamp}.csv", index=False)
    avito_df_cleaned.to_csv(f"../data/cleaned_avito_data_{timestamp}.csv", index=False)

    # Сохраняем статистику очистки
    cleaning_stats = {
        "HeadHunter": {
            "Исходное количество строк": len(hh_df),
            "Количество строк после очистки": len(hh_df_cleaned),
            "Удалено строк": len(hh_df) - len(hh_df_cleaned),
            "Процент удаленных строк": ((len(hh_df) - len(hh_df_cleaned)) / len(hh_df))
            * 100,
        },
        "Авито": {
            "Исходное количество строк": len(avito_df),
            "Количество строк после очистки": len(avito_df_cleaned),
            "Удалено строк": len(avito_df) - len(avito_df_cleaned),
            "Процент удаленных строк": (
                (len(avito_df) - len(avito_df_cleaned)) / len(avito_df)
            )
            * 100,
        },
    }

    with open(
        f"../analysis/cleaning_stats_{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(cleaning_stats, f, ensure_ascii=False, indent=2)

    print("\nОчистка данных завершена. Результаты сохранены в папке 'analysis'")


if __name__ == "__main__":
    main()
