import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import glob
import re


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

    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    plt.title(f"Карта пропущенных значений ({source})")
    plt.tight_layout()
    plt.savefig(f"../analysis/missing_values_{source.lower()}.png")
    plt.close()


def analyze_outliers(df, source):
    """Анализ выбросов в зарплатах"""
    
    Q1 = df["avg_salary"].quantile(0.25)
    Q3 = df["avg_salary"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df["avg_salary"] < lower_bound) | (df["avg_salary"] > upper_bound)]

    print(f"\nАнализ выбросов для {source}:")
    print(f"Количество выбросов: {len(outliers)}")
    print(f"Процент выбросов: {(len(outliers) / len(df)) * 100:.2f}%")

    
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
    
    df["profession_encoded"] = pd.factorize(df["search_query"])[0]

    
    df["region_encoded"] = pd.factorize(df["region"])[0]

    return df


def clean_hh_data(hh_data):
    cleaned_data = []

    for item in hh_data:
        if not isinstance(item, dict):
            continue

        cleaned_item = {
            "id": item.get("id"),
            "name": item.get("name"),
            "company": (
                item.get("employer", {}).get("name") if item.get("employer") else None
            ),
            "url": item.get("alternate_url"),
            "area": item.get("area", {}).get("name") if item.get("area") else None,
            "salary": item.get("salary", {}),
            "published_at": item.get("published_at"),
            "requirement": (
                item.get("snippet", {}).get("requirement")
                if item.get("snippet")
                else None
            ),
            "responsibility": (
                item.get("snippet", {}).get("responsibility")
                if item.get("snippet")
                else None
            ),
            "experience": (
                item.get("experience", {}).get("name")
                if item.get("experience")
                else None
            ),
            "schedule": (
                item.get("schedule", {}).get("name") if item.get("schedule") else None
            ),
            "employment": (
                item.get("employment", {}).get("name")
                if item.get("employment")
                else None
            ),
            "description": item.get("description"),
            "skills": (
                [skill.get("name") for skill in item.get("key_skills", [])]
                if item.get("key_skills")
                else []
            ),
            "region": item.get("area", {}).get("name") if item.get("area") else None,
            "source": "hh",
        }

        if cleaned_item["salary"]:
            from_salary = cleaned_item["salary"].get("from")
            to_salary = cleaned_item["salary"].get("to")

            if from_salary is not None and to_salary is not None:
                cleaned_item["avg_salary"] = (from_salary + to_salary) / 2
                cleaned_item["salary_from"] = from_salary
                cleaned_item["salary_to"] = to_salary
                cleaned_item["salary_currency"] = cleaned_item["salary"].get("currency")
            elif from_salary is not None:
                cleaned_item["avg_salary"] = from_salary
                cleaned_item["salary_from"] = from_salary
                cleaned_item["salary_to"] = None
                cleaned_item["salary_currency"] = cleaned_item["salary"].get("currency")
            elif to_salary is not None:
                cleaned_item["avg_salary"] = to_salary
                cleaned_item["salary_from"] = None
                cleaned_item["salary_to"] = to_salary
                cleaned_item["salary_currency"] = cleaned_item["salary"].get("currency")

        cleaned_data.append(cleaned_item)

    return cleaned_data


def clean_avito_data(avito_data):
    cleaned_data = []

    for item in avito_data:
        if not isinstance(item, dict):
            continue

        cleaned_item = {
            "id": item.get("id"),
            "name": item.get("title"),
            "company": item.get("company_name"),
            "url": item.get("url"),
            "area": item.get("location"),
            "salary": item.get("salary_data", {}),
            "published_at": item.get("date"),
            "requirement": item.get("requirements"),
            "responsibility": item.get("responsibilities"),
            "experience": item.get("experience"),
            "schedule": item.get("schedule"),
            "employment": item.get("employment_type"),
            "description": item.get("description"),
            "skills": item.get("skills", []),
            "region": item.get("location"),
            "source": "avito",
        }

        if cleaned_item["salary"]:
            from_salary = cleaned_item["salary"].get("from")
            to_salary = cleaned_item["salary"].get("to")

            if from_salary is not None and to_salary is not None:
                cleaned_item["avg_salary"] = (from_salary + to_salary) / 2
                cleaned_item["salary_from"] = from_salary
                cleaned_item["salary_to"] = to_salary
                cleaned_item["salary_currency"] = cleaned_item["salary"].get(
                    "currency", "RUB"
                )
            elif from_salary is not None:
                cleaned_item["avg_salary"] = from_salary
                cleaned_item["salary_from"] = from_salary
                cleaned_item["salary_to"] = None
                cleaned_item["salary_currency"] = cleaned_item["salary"].get(
                    "currency", "RUB"
                )
            elif to_salary is not None:
                cleaned_item["avg_salary"] = to_salary
                cleaned_item["salary_from"] = None
                cleaned_item["salary_to"] = to_salary
                cleaned_item["salary_currency"] = cleaned_item["salary"].get(
                    "currency", "RUB"
                )

        cleaned_data.append(cleaned_item)

    return cleaned_data


def clean_and_combine_data():
    output_dir = "../data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hh_files = glob.glob("../data/hh_api_data_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    latest_hh = max(hh_files) if hh_files else None
    latest_avito = max(avito_files) if avito_files else None

    all_data = []

    if latest_hh:
        with open(latest_hh, "r", encoding="utf-8") as f:
            hh_data = json.load(f)

        for i, item in enumerate(hh_data):
            if isinstance(item, dict) and "search_query" not in item:
                search_query = "unknown"
                if (
                    "alternate_url" in item
                    and "specialization" in item["alternate_url"]
                ):
                    match = re.search(r"specialization/([^/]+)", item["alternate_url"])
                    if match:
                        search_query = match.group(1)
                hh_data[i]["search_query"] = search_query

        cleaned_hh = clean_hh_data(hh_data)
        all_data.extend(cleaned_hh)
        print(f"Обработано {len(cleaned_hh)} вакансий из HeadHunter")

    if latest_avito:
        with open(latest_avito, "r", encoding="utf-8") as f:
            avito_data = json.load(f)

        for i, item in enumerate(avito_data):
            if isinstance(item, dict) and "search_query" not in item:
                avito_data[i]["search_query"] = "unknown"

        cleaned_avito = clean_avito_data(avito_data)
        all_data.extend(cleaned_avito)
        print(f"Обработано {len(cleaned_avito)} вакансий из Avito")

    df = pd.DataFrame(all_data)
    output_file = f"{output_dir}/combined_dataset_{timestamp}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Объединенный датасет сохранен в {output_file}")
    print(f"Общее количество вакансий: {len(df)}")

    return df, output_file


def main():
    
    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    
    hh_data, avito_data = load_data()

    
    hh_df = prepare_dataframe(hh_data)
    avito_df = prepare_dataframe(avito_data)

    
    analyze_missing_values(hh_df, "HeadHunter")
    analyze_missing_values(avito_df, "Авито")

    
    analyze_outliers(hh_df, "HeadHunter")
    analyze_outliers(avito_df, "Авито")

    
    hh_df_cleaned = handle_missing_values(hh_df.copy(), method="mean")
    avito_df_cleaned = handle_missing_values(avito_df.copy(), method="mean")

    
    hh_df_cleaned = handle_outliers(hh_df_cleaned, method="iqr")
    avito_df_cleaned = handle_outliers(avito_df_cleaned, method="iqr")

    
    hh_df_cleaned = process_categorical_features(hh_df_cleaned)
    avito_df_cleaned = process_categorical_features(avito_df_cleaned)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hh_df_cleaned.to_csv(f"../data/cleaned_hh_data_{timestamp}.csv", index=False)
    avito_df_cleaned.to_csv(f"../data/cleaned_avito_data_{timestamp}.csv", index=False)

    
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
