import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats


def load_and_combine_data():
    output_dir = "../data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_files = glob.glob("../data/combined_dataset_*.csv")

    if combined_files:
        latest_combined = max(combined_files)
        print(f"Используем существующий объединенный датасет: {latest_combined}")
        return pd.read_csv(latest_combined, encoding="utf-8")

    hh_files = glob.glob("../data/hh_api_data_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    latest_hh = max(hh_files) if hh_files else None
    latest_avito = max(avito_files) if avito_files else None

    all_data = []

    if latest_hh:
        with open(latest_hh, "r", encoding="utf-8") as f:
            hh_data = json.load(f)

        cleaned_hh = clean_hh_data(hh_data)
        all_data.extend(cleaned_hh)
        print(f"Обработано {len(cleaned_hh)} вакансий из HeadHunter")

    if latest_avito:
        with open(latest_avito, "r", encoding="utf-8") as f:
            avito_data = json.load(f)

        cleaned_avito = clean_avito_data(avito_data)
        all_data.extend(cleaned_avito)
        print(f"Обработано {len(cleaned_avito)} вакансий из Avito")

    df = pd.DataFrame(all_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/combined_dataset_{timestamp}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Объединенный датасет сохранен в {output_file}")
    print(f"Общее количество вакансий: {len(df)}")

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

        if "search_query" in item:
            cleaned_item["search_query"] = item["search_query"]

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

        if "search_query" in item:
            cleaned_item["search_query"] = item["search_query"]

        cleaned_data.append(cleaned_item)

    return cleaned_data


def analyze_missing_values(df):
    print("\nАнализ пропущенных значений:")

    total_cells = np.product(df.shape)
    missing_values = df.isnull().sum().sum()

    print(f"Общее количество ячеек: {total_cells}")
    print(f"Пропущенные значения: {missing_values}")
    print(f"Процент пропущенных значений: {missing_values / total_cells:.2%}")

    missing_by_column = df.isnull().sum()
    missing_percentage = (missing_by_column / len(df)) * 100

    missing_df = pd.DataFrame(
        {
            "Количество пропусков": missing_by_column,
            "Процент пропусков": missing_percentage,
        }
    ).sort_values("Процент пропусков", ascending=False)

    print("\nПропуски по столбцам:")
    print(missing_df[missing_df["Количество пропусков"] > 0])

    plt.figure(figsize=(12, 6))
    missing_df = missing_df[missing_df["Количество пропусков"] > 0]
    sns.barplot(x=missing_df.index, y="Процент пропусков", data=missing_df)
    plt.xticks(rotation=90)
    plt.title("Процент пропущенных значений по столбцам")
    plt.tight_layout()

    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    plt.savefig("../analysis/missing_values_matrix.png")

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    plt.title("Тепловая карта пропущенных значений")
    plt.tight_layout()
    plt.savefig("../analysis/missing_values_heatmap.png")

    missing_by_source = (
        df.groupby("source")["avg_salary"]
        .apply(lambda x: x.isnull().mean() * 100)
        .reset_index()
    )
    missing_by_source.columns = ["Источник", "Процент пропусков зарплаты"]

    print("\nПропуски зарплаты по источникам:")
    print(missing_by_source)

    missing_by_search_query = (
        df.groupby("search_query")["avg_salary"]
        .apply(lambda x: x.isnull().mean() * 100)
        .reset_index()
    )
    missing_by_search_query.columns = ["Профессия", "Процент пропусков зарплаты"]
    missing_by_search_query = missing_by_search_query.sort_values(
        "Процент пропусков зарплаты", ascending=False
    )

    print("\nПропуски зарплаты по профессиям:")
    print(missing_by_search_query)

    return missing_df, missing_by_source, missing_by_search_query


def analyze_outliers(df):
    print("\nАнализ выбросов в зарплатах:")

    salary_data = df[df["avg_salary"].notna()].copy()

    z_scores = np.abs(stats.zscore(salary_data["avg_salary"]))
    outliers_z = np.sum(z_scores > 3)

    Q1 = salary_data["avg_salary"].quantile(0.25)
    Q3 = salary_data["avg_salary"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = np.sum(
        (salary_data["avg_salary"] < lower_bound)
        | (salary_data["avg_salary"] > upper_bound)
    )

    print(f"Количество выбросов (Z-score > 3): {outliers_z}")
    print(f"Процент выбросов (Z-score): {outliers_z / len(salary_data):.2%}")

    print(f"Количество выбросов (IQR): {outliers_iqr}")
    print(f"Процент выбросов (IQR): {outliers_iqr / len(salary_data):.2%}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=salary_data["avg_salary"])
    plt.title("Диаграмма размаха зарплат")

    plt.subplot(1, 2, 2)
    sns.histplot(salary_data["avg_salary"], kde=True)
    plt.axvline(x=lower_bound, color="r", linestyle="--")
    plt.axvline(x=upper_bound, color="r", linestyle="--")
    plt.title("Распределение зарплат с границами выбросов")

    plt.tight_layout()
    plt.savefig("../analysis/salary_outliers_by_source.png")

    plt.figure(figsize=(16, 8))
    sns.boxplot(x="search_query", y="avg_salary", data=salary_data)
    plt.title("Диаграмма размаха зарплат по профессиям")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../analysis/salary_outliers_by_profession.png")

    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    salary_data["outlier"] = isolation_forest.fit_predict(salary_data[["avg_salary"]])
    outliers_if = np.sum(salary_data["outlier"] == -1)

    print(f"Количество выбросов (Isolation Forest): {outliers_if}")
    print(f"Процент выбросов (Isolation Forest): {outliers_if / len(salary_data):.2%}")

    outliers_by_method = {
        "Z-score": outliers_z / len(salary_data) * 100,
        "IQR": outliers_iqr / len(salary_data) * 100,
        "Isolation Forest": outliers_if / len(salary_data) * 100,
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(outliers_by_method.keys()), y=list(outliers_by_method.values()))
    plt.title("Процент выбросов, обнаруженных разными методами")
    plt.ylabel("Процент выбросов")
    plt.tight_layout()

    method_comparison = pd.DataFrame(
        {
            "Метод": list(outliers_by_method.keys()),
            "Процент выбросов": list(outliers_by_method.values()),
        }
    )

    method_comparison.to_csv("../analysis/method_comparison.csv", index=False)
    plt.savefig("../analysis/methods_comparison.png")

    return outliers_by_method, method_comparison


def handle_missing_values(df, method="simple"):
    print(f"\nОбработка пропущенных значений методом: {method}")

    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    if method == "simple":
        for col in numeric_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)

    elif method == "group_mean":
        for col in numeric_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col] = df_copy.groupby("search_query")[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)

    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

    elif method == "iterative":
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

    for col in df_copy.select_dtypes(include=["object"]).columns:
        df_copy[col].fillna("Unknown", inplace=True)

    print(
        f"Процент пропущенных значений после обработки: {df_copy.isnull().sum().sum() / np.product(df_copy.shape):.2%}"
    )

    return df_copy


def handle_outliers(df, method="iqr"):
    print(f"\nОбработка выбросов методом: {method}")

    df_copy = df.copy()

    if method == "none":
        return df_copy

    salary_data = df_copy[df_copy["avg_salary"].notna()].copy()

    if method == "iqr":
        Q1 = salary_data["avg_salary"].quantile(0.25)
        Q3 = salary_data["avg_salary"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
            (df_copy["avg_salary"] < lower_bound)
            | (df_copy["avg_salary"] > upper_bound)
        ) & df_copy["avg_salary"].notna()

        print(f"Обнаружено {outliers.sum()} выбросов методом IQR")

        df_copy.loc[df_copy["avg_salary"] < lower_bound, "avg_salary"] = lower_bound
        df_copy.loc[df_copy["avg_salary"] > upper_bound, "avg_salary"] = upper_bound

    elif method == "z_score":
        z_scores = np.abs(stats.zscore(salary_data["avg_salary"]))
        outliers = (z_scores > 3) & df_copy["avg_salary"].notna()

        print(f"Обнаружено {outliers.sum()} выбросов методом Z-score")

        df_copy.loc[outliers, "avg_salary"] = df_copy.groupby("search_query")[
            "avg_salary"
        ].transform("median")

    elif method == "isolation_forest":
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        salary_data["outlier"] = isolation_forest.fit_predict(
            salary_data[["avg_salary"]]
        )

        outliers_mask = salary_data["outlier"] == -1
        outliers_indices = salary_data[outliers_mask].index

        print(f"Обнаружено {len(outliers_indices)} выбросов методом Isolation Forest")

        df_copy.loc[outliers_indices, "avg_salary"] = df_copy.groupby("search_query")[
            "avg_salary"
        ].transform("median")

    return df_copy


def compare_methods(df):
    print("\nСравнение методов обработки данных:")

    methods_missing = {
        "simple": handle_missing_values(df, "simple"),
        "group_mean": handle_missing_values(df, "group_mean"),
        "knn": handle_missing_values(df, "knn"),
    }

    methods_outliers = {
        "none": handle_outliers(df, "none"),
        "iqr": handle_outliers(df, "iqr"),
        "z_score": handle_outliers(df, "z_score"),
    }

    all_combinations = {}

    for missing_method, df_missing in methods_missing.items():
        for outlier_method, df_outlier in methods_outliers.items():
            combined_name = f"{missing_method}_{outlier_method}"

            combined_df = handle_missing_values(df, missing_method)
            combined_df = handle_outliers(combined_df, outlier_method)

            all_combinations[combined_name] = {
                "df": combined_df,
                "missing_values": combined_df.isnull().sum().sum(),
                "avg_salary_mean": combined_df["avg_salary"].mean(),
                "avg_salary_median": combined_df["avg_salary"].median(),
                "avg_salary_std": combined_df["avg_salary"].std(),
            }

    results_df = pd.DataFrame(
        {
            "Метод": list(all_combinations.keys()),
            "Пропущенные значения": [
                all_combinations[k]["missing_values"] for k in all_combinations
            ],
            "Средняя зарплата": [
                all_combinations[k]["avg_salary_mean"] for k in all_combinations
            ],
            "Медианная зарплата": [
                all_combinations[k]["avg_salary_median"] for k in all_combinations
            ],
            "Стандартное отклонение": [
                all_combinations[k]["avg_salary_std"] for k in all_combinations
            ],
        }
    )

    print("\nРезультаты сравнения методов:")
    print(results_df)

    return results_df, all_combinations


def generate_recommendations(df, missing_analysis, outlier_analysis, method_comparison):
    print("\nГенерация рекомендаций по обработке данных:")

    recommendations = []

    missing_salary_pct = df["avg_salary"].isnull().mean() * 100

    recommendations.append(
        f"1. Пропущенные значения зарплаты ({missing_salary_pct:.1f}%):"
    )

    missing_by_source = (
        df.groupby("source")["avg_salary"]
        .apply(lambda x: x.isnull().mean() * 100)
        .reset_index()
    )

    if missing_salary_pct > 50:
        recommendations.append(
            "   - Высокий процент пропущенных значений зарплаты может означать, что данные не репрезентативны."
        )
        recommendations.append(
            "   - Рекомендуется использовать метод групповых средних (group_mean) для заполнения пропусков."
        )
        recommendations.append(
            "   - Возможна дополнительная сегментация по типу занятости, региону и т.д. для более точных оценок."
        )
    else:
        recommendations.append("   - Умеренный процент пропущенных значений зарплаты.")
        recommendations.append(
            "   - Рекомендуется использовать KNN или метод групповых средних для заполнения пропусков."
        )

    recommendations.append(f"\n2. Выбросы в данных:")

    outlier_methods = outlier_analysis[0]

    if outlier_methods["IQR"] > 10:
        recommendations.append(
            "   - Значительное количество выбросов в данных о зарплатах."
        )
        recommendations.append(
            "   - Рекомендуется использовать метод IQR для обработки выбросов."
        )
        recommendations.append(
            "   - Альтернатива: винзоризация (ограничение экстремальных значений) с использованием IQR."
        )
    else:
        recommendations.append(
            "   - Умеренное количество выбросов в данных о зарплатах."
        )
        recommendations.append(
            "   - Рекомендуется анализировать выбросы в контексте конкретных профессий."
        )
        recommendations.append(
            "   - Для общего анализа можно использовать метод Z-score с порогом 3."
        )

    recommendations.append(f"\n3. Оптимальная комбинация методов:")

    best_combination = "group_mean_iqr"

    recommendations.append(
        f"   - Рекомендуемый метод обработки пропущенных значений: групповое среднее по профессии (group_mean)"
    )
    recommendations.append(
        f"   - Рекомендуемый метод обработки выбросов: метод межквартильного размаха (IQR)"
    )
    recommendations.append(
        f"   - Эта комбинация сохраняет особенности распределения данных и учитывает специфику разных профессий."
    )

    recommendations_text = "\n".join(recommendations)

    with open(
        "../analysis/data_processing_recommendations.txt", "w", encoding="utf-8"
    ) as f:
        f.write(recommendations_text)

    print(
        "\nРекомендации сохранены в файл: ../analysis/data_processing_recommendations.txt"
    )

    return recommendations_text


def main():
    print("Запуск расширенного анализа данных...")

    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    df = load_and_combine_data()

    print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

    missing_analysis = analyze_missing_values(df)
    outlier_analysis = analyze_outliers(df)

    methods_comparison, all_combinations = compare_methods(df)

    recommendations = generate_recommendations(
        df, missing_analysis, outlier_analysis, methods_comparison
    )

    print("\nАнализ завершен! Результаты сохранены в директории 'analysis'.")


if __name__ == "__main__":
    main()
