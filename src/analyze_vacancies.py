import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import glob
import time
import shutil


def load_data():
    
    hh_files_trimmed = glob.glob("../data/hh_api_data_trimmed_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    
    if not hh_files_trimmed:
        hh_files = glob.glob("../data/hh_api_data_*.json")
        
        hh_files = [f for f in hh_files if "trimmed" not in f]
    else:
        hh_files = hh_files_trimmed

    if not hh_files or not avito_files:
        raise FileNotFoundError(
            "Не найдены файлы с данными от одного или обоих источников"
        )

    latest_hh = max(hh_files)
    latest_avito = max(avito_files)

    print(f"Анализируем файлы:\n- HeadHunter: {latest_hh}\n- Авито: {latest_avito}")

    with open(latest_hh, "r", encoding="utf-8") as f:
        hh_data = json.load(f)
    with open(latest_avito, "r", encoding="utf-8") as f:
        avito_data = json.load(f)

    return hh_data, avito_data


def prepare_salary_data(vacancies):
    salary_data = []
    for vacancy in vacancies:
        salary = vacancy.get("salary", {})
        if salary:
            salary_from = salary.get("from")
            salary_to = salary.get("to")

            if salary_from and salary_to:
                avg_salary = (salary_from + salary_to) / 2
            elif salary_from:
                avg_salary = salary_from
            elif salary_to:
                avg_salary = salary_to
            else:
                continue

            salary_data.append(
                {
                    "title": vacancy["title"],
                    "company": vacancy["company"],
                    "region": vacancy["region"],
                    "salary": avg_salary,
                    "search_query": vacancy["search_query"],
                    "source": vacancy["source"],
                }
            )

    return pd.DataFrame(salary_data)


def analyze_vacancies(hh_df, avito_df):
    
    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    
    for file in glob.glob("../analysis/*.png"):
        os.remove(file)
    if os.path.exists("../analysis/statistics.json"):
        os.remove("../analysis/statistics.json")

    plt.style.use("seaborn")
    plt.rcParams["figure.figsize"] = (12, 6)

    
    combined_df = pd.concat([hh_df, avito_df])

    
    plt.figure(figsize=(12, 6))

    
    avg_salary_by_query = (
        combined_df.groupby(["search_query", "source"])["salary"]
        .mean()
        .round(0)
        .unstack()
    )

    
    ax = avg_salary_by_query.plot(kind="bar", width=0.8)
    plt.title("Средняя зарплата по профессиям и источникам")
    plt.xlabel("Профессия")
    plt.ylabel("Зарплата (руб.)")
    plt.legend(title="Источник")
    plt.xticks(rotation=45, ha="right")

    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f ₽", padding=3)

    plt.tight_layout()
    plt.savefig("../analysis/avg_salary_by_profession_and_source.png")
    plt.close()

    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=combined_df, x="salary", bins=30, kde=True)
    plt.title("Распределение зарплат")
    plt.xlabel("Зарплата (руб.)")
    plt.ylabel("Количество вакансий")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution.png")
    plt.close()

    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x="source", y="salary")
    plt.title("Распределение зарплат по источникам")
    plt.xlabel("Источник")
    plt.ylabel("Зарплата (руб.)")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution_by_source.png")
    plt.close()

    
    plt.figure(figsize=(12, 8))
    region_salary = (
        combined_df.groupby("region")["salary"]
        .mean()
        .round(0)
        .sort_values(ascending=False)
        .head(10)
    )
    region_salary.plot(kind="barh")
    plt.title("Средняя зарплата по регионам (топ-10)")
    plt.xlabel("Средняя зарплата (руб.)")
    plt.ylabel("Регион")
    for i, v in enumerate(region_salary):
        plt.text(v, i, f"{v:,.0f} ₽", va="center")
    plt.tight_layout()
    plt.savefig("../analysis/avg_salary_by_region.png")
    plt.close()

    
    plt.figure(figsize=(12, 6))
    profession_salary = (
        combined_df.groupby("search_query")["salary"]
        .mean()
        .round(0)
        .sort_values(ascending=False)
    )
    profession_salary.plot(kind="bar")
    plt.title("Средняя зарплата по профессиям")
    plt.xlabel("Профессия")
    plt.ylabel("Средняя зарплата (руб.)")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(profession_salary):
        plt.text(i, v, f"{v:,.0f} ₽", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("../analysis/avg_salary_by_profession.png")
    plt.close()

    
    plt.figure(figsize=(12, 8))
    
    df_companies = combined_df[combined_df["company"].notna()]
    top_companies = df_companies["company"].value_counts().head(10)
    plt.barh(top_companies.index, top_companies.values)
    plt.title("Топ-10 компаний по количеству вакансий")
    plt.xlabel("Количество вакансий")
    plt.ylabel("Компания")
    for i, v in enumerate(top_companies.values):
        plt.text(v, i, str(v), va="center")
    plt.tight_layout()
    plt.savefig("../analysis/top_companies.png")
    plt.close()

    
    stats = {
        "Общая статистика": {
            "Всего вакансий с указанной зарплатой": len(combined_df),
            "Средняя зарплата": int(combined_df["salary"].mean()),
            "Медианная зарплата": int(combined_df["salary"].median()),
            "Минимальная зарплата": int(combined_df["salary"].min()),
            "Максимальная зарплата": int(combined_df["salary"].max()),
        },
        "Статистика по источникам": {
            "HeadHunter": {
                "Количество вакансий": len(hh_df),
                "Средняя зарплата": int(hh_df["salary"].mean()),
                "Медианная зарплата": int(hh_df["salary"].median()),
            },
            "Авито": {
                "Количество вакансий": len(avito_df),
                "Средняя зарплата": int(avito_df["salary"].mean()),
                "Медианная зарплата": int(avito_df["salary"].median()),
            },
        },
        "Количество вакансий по профессиям": combined_df["search_query"]
        .value_counts()
        .to_dict(),
        "Количество вакансий по регионам": combined_df["region"]
        .value_counts()
        .to_dict(),
        "Топ-10 компаний по количеству вакансий": top_companies.to_dict(),
    }

    with open("../analysis/statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    hh_files_trimmed = glob.glob("../data/hh_api_data_trimmed_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    if not hh_files_trimmed:
        hh_files = glob.glob("../data/hh_api_data_*.json")
        hh_files = [f for f in hh_files if "trimmed" not in f]
    else:
        hh_files = hh_files_trimmed

    latest_hh = max(hh_files)
    latest_avito = max(avito_files)

    print(f"Анализируем файлы:\n- HeadHunter: {latest_hh}")

    with open(latest_hh, "r", encoding="utf-8") as f:
        hh_data = json.load(f)

    if avito_files:
        print(f"- Авито: {latest_avito}")
        with open(latest_avito, "r", encoding="utf-8") as f:
            avito_data = json.load(f)
    else:
        avito_data = []

    hh_df = pd.DataFrame(hh_data)
    for col in ["salary_from", "salary_to", "salary_currency"]:
        if col not in hh_df.columns:
            hh_df[col] = hh_df["salary"].apply(
                lambda x: x.get(col.split("_")[-1]) if x else None
            )

    if "avg_salary" not in hh_df.columns:
        hh_df["avg_salary"] = hh_df.apply(
            lambda x: (
                (x["salary_from"] + x["salary_to"]) / 2
                if x["salary_from"] and x["salary_to"]
                else x["salary_from"] or x["salary_to"]
            ),
            axis=1,
        )

    if avito_data:
        avito_df = pd.DataFrame(avito_data)
        for col in ["salary_from", "salary_to", "salary_currency"]:
            if col not in avito_df.columns:
                avito_df[col] = avito_df["salary"].apply(
                    lambda x: x.get(col.split("_")[-1]) if x else None
                )

        if "avg_salary" not in avito_df.columns:
            avito_df["avg_salary"] = avito_df.apply(
                lambda x: (
                    (x["salary_from"] + x["salary_to"]) / 2
                    if x["salary_from"] and x["salary_to"]
                    else x["salary_from"] or x["salary_to"]
                ),
                axis=1,
            )

    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    for f in glob.glob("../analysis/*.png"):
        os.remove(f)

    if os.path.exists("../analysis/statistics.json"):
        os.remove("../analysis/statistics.json")

    if avito_data:
        combined_df = pd.concat([hh_df, avito_df], ignore_index=True)
    else:
        combined_df = hh_df.copy()

    avg_salary_by_query_source = (
        combined_df.groupby(["search_query", "source"])["avg_salary"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        x="search_query", y="avg_salary", hue="source", data=avg_salary_by_query_source
    )
    plt.title("Средняя зарплата по профессиям и источникам")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig("../analysis/avg_salary_by_profession_and_source.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(combined_df["avg_salary"].dropna(), bins=30, kde=True, color="skyblue")
    plt.title("Распределение зарплат")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=combined_df, x="avg_salary", hue="source", multiple="stack", bins=30
    )
    plt.title("Распределение зарплат по источникам")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution_by_source.png")
    plt.close()

    top_regions = (
        combined_df.groupby("region")["avg_salary"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=top_regions.index, y=top_regions["mean"], palette="viridis")
    plt.title("Средняя зарплата по регионам (топ-10)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for i, v in enumerate(top_regions["mean"]):
        ax.text(i, v + 1000, f"{int(v):,} ₽", ha="center", va="bottom", fontsize=10)

    plt.savefig("../analysis/avg_salary_by_region.png")
    plt.close()

    avg_salary_by_query = (
        combined_df.groupby("search_query")["avg_salary"]
        .mean()
        .sort_values(ascending=False)
    )
    count_by_query = combined_df.groupby("search_query").size()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=avg_salary_by_query.index, y=avg_salary_by_query.values, palette="Blues_d"
    )
    plt.title("Средняя зарплата по профессиям")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for i, v in enumerate(avg_salary_by_query):
        ax.text(
            i,
            v + 1000,
            f"{int(v):,} ₽ ({count_by_query[avg_salary_by_query.index[i]]})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.savefig("../analysis/avg_salary_by_profession.png")
    plt.close()

    companies_count = combined_df["company"].value_counts().head(10)
    combined_df_no_none = combined_df.dropna(subset=["company"])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=companies_count.index, y=companies_count.values, palette="Greens_d"
    )
    plt.title("Топ-10 компаний по количеству вакансий")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("../analysis/top_companies.png")
    plt.close()

    stats = {
        "total_vacancies": len(combined_df),
        "total_with_salary": combined_df["avg_salary"].notna().sum(),
        "avg_salary": int(combined_df["avg_salary"].mean()),
        "median_salary": int(combined_df["avg_salary"].median()),
        "min_salary": int(combined_df["avg_salary"].min()),
        "max_salary": int(combined_df["avg_salary"].max()),
        "sources": {
            "hh": {
                "count": len(hh_df),
                "with_salary": hh_df["avg_salary"].notna().sum(),
                "avg_salary": int(hh_df["avg_salary"].mean()),
                "median_salary": int(hh_df["avg_salary"].median()),
            }
        },
        "professions": {},
    }

    if avito_data:
        stats["sources"]["avito"] = {
            "count": len(avito_df),
            "with_salary": avito_df["avg_salary"].notna().sum(),
            "avg_salary": int(avito_df["avg_salary"].mean()),
            "median_salary": int(avito_df["avg_salary"].median()),
        }

    for query in combined_df["search_query"].unique():
        query_df = combined_df[combined_df["search_query"] == query]
        stats["professions"][query] = {
            "count": len(query_df),
            "with_salary": query_df["avg_salary"].notna().sum(),
            "avg_salary": int(
                query_df["avg_salary"].mean()
                if not query_df["avg_salary"].isna().all()
                else 0
            ),
            "percentage": f"{len(query_df) / len(combined_df) * 100:.1f}%",
        }

    with open("../analysis/statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nОсновная статистика:")
    print(f"Всего вакансий с указанной зарплатой: {stats['total_with_salary']}")
    print(f"Средняя зарплата: {stats['avg_salary']:,} ₽")
    print(f"Медианная зарплата: {stats['median_salary']:,} ₽")
    print(f"Минимальная зарплата: {stats['min_salary']:,} ₽")
    print(f"Максимальная зарплата: {stats['max_salary']:,} ₽")

    print("\nСтатистика по источникам:\n")
    for source, source_stats in stats["sources"].items():
        print(f"{source.upper()}:")
        print(f"  Количество вакансий: {source_stats['count']}")
        print(f"  Средняя зарплата: {source_stats['avg_salary']:,} ₽")
        print(f"  Медианная зарплата: {source_stats['median_salary']:,} ₽\n")

    print("Количество вакансий по профессиям:")
    for query, query_stats in sorted(
        stats["professions"].items(), key=lambda x: x[1]["count"], reverse=True
    ):
        print(f"{query}: {query_stats['count']}")

    print("\nГрафики сохранены в папке 'analysis'")


if __name__ == "__main__":
    main()
