import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import glob


def load_data():
    # Загружаем последние файлы с данными от каждого источника
    hh_files_trimmed = glob.glob("../data/hh_api_data_trimmed_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    # Если нет trimmed файлов, используем обычные hh-файлы
    if not hh_files_trimmed:
        hh_files = glob.glob("../data/hh_api_data_*.json")
        # Исключаем trimmed файлы, если они есть
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
    # Создаем директорию для результатов анализа
    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    # Удаляем старые графики и статистику
    for file in glob.glob("../analysis/*.png"):
        os.remove(file)
    if os.path.exists("../analysis/statistics.json"):
        os.remove("../analysis/statistics.json")

    plt.style.use("seaborn")
    plt.rcParams["figure.figsize"] = (12, 6)

    # Объединяем данные для общего анализа
    combined_df = pd.concat([hh_df, avito_df])

    # 1. Средняя зарплата по профессиям и источникам
    plt.figure(figsize=(12, 6))

    # Группируем данные
    avg_salary_by_query = (
        combined_df.groupby(["search_query", "source"])["salary"]
        .mean()
        .round(0)
        .unstack()
    )

    # Создаем график
    ax = avg_salary_by_query.plot(kind="bar", width=0.8)
    plt.title("Средняя зарплата по профессиям и источникам")
    plt.xlabel("Профессия")
    plt.ylabel("Зарплата (руб.)")
    plt.legend(title="Источник")
    plt.xticks(rotation=45, ha="right")

    # Добавляем значения на столбцы
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f ₽", padding=3)

    plt.tight_layout()
    plt.savefig("../analysis/avg_salary_by_profession_and_source.png")
    plt.close()

    # 2. Общее распределение зарплат
    plt.figure(figsize=(12, 6))
    sns.histplot(data=combined_df, x="salary", bins=30, kde=True)
    plt.title("Распределение зарплат")
    plt.xlabel("Зарплата (руб.)")
    plt.ylabel("Количество вакансий")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution.png")
    plt.close()

    # 3. Распределение зарплат по источникам
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x="source", y="salary")
    plt.title("Распределение зарплат по источникам")
    plt.xlabel("Источник")
    plt.ylabel("Зарплата (руб.)")
    plt.tight_layout()
    plt.savefig("../analysis/salary_distribution_by_source.png")
    plt.close()

    # 4. Средняя зарплата по регионам (топ-10)
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

    # 5. Средняя зарплата по профессиям
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

    # 6. Топ компаний по количеству вакансий
    plt.figure(figsize=(12, 8))
    # Удаляем None из компаний
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

    # 7. Статистика в текстовом виде
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
    hh_data, avito_data = load_data()
    hh_df = prepare_salary_data(hh_data)
    avito_df = prepare_salary_data(avito_data)
    stats = analyze_vacancies(hh_df, avito_df)

    print("\nОсновная статистика:")
    for key, value in stats["Общая статистика"].items():
        print(f"{key}: {value:,} ₽" if "зарплата" in key else f"{key}: {value}")

    print("\nСтатистика по источникам:")
    for source, source_stats in stats["Статистика по источникам"].items():
        print(f"\n{source}:")
        for key, value in source_stats.items():
            print(f"  {key}: {value:,} ₽" if "зарплата" in key else f"  {key}: {value}")

    print("\nКоличество вакансий по профессиям:")
    for query, count in stats["Количество вакансий по профессиям"].items():
        print(f"{query}: {count}")

    print("\nГрафики сохранены в папке 'analysis'")


if __name__ == "__main__":
    main()
