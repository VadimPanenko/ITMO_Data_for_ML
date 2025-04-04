import requests
import json
import time
import os
from datetime import datetime


def get_vacancies(search_queries, per_page=100, max_pages=10):
    base_url = "https://api.hh.ru/vacancies"
    all_vacancies = []

    for query in search_queries:
        print(f"Запрос: {query}")

        for page in range(max_pages):
            params = {
                "text": query,
                "per_page": per_page,
                "page": page,
                "area": 113,  
                "search_field": "name",
            }

            headers = {"User-Agent": "JobSalaryAnalyzer/1.0 (alexsmith@example.com)"}

            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                if "items" not in data or not data["items"]:
                    print(
                        f"Нет результатов или достигнут конец результатов для {query} на странице {page}"
                    )
                    break

                vacancies = data["items"]
                print(
                    f"Получено {len(vacancies)} вакансий для {query} на странице {page}"
                )

                for vacancy in vacancies:
                    vacancy_full = get_vacancy_details(vacancy["id"], headers)
                    if vacancy_full:
                        vacancy_full["search_query"] = query
                        all_vacancies.append(vacancy_full)

                print(f"Всего собрано {len(all_vacancies)} вакансий")

                if len(vacancies) < per_page:
                    print(f"Достигнут конец результатов для {query}")
                    break

                time.sleep(0.25)  

            except requests.exceptions.RequestException as e:
                print(f"Ошибка при запросе к API: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Неожиданная ошибка: {e}")
                continue

    return all_vacancies


def get_vacancy_details(vacancy_id, headers):
    url = f"https://api.hh.ru/vacancies/{vacancy_id}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении деталей вакансии {vacancy_id}: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при обработке вакансии {vacancy_id}: {e}")
        return None


def save_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists("../data"):
        os.makedirs("../data")

    filename = f"../data/hh_api_data_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Данные сохранены в файл: {filename}")
    return filename


def main():
    search_queries = [
        "Data Scientist",
        "Data Analyst",
        "Machine Learning",
        "Data Engineer",
        "ML Engineer",
        "AI Engineer",
        "Python Developer",
        "Data Architect",
    ]

    print("Начинаем сбор данных с HeadHunter API...")
    vacancies = get_vacancies(search_queries, per_page=100, max_pages=10)
    save_data(vacancies)
    print(f"\nВсего собрано {len(vacancies)} вакансий с HeadHunter")


if __name__ == "__main__":
    main()
