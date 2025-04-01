import requests
import json
from datetime import datetime
import os
import time
import random
from fake_useragent import UserAgent


def get_vacancies():
    all_vacancies = []
    page = 1
    max_retries = 3

    # Поисковые запросы
    search_queries = [
        "электрослесарь",
        "электромонтер",
        "электромонтажник",
        "слесарь-электрик",
    ]

    # Создаем объект для генерации случайных User-Agent
    ua = UserAgent()

    for query in search_queries:
        page = 1
        while True:
            # Формируем URL для поиска
            url = "https://api.hh.ru/vacancies"
            params = {
                "text": query,
                "area": 2,  # Санкт-Петербург
                "page": page,
                "per_page": 100,
                "only_with_salary": True,
            }

            retries = 0
            while retries < max_retries:
                try:
                    print(
                        f"Отправляем запрос к HeadHunter (запрос: {query}, страница {page})..."
                    )

                    # Заголовки запроса
                    headers = {
                        "User-Agent": ua.random,
                        "Accept": "application/json",
                        "Authorization": "Bearer YOUR_API_KEY",  # Замените на ваш API ключ
                    }

                    response = requests.get(url, params=params, headers=headers)
                    response.raise_for_status()

                    data = response.json()
                    vacancies = data.get("items", [])

                    if not vacancies:
                        print("Вакансии не найдены на странице")
                        break

                    for vacancy in vacancies:
                        # Проверяем, не дублируется ли вакансия
                        if any(v["id"] == vacancy["id"] for v in all_vacancies):
                            continue

                        # Извлекаем нужные данные
                        vacancy_data = {
                            "id": vacancy["id"],
                            "title": vacancy["name"],
                            "salary": vacancy["salary"],
                            "company": vacancy["employer"]["name"],
                            "url": vacancy["alternate_url"],
                            "published_at": vacancy["published_at"],
                            "region": vacancy["area"]["name"],
                            "search_query": query,
                            "source": "hh",
                        }

                        all_vacancies.append(vacancy_data)

                    print(f"Собрано вакансий: {len(all_vacancies)}")

                    # Проверяем, есть ли следующая страница
                    if page >= data.get("pages", 1):
                        break

                    page += 1
                    time.sleep(random.uniform(1, 2))  # Задержка между запросами
                    break

                except requests.exceptions.RequestException as e:
                    retries += 1
                    print(f"Ошибка при запросе (попытка {retries}): {e}")
                    if retries == max_retries:
                        print("Достигнуто максимальное количество попыток")
                        break
                    time.sleep(random.uniform(2, 4))

            if not vacancies or page >= data.get("pages", 1):
                break

    return all_vacancies


def save_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("../data"):
        os.makedirs("../data")

    filename = f"../data/hh_api_data_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Данные сохранены в файл: {filename}")


def main():
    print("Начинаем сбор данных с HeadHunter...")
    vacancies = get_vacancies()
    save_data(vacancies)
    print(f"\nВсего собрано {len(vacancies)} уникальных вакансий с HeadHunter")


if __name__ == "__main__":
    main()
