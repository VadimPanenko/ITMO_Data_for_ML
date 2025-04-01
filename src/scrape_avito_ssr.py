import requests
import json
from datetime import datetime
import os
import time
import random
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re


def extract_salary(salary_text):
    if not salary_text:
        return None

    # Удаляем пробелы и символы валюты
    salary_text = salary_text.replace(" ", "").replace("₽", "").replace("руб.", "")

    # Ищем числа в строке
    numbers = re.findall(r"\d+", salary_text)
    numbers = [int(num) for num in numbers]

    if "от" in salary_text.lower():
        return {"from": numbers[0], "to": None, "currency": "RUR"}
    elif "до" in salary_text.lower():
        return {"from": None, "to": numbers[0], "currency": "RUR"}
    elif len(numbers) >= 2:
        return {"from": numbers[0], "to": numbers[1], "currency": "RUR"}
    elif len(numbers) == 1:
        return {"from": numbers[0], "to": numbers[0], "currency": "RUR"}

    return None


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
            url = f"https://www.avito.ru/sankt-peterburg/vakansii?q={query}&s=104&p={page}"

            retries = 0
            while retries < max_retries:
                try:
                    print(
                        f"Отправляем запрос к Авито (запрос: {query}, страница {page})..."
                    )

                    # Заголовки запроса
                    headers = {
                        "User-Agent": ua.random,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Cache-Control": "max-age=0",
                    }

                    response = requests.get(url, headers=headers)
                    response.raise_for_status()

                    # Парсим HTML
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Находим все вакансии на странице
                    vacancy_items = soup.find_all("div", {"data-marker": "item"})

                    if not vacancy_items:
                        print("Вакансии не найдены на странице")
                        break

                    for item in vacancy_items:
                        # Извлекаем ID вакансии
                        item_id = item.get("data-item-id")
                        if not item_id or any(
                            v["id"] == item_id for v in all_vacancies
                        ):
                            continue

                        # Извлекаем заголовок
                        title_elem = item.find("h3", {"itemprop": "name"})
                        if not title_elem:
                            continue
                        title = title_elem.text.strip()

                        # Извлекаем зарплату
                        salary_elem = item.find("meta", {"itemprop": "price"})
                        salary_text = (
                            salary_elem.get("content") if salary_elem else None
                        )
                        salary = extract_salary(salary_text)

                        if not salary:
                            continue

                        # Извлекаем компанию
                        company_elem = item.find(
                            "div", {"class": "style-company-title-_cIhI"}
                        )
                        company = company_elem.text.strip() if company_elem else None

                        # Формируем URL вакансии
                        vacancy_url = "https://www.avito.ru" + item.find(
                            "a", {"itemprop": "url"}
                        ).get("href")

                        # Добавляем вакансию в список
                        all_vacancies.append(
                            {
                                "id": item_id,
                                "title": title,
                                "salary": salary,
                                "company": company,
                                "url": vacancy_url,
                                "published_at": datetime.now().strftime("%Y-%m-%d"),
                                "region": "Санкт-Петербург",
                                "search_query": query,
                                "source": "avito",
                            }
                        )

                    print(f"Собрано вакансий: {len(all_vacancies)}")

                    # Проверяем, есть ли следующая страница
                    next_page = soup.find(
                        "a", {"data-marker": "pagination-button/next"}
                    )
                    if not next_page:
                        break

                    page += 1
                    time.sleep(
                        random.uniform(2, 4)
                    )  # Увеличенная задержка между запросами
                    break

                except requests.exceptions.RequestException as e:
                    retries += 1
                    print(f"Ошибка при запросе (попытка {retries}): {e}")
                    if retries == max_retries:
                        print("Достигнуто максимальное количество попыток")
                        break
                    time.sleep(random.uniform(3, 6))

            if not vacancy_items or not next_page:
                break

    return all_vacancies


def save_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("../data"):
        os.makedirs("../data")

    filename = f"../data/avito_api_data_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Данные сохранены в файл: {filename}")


def main():
    print("Начинаем сбор данных с Авито...")
    vacancies = get_vacancies()
    save_data(vacancies)
    print(f"\nВсего собрано {len(vacancies)} уникальных вакансий с Авито")


if __name__ == "__main__":
    main()
