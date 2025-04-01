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
    """Извлекает информацию о зарплате из текста"""
    if not salary_text or salary_text == "0":
        return None

    # Удаляем пробелы и символы валюты
    salary_text = salary_text.replace(" ", "").replace("₽", "").replace("руб.", "")

    # Ищем числа в строке
    numbers = re.findall(r"\d+", salary_text)
    numbers = [int(num) for num in numbers]

    if "от" in salary_text.lower() and numbers:
        return {"from": numbers[0], "to": None, "currency": "RUR"}
    elif "до" in salary_text.lower() and numbers:
        return {"from": None, "to": numbers[0], "currency": "RUR"}
    elif len(numbers) >= 2:
        return {"from": numbers[0], "to": numbers[1], "currency": "RUR"}
    elif len(numbers) == 1:
        return {"from": numbers[0], "to": numbers[0], "currency": "RUR"}

    return None


def get_vacancies(target_count=1000):
    """Собирает вакансии с Авито"""
    all_vacancies = []
    max_retries = 5

    # Поисковые запросы
    search_queries = [
        "электрослесарь",
        "электромонтер",
        "электромонтажник",
        "слесарь-электрик",
        "электрик",  # Добавлен новый запрос для увеличения количества вакансий
    ]

    # Список регионов для поиска
    regions = [
        {"name": "Москва", "url": "moskva"},
        {"name": "Санкт-Петербург", "url": "sankt-peterburg"},
        {"name": "Новосибирск", "url": "novosibirsk"},
        {"name": "Екатеринбург", "url": "ekaterinburg"},
        {"name": "Казань", "url": "kazan"},
        {"name": "Нижний Новгород", "url": "nizhniy_novgorod"},
        {"name": "Челябинск", "url": "chelyabinsk"},
        {"name": "Омск", "url": "omsk"},
        {"name": "Самара", "url": "samara"},
        {"name": "Ростов-на-Дону", "url": "rostov-na-donu"},
        {"name": "Уфа", "url": "ufa"},
        {"name": "Красноярск", "url": "krasnoyarsk"},
        {"name": "Воронеж", "url": "voronezh"},
        {"name": "Пермь", "url": "perm"},
        {"name": "Волгоград", "url": "volgograd"},
    ]

    # Создаем объект для генерации случайных User-Agent
    ua = UserAgent()

    # Перебираем регионы и поисковые запросы
    for region in regions:
        print(f"\nПоиск вакансий в регионе: {region['name']}")

        for query in search_queries:
            page = 1

            # Проверяем, достигли ли целевого количества вакансий
            if len(all_vacancies) >= target_count:
                print(f"Достигнуто целевое количество вакансий: {target_count}")
                return all_vacancies

            while True:
                # Формируем URL для поиска
                url = f"https://www.avito.ru/{region['url']}/vakansii?q={query}&s=104&p={page}"

                retries = 0
                while retries < max_retries:
                    try:
                        print(
                            f"Запрос к Авито (регион: {region['name']}, запрос: {query}, страница {page})..."
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

                        response = requests.get(url, headers=headers, timeout=30)
                        response.raise_for_status()

                        # Парсим HTML
                        soup = BeautifulSoup(response.text, "html.parser")

                        # Находим все вакансии на странице
                        vacancy_items = soup.find_all("div", {"data-marker": "item"})

                        if not vacancy_items:
                            print("Вакансии не найдены на странице")
                            break

                        for item in vacancy_items:
                            # Проверяем, достигли ли целевого количества вакансий
                            if len(all_vacancies) >= target_count:
                                return all_vacancies

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
                            # Пробуем разные варианты поиска элемента с зарплатой
                            salary_elem = item.find("meta", {"itemprop": "price"})
                            if salary_elem:
                                salary_text = salary_elem.get("content")
                            else:
                                # Альтернативный способ извлечения зарплаты
                                salary_elem = item.find("span", {"class": "price-text"})
                                if salary_elem:
                                    salary_text = salary_elem.text.strip()
                                else:
                                    # Ищем любой элемент, который может содержать зарплату
                                    for span in item.find_all("span"):
                                        if (
                                            "₽" in span.text
                                            or "руб" in span.text.lower()
                                        ):
                                            salary_text = span.text.strip()
                                            break
                                    else:
                                        salary_text = None

                            salary = extract_salary(salary_text)

                            # Если не удалось извлечь зарплату, добавляем null
                            if not salary:
                                salary = {"from": None, "to": None, "currency": None}

                            # Извлекаем компанию
                            company = None
                            # Пробуем разные селекторы для компании
                            for class_name in [
                                "style-company-title-_cIhI",
                                "style-text-",
                                "style-company-name-",
                            ]:
                                company_elem = item.find(
                                    "div", {"class": lambda x: x and class_name in x}
                                )
                                if company_elem:
                                    company = company_elem.text.strip()
                                    break

                            # Формируем URL вакансии
                            try:
                                vacancy_url_elem = item.find("a", {"itemprop": "url"})
                                if vacancy_url_elem:
                                    vacancy_url = (
                                        "https://www.avito.ru"
                                        + vacancy_url_elem.get("href")
                                    )
                                else:
                                    # Альтернативный способ получения URL
                                    vacancy_url_elem = item.find(
                                        "a", {"data-marker": "item-title"}
                                    )
                                    if vacancy_url_elem:
                                        vacancy_url = (
                                            "https://www.avito.ru"
                                            + vacancy_url_elem.get("href")
                                        )
                                    else:
                                        vacancy_url = (
                                            f"https://www.avito.ru/vacancy/{item_id}"
                                        )
                            except Exception as e:
                                print(f"Ошибка при извлечении URL: {e}")
                                vacancy_url = f"https://www.avito.ru/vacancy/{item_id}"

                            # Добавляем вакансию в список
                            all_vacancies.append(
                                {
                                    "id": item_id,
                                    "title": title,
                                    "salary": salary,
                                    "company": company,
                                    "url": vacancy_url,
                                    "published_at": datetime.now().strftime("%Y-%m-%d"),
                                    "region": region["name"],
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
                        # Задержка для избежания блокировки
                        delay = random.uniform(3, 6)
                        print(
                            f"Ожидание {delay:.1f} секунд перед следующим запросом..."
                        )
                        time.sleep(delay)
                        break

                    except requests.exceptions.RequestException as e:
                        retries += 1
                        print(f"Ошибка при запросе (попытка {retries}): {e}")
                        if retries == max_retries:
                            print("Достигнуто максимальное количество попыток")
                            break
                        delay = random.uniform(5, 10)
                        print(
                            f"Ожидание {delay:.1f} секунд перед повторной попыткой..."
                        )
                        time.sleep(delay)
                    except Exception as e:
                        retries += 1
                        print(f"Непредвиденная ошибка (попытка {retries}): {e}")
                        if retries == max_retries:
                            print("Достигнуто максимальное количество попыток")
                            break
                        delay = random.uniform(5, 10)
                        print(
                            f"Ожидание {delay:.1f} секунд перед повторной попыткой..."
                        )
                        time.sleep(delay)

                if not vacancy_items or not next_page or retries == max_retries:
                    break

            # Задержка между запросами по разным поисковым терминам
            delay = random.uniform(2, 5)
            print(
                f"Переход к следующему поисковому запросу через {delay:.1f} секунд..."
            )
            time.sleep(delay)

    return all_vacancies


def save_data(data):
    """Сохраняет собранные данные в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("../data"):
        os.makedirs("../data")

    filename = f"../data/avito_api_data_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Данные сохранены в файл: {filename}")
    return filename


def main():
    try:
        print("Начинаем сбор данных с Авито...")
        target_count = 1000  # Целевое количество вакансий
        vacancies = get_vacancies(target_count)

        # Удаляем дубликаты по ID
        unique_ids = set()
        unique_vacancies = []
        for vacancy in vacancies:
            if vacancy["id"] not in unique_ids:
                unique_ids.add(vacancy["id"])
                unique_vacancies.append(vacancy)

        print(f"Удалено дубликатов: {len(vacancies) - len(unique_vacancies)}")

        # Сохраняем данные
        filename = save_data(unique_vacancies)

        print(f"\nВсего собрано {len(unique_vacancies)} уникальных вакансий с Авито")
        print(f"Файл: {filename}")

        return filename, len(unique_vacancies)
    except Exception as e:
        print(f"Произошла ошибка при выполнении скрипта: {e}")
        return None, 0
