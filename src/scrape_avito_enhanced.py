import json
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import threading

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
lock = threading.Lock()


def get_random_ua():
    ua = UserAgent()
    return ua.random


def get_proxy():
    return None


def extract_salary(salary_text):
    if not salary_text or salary_text.lower() == "по договорённости":
        return {"from": None, "to": None, "currency": None}

    salary_text = salary_text.replace("\xa0", "").strip()

    if "–" in salary_text:
        parts = salary_text.split("–")
        min_salary = int("".join(filter(str.isdigit, parts[0])))

        max_parts = parts[1].split()
        max_salary = int("".join(filter(str.isdigit, max_parts[0])))

        currency = "".join(
            [c for c in max_parts[-1] if not c.isdigit() and c != " "]
        ).strip()

        return {"from": min_salary, "to": max_salary, "currency": currency}
    else:
        parts = salary_text.split()
        amount = int("".join(filter(str.isdigit, parts[0])))

        currency = "".join(
            [c for c in parts[-1] if not c.isdigit() and c != " "]
        ).strip()

        if "от" in salary_text.lower():
            return {"from": amount, "to": None, "currency": currency}
        elif "до" in salary_text.lower():
            return {"from": None, "to": amount, "currency": currency}
        else:
            return {"from": amount, "to": amount, "currency": currency}


def extract_skills(soup):
    skills = []
    skills_div = soup.select_one("div[data-marker='item-skills']")

    if skills_div:
        skill_items = skills_div.select("li")
        for skill in skill_items:
            skills.append(skill.text.strip())

    return skills


def get_requirements_responsibilities(description_text):
    if not description_text:
        return None, None

    requirements = None
    responsibilities = None

    paragraphs = description_text.split("\n\n")

    for p in paragraphs:
        lower_p = p.lower()
        if "требования" in lower_p or "ожидания" in lower_p:
            requirements = p
        elif "обязанности" in lower_p or "задачи" in lower_p:
            responsibilities = p

    return requirements, responsibilities


def extract_job_experience(soup):
    experience_div = soup.select_one("div[data-marker='item-experience']")
    if experience_div:
        return experience_div.text.strip()
    return None


def extract_employment_type(soup):
    div = soup.select_one("div[data-marker='item-employment-type']")
    if div:
        return div.text.strip()
    return None


def extract_schedule(soup):
    div = soup.select_one("div[data-marker='item-schedule-type']")
    if div:
        return div.text.strip()
    return None


def parse_job_page(url):
    try:
        headers = {
            "User-Agent": get_random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
        }

        proxy = get_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None

        time.sleep(random.uniform(2, 4))

        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title_element = soup.select_one("h1[data-marker='item-view/title-info']")
        title = title_element.text.strip() if title_element else None

        price_element = soup.select_one("span[data-marker='item-view/item-price']")
        salary_text = price_element.text.strip() if price_element else None
        salary_data = extract_salary(salary_text)

        company_name_element = soup.select_one("div[data-marker='seller-info/name']")
        company_name = (
            company_name_element.text.strip() if company_name_element else None
        )

        location_element = soup.select_one("div[data-marker='item-view/item-address']")
        location = location_element.text.strip() if location_element else None

        description_element = soup.select_one(
            "div[data-marker='item-view/item-description']"
        )
        description = description_element.text.strip() if description_element else None

        date_element = soup.select_one("div[data-marker='item-view/item-date']")
        date_str = date_element.text.strip() if date_element else None

        if date_str:
            match = re.search(r"(\d{1,2}\s\w+\s\d{4})", date_str)
            if match:
                date_str = match.group(1)
            else:
                date_str = datetime.now().strftime("%d %B %Y")

        skills = extract_skills(soup)
        requirements, responsibilities = get_requirements_responsibilities(description)
        experience = extract_job_experience(soup)
        employment_type = extract_employment_type(soup)
        schedule = extract_schedule(soup)

        vacancy_id = url.split("_")[-1]

        return {
            "id": vacancy_id,
            "title": title,
            "salary_data": salary_data,
            "company_name": company_name,
            "location": location,
            "description": description,
            "url": url,
            "date": date_str,
            "skills": skills,
            "requirements": requirements,
            "responsibilities": responsibilities,
            "experience": experience,
            "employment_type": employment_type,
            "schedule": schedule,
            "source": "avito",
        }
    except requests.RequestException as e:
        logging.error(f"Error fetching job page {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing job page {url}: {e}")
        return None


def search_jobs(search_query, location=None, page=1):
    try:
        base_url = "https://www.avito.ru/all/vakansii"
        params = {"q": search_query}

        if location:
            params["location"] = location

        if page > 1:
            params["p"] = page

        headers = {
            "User-Agent": get_random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        proxy = get_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None

        response = requests.get(
            base_url, params=params, headers=headers, proxies=proxies, timeout=10
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        job_items = soup.select("div[data-marker='item']")

        logging.info(
            f"Found {len(job_items)} jobs for query '{search_query}' on page {page}"
        )

        job_urls = []
        for item in job_items:
            link = item.select_one("a[itemprop='url']")
            if link and "href" in link.attrs:
                job_url = "https://www.avito.ru" + link["href"]
                job_urls.append(job_url)

        return job_urls
    except requests.RequestException as e:
        logging.error(f"Error searching jobs for {search_query}, page {page}: {e}")
        return []
    except Exception as e:
        logging.error(
            f"Error parsing search results for {search_query}, page {page}: {e}"
        )
        return []


def process_job_urls(job_urls, search_query, results):
    for url in job_urls:
        try:
            job_data = parse_job_page(url)
            if job_data:
                job_data["search_query"] = search_query
                with lock:
                    results.append(job_data)
                    logging.info(
                        f"Parsed job: {job_data['title']} - {job_data['company_name']}"
                    )
        except Exception as e:
            logging.error(f"Error processing job URL {url}: {e}")


def scrape_avito(search_queries, max_pages=2, max_workers=3):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for query in search_queries:
            logging.info(f"Searching for: {query}")

            for page in range(1, max_pages + 1):
                job_urls = search_jobs(query, page=page)

                if not job_urls:
                    logging.info(f"No more jobs found for {query} at page {page}")
                    break

                process_job_urls(job_urls, query, results)

                time.sleep(random.uniform(3, 5))

    return results


def save_results(results, output_dir="../data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/avito_api_data_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Scrape job listings from Avito")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "Data Scientist",
            "Data Analyst",
            "Machine Learning",
            "Python Developer",
        ],
    )
    parser.add_argument(
        "--max-pages", type=int, default=2, help="Maximum pages to scrape per query"
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="Maximum number of worker threads"
    )
    args = parser.parse_args()

    logging.info("Starting Avito scraper")
    results = scrape_avito(args.queries, args.max_pages, args.max_workers)

    output_file = save_results(results)
    logging.info(f"Saved {len(results)} job listings to {output_file}")


if __name__ == "__main__":
    main()
