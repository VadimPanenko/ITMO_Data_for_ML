import json
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import argparse
import re


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


def parse_job_page(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        time.sleep(random.uniform(2, 4))

        response = requests.get(url, headers=headers, timeout=10)
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
            "source": "avito",
        }
    except requests.RequestException as e:
        print(f"Error fetching job page {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing job page {url}: {e}")
        return None


def scrape_avito(search_queries, max_urls=10):
    results = []

    for query in search_queries:
        print(f"Searching for: {query}")

        page = 1
        urls_collected = 0

        while urls_collected < max_urls:
            base_url = "https://www.avito.ru/all/vakansii"
            params = {"q": query}

            if page > 1:
                params["p"] = page

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            }

            try:
                response = requests.get(
                    base_url, params=params, headers=headers, timeout=10
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                job_items = soup.select("div[data-marker='item']")

                if not job_items:
                    print(f"No more results found for {query} on page {page}")
                    break

                print(f"Found {len(job_items)} jobs for {query} on page {page}")

                for item in job_items:
                    if urls_collected >= max_urls:
                        break

                    link = item.select_one("a[itemprop='url']")
                    if link and "href" in link.attrs:
                        job_url = "https://www.avito.ru" + link["href"]
                        job_data = parse_job_page(job_url)

                        if job_data:
                            job_data["search_query"] = query
                            print(
                                f"Parsed job: {job_data['title']} - {job_data['company_name']}"
                            )
                            results.append(job_data)
                            urls_collected += 1

                page += 1
                time.sleep(random.uniform(3, 5))

            except requests.RequestException as e:
                print(f"Error searching jobs for {query}, page {page}: {e}")
                break
            except Exception as e:
                print(f"Error parsing search results for {query}, page {page}: {e}")
                break

    return results


def save_results(results, output_dir="../data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/avito_api_data_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} job listings to {output_file}")
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
        "--max-urls",
        type=int,
        default=10,
        help="Maximum number of URLs to scrape per query",
    )
    args = parser.parse_args()

    print("Starting Avito scraper")
    results = scrape_avito(args.queries, args.max_urls)

    save_results(results)
    print(f"Done scraping {len(results)} jobs")


if __name__ == "__main__":
    main()
