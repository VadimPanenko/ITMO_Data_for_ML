import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import missingno as msno

# Настройки для графиков
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_and_combine_data():
    """Загрузка данных и создание единого датасета"""
    # Загружаем последние файлы с данными от каждого источника
    hh_files_trimmed = glob.glob("../data/hh_api_data_trimmed_*.json")
    avito_files = glob.glob("../data/avito_api_data_*.json")

    if not hh_files_trimmed or not avito_files:
        raise FileNotFoundError(
            "Не найдены файлы с данными от одного или обоих источников"
        )

    latest_hh = max(hh_files_trimmed)
    latest_avito = max(avito_files)

    print(f"Анализируем файлы:\n- HeadHunter: {latest_hh}\n- Авито: {latest_avito}")

    with open(latest_hh, "r", encoding="utf-8") as f:
        hh_data = json.load(f)
    with open(latest_avito, "r", encoding="utf-8") as f:
        avito_data = json.load(f)

    # Создаем DataFrame
    hh_df = pd.DataFrame(hh_data)
    avito_df = pd.DataFrame(avito_data)

    # Извлекаем поля зарплаты
    for df in [hh_df, avito_df]:
        df["salary_from"] = df["salary"].apply(lambda x: x.get("from") if x else None)
        df["salary_to"] = df["salary"].apply(lambda x: x.get("to") if x else None)
        df["salary_currency"] = df["salary"].apply(
            lambda x: x.get("currency") if x else None
        )
        # Рассчитываем среднюю зарплату
        df["avg_salary"] = df.apply(
            lambda x: (
                (x["salary_from"] + x["salary_to"]) / 2
                if x["salary_from"] and x["salary_to"]
                else x["salary_from"] or x["salary_to"]
            ),
            axis=1,
        )

    # Объединяем данные
    combined_df = pd.concat([hh_df, avito_df], ignore_index=True)

    # Сохраняем объединенный датасет
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("../data"):
        os.makedirs("../data")

    combined_df.to_csv(f"../data/combined_dataset_{timestamp}.csv", index=False)
    print(
        f"Объединенный датасет сохранен в файл: ../data/combined_dataset_{timestamp}.csv"
    )

    return combined_df, hh_df, avito_df


def analyze_missing_values(df):
    """Подробный анализ пропущенных значений"""
    print("\n=== Анализ пропущенных значений ===")

    # Количество и процент пропусков по каждому столбцу
    missing_data = pd.DataFrame(
        {
            "Количество пропусков": df.isnull().sum(),
            "Процент пропусков": df.isnull().sum() / len(df) * 100,
        }
    )
    missing_data = missing_data.sort_values("Процент пропусков", ascending=False)

    print(missing_data)

    # Анализ пропусков по источникам данных
    missing_by_source = df.groupby("source").apply(
        lambda x: x.isnull().sum() / len(x) * 100
    )
    print("\nПроцент пропусков по источникам данных:")
    print(missing_by_source)

    # Анализ пропусков по профессиям
    missing_by_query = df.groupby("search_query").apply(
        lambda x: x.isnull().sum() / len(x) * 100
    )
    print("\nПроцент пропусков по профессиям:")
    print(missing_by_query)

    # Визуализация пропусков
    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    plt.title("Матрица пропущенных значений")
    plt.tight_layout()
    plt.savefig("../analysis/missing_values_matrix.png")

    plt.figure(figsize=(12, 8))
    msno.heatmap(df)
    plt.title("Тепловая карта корреляций пропущенных значений")
    plt.tight_layout()
    plt.savefig("../analysis/missing_values_heatmap.png")

    # Гипотеза о причинах пропусков
    print("\n=== Гипотеза о причинах пропусков ===")
    print(
        "1. Пропуски в полях зарплат могут быть связаны с политикой компаний не указывать зарплату в вакансиях."
    )
    print(
        "2. Некоторые компании могут указывать только нижнюю или верхнюю границу зарплаты."
    )
    print(
        "3. Пропуски в названиях компаний могут быть связаны с тем, что некоторые вакансии размещаются частными лицами."
    )
    print(
        "4. Возможно существуют различия в политике указания зарплаты между HeadHunter и Авито."
    )

    return missing_data


def analyze_outliers(df):
    """Анализ выбросов в данных"""
    print("\n=== Анализ выбросов ===")

    # Создаем копию DataFrame только с числовыми столбцами, связанными с зарплатой
    numeric_df = df[["salary_from", "salary_to", "avg_salary"]].copy()

    # Метод 1: Z-score
    z_scores = numeric_df.apply(lambda x: (x - x.mean()) / x.std())
    abs_z_scores = z_scores.abs()
    outliers_z_score = (abs_z_scores > 3).any(axis=1)
    print(f"Количество выбросов (Z-score > 3): {outliers_z_score.sum()}")
    print(f"Процент выбросов: {outliers_z_score.sum() / len(df) * 100:.2f}%")

    # Метод 2: IQR (межквартильный размах)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (
        (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    print(f"Количество выбросов (IQR): {outliers_iqr.sum()}")
    print(f"Процент выбросов: {outliers_iqr.sum() / len(df) * 100:.2f}%")

    # Визуализация распределения зарплат с указанием выбросов
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="source", y="avg_salary")
    plt.title("Распределение зарплат по источникам с выбросами (метод IQR)")
    plt.tight_layout()
    plt.savefig("../analysis/salary_outliers_by_source.png")

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x="search_query", y="avg_salary")
    plt.title("Распределение зарплат по профессиям с выбросами (метод IQR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../analysis/salary_outliers_by_profession.png")

    return outliers_z_score, outliers_iqr


def handle_missing_values(df, method="simple"):
    """Обработка пропущенных значений разными методами"""
    print(f"\n=== Обработка пропущенных значений: метод '{method}' ===")

    # Создаем копию DataFrame
    df_cleaned = df.copy()

    if method == "simple":
        # Простая замена средними значениями
        for col in ["salary_from", "salary_to", "avg_salary"]:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

        # Заполнение категориальных переменных наиболее частыми значениями
        for col in ["company", "region"]:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    elif method == "group_mean":
        # Заполнение средними значениями по группам
        for col in ["salary_from", "salary_to", "avg_salary"]:
            # Заполняем пропуски средними по профессии и источнику
            group_means = df_cleaned.groupby(["search_query", "source"])[col].transform(
                "mean"
            )
            df_cleaned[col] = df_cleaned[col].fillna(group_means)

            # Если остались пропуски, заполняем общими средними
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

        # Заполнение категориальных переменных наиболее частыми значениями по группам
        for col in ["company", "region"]:
            df_cleaned[col] = df_cleaned[col].fillna("Не указано")

    elif method == "knn":
        from sklearn.impute import KNNImputer

        # Подготовка данных для KNN
        numeric_cols = ["salary_from", "salary_to", "avg_salary"]
        # Создаем временную копию только числовых данных
        numeric_df = df_cleaned[numeric_cols].copy()

        # Заполняем пропуски с помощью KNN
        imputer = KNNImputer(n_neighbors=5)
        numeric_df_imputed = pd.DataFrame(
            imputer.fit_transform(numeric_df),
            columns=numeric_cols,
            index=numeric_df.index,
        )

        # Заменяем числовые столбцы на заполненные
        for col in numeric_cols:
            df_cleaned[col] = numeric_df_imputed[col]

        # Заполнение категориальных переменных
        for col in ["company", "region"]:
            df_cleaned[col] = df_cleaned[col].fillna("Не указано")

    elif method == "drop":
        # Удаление строк с пропущенными значениями
        df_cleaned = df_cleaned.dropna(subset=["avg_salary"])

    # Рассчитываем статистику до и после обработки
    before_stats = {
        "count": len(df),
        "missing_avg_salary": df["avg_salary"].isnull().sum(),
        "mean_avg_salary": df["avg_salary"].mean(),
        "median_avg_salary": df["avg_salary"].median(),
    }

    after_stats = {
        "count": len(df_cleaned),
        "missing_avg_salary": df_cleaned["avg_salary"].isnull().sum(),
        "mean_avg_salary": df_cleaned["avg_salary"].mean(),
        "median_avg_salary": df_cleaned["avg_salary"].median(),
    }

    print(f"До обработки: {before_stats}")
    print(f"После обработки: {after_stats}")

    return df_cleaned, before_stats, after_stats


def handle_outliers(df, method="iqr"):
    """Обработка выбросов разными методами"""
    print(f"\n=== Обработка выбросов: метод '{method}' ===")

    # Создаем копию DataFrame
    df_cleaned = df.copy()

    # Выделяем числовые столбцы, связанные с зарплатой
    numeric_cols = ["salary_from", "salary_to", "avg_salary"]
    numeric_df = df_cleaned[numeric_cols].copy()

    outliers_mask = None

    if method == "iqr":
        # Метод IQR (межквартильный размах)
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        outliers_mask = (
            (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        ).any(axis=1)

        # Удаляем выбросы
        df_cleaned = df_cleaned[~outliers_mask]

    elif method == "z_score":
        # Метод Z-score
        z_scores = numeric_df.apply(lambda x: (x - x.mean()) / x.std())
        abs_z_scores = z_scores.abs()
        outliers_mask = (abs_z_scores > 3).any(axis=1)

        # Удаляем выбросы
        df_cleaned = df_cleaned[~outliers_mask]

    elif method == "winsorize":
        # Винзоризация данных (обрезка экстремальных значений)
        from scipy.stats import mstats

        for col in numeric_cols:
            if (
                df_cleaned[col].notna().any()
            ):  # проверяем, что в столбце есть непустые значения
                # Сохраняем маску непустых значений
                not_null_mask = df_cleaned[col].notna()

                # Применяем винзоризацию только к непустым значениям
                winsorized_data = mstats.winsorize(
                    df_cleaned.loc[not_null_mask, col], limits=[0.05, 0.05]
                )

                # Обновляем только непустые значения
                df_cleaned.loc[not_null_mask, col] = winsorized_data

    elif method == "cap":
        # Метод ограничения (capping)
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Ограничиваем значения
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

    # Рассчитываем статистику до и после обработки
    before_stats = {
        "count": len(df),
        "mean_avg_salary": df["avg_salary"].mean(),
        "median_avg_salary": df["avg_salary"].median(),
        "std_avg_salary": df["avg_salary"].std(),
    }

    after_stats = {
        "count": len(df_cleaned),
        "mean_avg_salary": df_cleaned["avg_salary"].mean(),
        "median_avg_salary": df_cleaned["avg_salary"].median(),
        "std_avg_salary": df_cleaned["avg_salary"].std(),
    }

    print(f"До обработки: {before_stats}")
    print(f"После обработки: {after_stats}")

    if outliers_mask is not None:
        print(f"Количество удаленных выбросов: {outliers_mask.sum()}")
        print(f"Процент удаленных выбросов: {outliers_mask.sum() / len(df) * 100:.2f}%")

    return df_cleaned, before_stats, after_stats


def process_categorical_features(df):
    """Обработка категориальных признаков"""
    print("\n=== Обработка категориальных признаков ===")

    # Создаем копию DataFrame
    df_processed = df.copy()

    # 1. Кодирование 'source' с помощью Label Encoding (бинарное кодирование)
    df_processed["source_encoded"] = df_processed["source"].map({"hh": 0, "avito": 1})

    # 2. Кодирование 'search_query' с помощью One-Hot Encoding
    search_query_dummies = pd.get_dummies(df_processed["search_query"], prefix="query")
    df_processed = pd.concat([df_processed, search_query_dummies], axis=1)

    # 3. Кодирование 'region' с помощью Target Encoding (среднее значение целевой переменной)
    region_target_means = df_processed.groupby("region")["avg_salary"].mean().to_dict()
    df_processed["region_encoded"] = df_processed["region"].map(region_target_means)

    # Заполняем пропуски после кодирования
    df_processed["region_encoded"] = df_processed["region_encoded"].fillna(
        df_processed["avg_salary"].mean()
    )

    # Анализируем закодированные данные
    print("Закодированные признаки:")
    for col in ["source_encoded", "region_encoded"] + list(
        search_query_dummies.columns
    ):
        if col in df_processed.columns:
            print(f"{col}: {df_processed[col].nunique()} уникальных значений")

    return df_processed


def evaluate_methods(df):
    """Оценка различных методов обработки данных"""
    print("\n=== Оценка методов обработки данных ===")

    # Создаем словарь для хранения результатов
    results = {}

    # Исходные данные (с удалением строк с пропущенными значениями для базового сравнения)
    df_base = df.dropna(subset=["avg_salary"]).copy()

    # Подготовка данных для регрессии
    X_base = pd.get_dummies(df_base[["source", "search_query"]], drop_first=True)
    y_base = df_base["avg_salary"]

    # Разделение на обучающую и тестовую выборки
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_base, y_base, test_size=0.3, random_state=42
    )

    # Обучение базовой модели
    model_base = LinearRegression()
    model_base.fit(X_train_base, y_train_base)
    y_pred_base = model_base.predict(X_test_base)

    # Оценка базовой модели
    base_rmse = np.sqrt(mean_squared_error(y_test_base, y_pred_base))
    base_r2 = r2_score(y_test_base, y_pred_base)

    results["base"] = {
        "description": "Базовая модель (удаление пропусков)",
        "count": len(df_base),
        "mean": df_base["avg_salary"].mean(),
        "median": df_base["avg_salary"].median(),
        "std": df_base["avg_salary"].std(),
        "rmse": base_rmse,
        "r2": base_r2,
    }

    # Методы обработки пропусков
    missing_methods = ["simple", "group_mean", "knn", "drop"]
    for method in missing_methods:
        # Обработка пропусков
        df_cleaned, _, _ = handle_missing_values(df, method=method)

        # Подготовка данных для регрессии
        X = pd.get_dummies(df_cleaned[["source", "search_query"]], drop_first=True)
        y = df_cleaned["avg_salary"]

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценка модели
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[f"missing_{method}"] = {
            "description": f"Обработка пропусков: {method}",
            "count": len(df_cleaned),
            "mean": df_cleaned["avg_salary"].mean(),
            "median": df_cleaned["avg_salary"].median(),
            "std": df_cleaned["avg_salary"].std(),
            "rmse": rmse,
            "r2": r2,
        }

    # Методы обработки выбросов (применяем к данным после обработки пропусков методом group_mean)
    df_no_missing, _, _ = handle_missing_values(df, method="group_mean")

    outlier_methods = ["iqr", "z_score", "winsorize", "cap"]
    for method in outlier_methods:
        # Обработка выбросов
        df_no_outliers, _, _ = handle_outliers(df_no_missing, method=method)

        # Подготовка данных для регрессии
        X = pd.get_dummies(df_no_outliers[["source", "search_query"]], drop_first=True)
        y = df_no_outliers["avg_salary"]

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценка модели
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[f"outliers_{method}"] = {
            "description": f"Обработка выбросов: {method}",
            "count": len(df_no_outliers),
            "mean": df_no_outliers["avg_salary"].mean(),
            "median": df_no_outliers["avg_salary"].median(),
            "std": df_no_outliers["avg_salary"].std(),
            "rmse": rmse,
            "r2": r2,
        }

    # Комбинация лучших методов (предположим, что это group_mean + winsorize)
    df_best_missing, _, _ = handle_missing_values(df, method="group_mean")
    df_best, _, _ = handle_outliers(df_best_missing, method="winsorize")

    # Обработка категориальных признаков
    df_best_categorical = process_categorical_features(df_best)

    # Подготовка данных для регрессии
    X = df_best_categorical[
        ["source_encoded", "region_encoded"]
        + [col for col in df_best_categorical.columns if col.startswith("query_")]
    ]
    y = df_best_categorical["avg_salary"]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Оценка модели
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results["best_combined"] = {
        "description": "Лучшая комбинация методов (group_mean + winsorize + обработка категориальных)",
        "count": len(df_best_categorical),
        "mean": df_best_categorical["avg_salary"].mean(),
        "median": df_best_categorical["avg_salary"].median(),
        "std": df_best_categorical["avg_salary"].std(),
        "rmse": rmse,
        "r2": r2,
    }

    # Вывод результатов
    results_df = pd.DataFrame(results).T
    print(results_df[["description", "count", "mean", "median", "std", "rmse", "r2"]])

    # Сохранение результатов
    results_df.to_csv("../analysis/method_comparison.csv")

    # Визуализация сравнения методов
    plt.figure(figsize=(14, 8))

    # Метрика RMSE (меньше - лучше)
    plt.subplot(1, 2, 1)
    plt.bar(results_df.index, results_df["rmse"])
    plt.title("Сравнение методов по RMSE")
    plt.xticks(rotation=45)
    plt.ylabel("RMSE")

    # Метрика R^2 (больше - лучше)
    plt.subplot(1, 2, 2)
    plt.bar(results_df.index, results_df["r2"])
    plt.title("Сравнение методов по R²")
    plt.xticks(rotation=45)
    plt.ylabel("R²")

    plt.tight_layout()
    plt.savefig("../analysis/methods_comparison.png")

    # Определение лучшего метода
    best_method = results_df.loc[results_df["rmse"].idxmin()]

    print("\n=== Выбор оптимального метода ===")
    print(f"Лучший метод: {best_method['description']}")
    print(f"RMSE: {best_method['rmse']:.2f}")
    print(f"R²: {best_method['r2']:.2f}")

    return results, best_method


def formulate_recommendations():
    """Формулировка рекомендаций по обработке данных"""
    print("\n=== Рекомендации по обработке данных ===")

    recommendations = """
Рекомендации по оптимальному подходу к обработке данных:

1. Обработка пропущенных значений:
   - Для зарплат рекомендуется использовать групповое заполнение пропусков (group_mean), 
     которое учитывает специфику профессии и источника данных.
   - Для категориальных признаков (компания, регион) рекомендуется 
     использовать специальную категорию "Не указано".

2. Обработка выбросов:
   - Рекомендуется применять метод винзоризации (winsorize), который 
     ограничивает экстремальные значения, но сохраняет все наблюдения.
   - В случае сильно зашумленных данных может быть эффективным метод ограничения 
     значений (capping).

3. Обработка категориальных признаков:
   - Для переменных с малым числом категорий (source) рекомендуется 
     бинарное кодирование.
   - Для переменных с большим числом категорий (region) рекомендуется 
     target encoding.
   - Для поисковых запросов рекомендуется one-hot encoding.

4. Общие рекомендации:
   - Важно анализировать данные до и после обработки, чтобы убедиться, что 
     преобразования не искажают важные характеристики данных.
   - Рекомендуется сохранять информацию о проведенных преобразованиях для 
     последующей интерпретации результатов.
   - Для итоговой модели машинного обучения рекомендуется использовать 
     комбинацию методов: group_mean для пропусков, winsorize для выбросов 
     и соответствующие методы кодирования для категориальных переменных.
    """

    print(recommendations)

    # Сохраняем рекомендации в файл
    with open(
        "../analysis/data_processing_recommendations.txt", "w", encoding="utf-8"
    ) as f:
        f.write(recommendations)

    return recommendations


def main():
    """Основная функция"""
    # Создаем директорию для результатов, если её нет
    if not os.path.exists("../analysis"):
        os.makedirs("../analysis")

    # Шаг 1: Загрузка и объединение данных
    print("Шаг 1: Загрузка и объединение данных")
    combined_df, hh_df, avito_df = load_and_combine_data()

    # Шаг 2: Анализ пропущенных значений
    print("\nШаг 2: Анализ пропущенных значений")
    missing_data = analyze_missing_values(combined_df)

    # Шаг 3: Анализ выбросов
    print("\nШаг 3: Анализ выбросов")
    outliers_z_score, outliers_iqr = analyze_outliers(combined_df)

    # Шаг 4: Сравнение методов обработки данных
    print("\nШаг 4: Сравнение методов обработки данных")
    results, best_method = evaluate_methods(combined_df)

    # Шаг 5: Формулировка рекомендаций
    print("\nШаг 5: Формулировка рекомендаций")
    recommendations = formulate_recommendations()

    print("\nАнализ данных завершен. Результаты сохранены в папке 'analysis'.")


if __name__ == "__main__":
    main()
