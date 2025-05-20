  {
        "question_id": 0,
        "db_id": "california_schools",
        "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
        "difficulty": "simple"
    },


    {
        "question_id": 90,
        "db_id": "financial",
        "question": "How many accounts who have region in Prague are eligible for loans?",
        "evidence": "A3 contains the data of region",
        "SQL": "SELECT COUNT(T1.account_id) FROM account AS T1 INNER JOIN loan AS T2 ON T1.account_id = T2.account_id INNER JOIN district AS T3 ON T1.district_id = T3.district_id WHERE T3.A3 = 'Prague'",
        "difficulty": "simple"
    },

 {                                                                                                                                                               "question_id": 118,                                                                                                                                         "db_id": "financial",                                                                                                                                       "question": "For loan amount less than USD100,000, what is the percentage of accounts that is still running with no issue.",                                "evidence": "Status = 'C' stands for running contract, ok so far; Percentage of accounts by condition = [(total(amount) & condition) / (total amount)] * 100%.",
        "SQL": "SELECT CAST(SUM(status = 'C') AS REAL) * 100 / COUNT(amount) FROM loan WHERE amount < 100000",
        "difficulty": "moderate"
    },

