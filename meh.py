Baseline:
{
    "0": "SELECT MAX(`Percent (%) Eligible Free (K-12)`) AS MaxEligibleFreeRate\nFROM frpm\nJOIN schools ON frpm.CDSCode = schools.CDSCode\nWHERE schools.County = 'Alameda'\t----- bird -----\tcalifornia_schools",
    "1": "SELECT COUNT(DISTINCT account.account_id) AS eligible_accounts\nFROM account\nJOIN district ON account.district_id = district.district_id\nJOIN loan ON account.account_id = loan.account_id\nWHERE district.A2 = 'Prague'\t----- bird -----\tfinancial",
    "2": "SELECT \n    (COUNT(CASE WHEN l.status = 'A' THEN 1 END) * 100.0 / COUNT(*)) AS percentage_running_no_issue\nFROM \n    loan l\nJOIN \n    account a ON l.account_id = a.account_id\nWHERE \n    l.amount < 100000\t----- bird -----\tfinancial",
    "3": "SELECT f.CDSCode, f.`School Name`, f.`Percent (%) Eligible Free (Ages 5-17)`\nFROM frpm f\nJOIN schools s ON f.CDSCode = s.CDSCode\nWHERE s.StatusType = 'Continuation'\nORDER BY f.`Percent (%) Eligible Free (Ages 5-17)` ASC\nLIMIT 3\t----- bird -----\tcalifornia_schools"


Attention:
"0": "SELECT MAX(`Percent (%) Eligible Free (K-12)`)\nFROM frpm\nWHERE `County Name` = 'Alameda'\t----- bird -----\tcalifornia_schools",
    "1": "SELECT `Percent (%) Eligible Free (Ages 5-17)`\nFROM frpm\nJOIN schools ON frpm.CDSCode = schools.CDSCode\nWHERE schools.SchoolType = 'Continuation'\nORDER BY `Percent (%) Eligible Free (Ages 5-17)` ASC\nLIMIT 3\t----- bird -----\tcalifornia_schools",
    "2": "SELECT schools.Zip\nFROM schools\nJOIN frpm ON schools.CDSCode = frpm.CDSCode\nWHERE schools.County = 'Fresno' AND frpm.`Charter School (Y/N)` = 1\t----- bird -----\tcalifornia_schools",
    "3": "SELECT schools.MailStreet, schools.MailCity, schools.MailZip, schools.MailState\nFROM frpm\nJOIN schools ON frpm.CDSCode = schools.CDSCode\nORDER BY frpm.`FRPM Count (K-12)` DESC\nLIMIT 1\t----- bird -----\tcalifornia_schools",
    "4": "SELECT Phone \nFROM schools \nWHERE Charter = 1 AND FundingType = 'charter' AND OpenDate > '2000-01-01'\t----- bird -----\tcalifornia_schools",
    "5": "SELECT COUNT(*)\nFROM satscores\nJOIN schools ON satscores.cds = schools.CDSCode\nWHERE satscores.AvgScrMath < 400 AND schools.Virtual = 'Yes'\t----- bird -----\tcalifornia_schools",

    "0": "SELECT MAX(Percent (%) Eligible Free (K-12)) FROM frpm JOIN schools ON frpm.CDSCode = schools.CDSCode WHERE schools.County = 'Alameda'\t----- bird -----\tcalifornia_schools",
    "1": "SELECT COUNT(DISTINCT account.account_id) FROM account JOIN disp ON account.account_id = disp.account_id JOIN card ON disp.disp_id = card.disp_id JOIN district ON account.district_id = district.district_id WHERE account.frequency = 'issuance after transaction' AND district.A2 = 'East Bohemia'\t----- bird -----\tfinancial",
    "2": "SELECT COUNT(DISTINCT account.account_id) FROM account JOIN district ON account.district_id = district.district_id JOIN loan ON account.account_id = loan.account_id WHERE district.A2 = 'Prague'\t----- bird -----\tfinancial",
    "3": "SELECT a.account_id, d.A2 AS district_name, d.A3 AS district_region FROM account a JOIN district d ON a.district_id = d.district_id JOIN trans t ON a.account_id = t.account_id JOIN card c ON c.disp_id IN ( SELECT disp_id FROM disp WHERE account_id = a.account_id) WHERE a.date LIKE '1993%' AND c.issued > t.date\t----- bird -----\tfinancial",    "4": "SELECT frpm.Percent (%) Eligible Free (Ages 5-17) FROM frpm JOIN schools ON frpm.CDSCode = schools.CDSCode WHERE schools.SchoolType = 'Continuation' ORDER BY frpm.Percent (%) Eligible Free (Ages 5-17) ASC LIMIT 3\t----- bird -----\tcalifornia_schools"
