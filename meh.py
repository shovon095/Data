
What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Baseline- GPT-4o: "SELECT schools.MailStreet, schools.MailCity, schools.MailState, schools.MailZip\nFROM frpm\nJOIN schools ON frpm.CDSCode = schools.CDSCode\nORDER BY `FRPM Count (K-12)` DESC\nLIMIT 1\t----- bird -----\tcalifornia_schools",
attention -GpT-4o+Attention": "SELECT frpm.`Percent (%) Eligible Free (Ages 5-17)`\nFROM frpm\nJOIN schools ON frpm.CDSCode = schools.CDSCode\nWHERE schools.`EdOpsName` = 'Continuation'\nORDER BY frpm.`Percent (%) Eligible Free (Ages 5-17)` ASC\nLIMIT 3\t----- bird -----\tcalifornia_schools"
