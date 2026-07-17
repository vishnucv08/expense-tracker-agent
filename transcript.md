# Sample Run — Expense Tracker Agent (live, real data, via Groq API)

This transcript uses `icici_expenses_jul25_jul26.csv` — 1,189 real spend
transactions parsed from an actual ICICI Bank statement (Jul 2025 - Jul
2026), categorized live via Groq (`llama-3.1-8b-instant`). All numbers
below are cross-checked against the categorization summary and the
month-on-month table for consistency.

## Load & categorize

```
python expense_tracker.py icici_expenses_jul25_jul26.csv --table --excel --chart
```

```
Loading icici_expenses_jul25_jul26.csv and categorizing transactions (via LLM API)...
Loaded 1189 transactions across 13 months.

category
Other            1056020.80
Investments       945658.00
Bills             627596.20
Transfers         104820.60
Dining            103416.27
Entertainment      43599.17
Transport          29071.26
Shopping           17437.85
Loan/EMI           14410.00
Health             14057.00
Groceries           8042.43
Travel              2477.00

Saved chart to spending_by_category.png
Saved Excel summary to spending_by_month.xlsx
```

(45 rows with a genuinely empty date cell in the source CSV are excluded
from date-based views but still count in the category totals above — see
NOTES.md.)

## Q&A — every aggregation type, all verified against the table

```
> How much did I spend on dining last month?
You spent Rs. 5,338.75 on Dining between 2026-06-01 and 2026-07-01 across 27 transaction(s).

> And what about the month before?
You spent Rs. 9,381.94 on Dining between 2026-05-01 and 2026-06-01 across 46 transaction(s).

> What was my biggest expense category last month?
Your biggest spending category was Other at Rs. 1,004,297.00.

> Give me a breakdown of my spending by category.
Spending breakdown:
  - Other: Rs. 1,056,020.80
  - Investments: Rs. 945,658.00
  - Bills: Rs. 627,596.20
  - Transfers: Rs. 104,820.60
  - Dining: Rs. 103,416.27
  - Entertainment: Rs. 43,599.17
  - Transport: Rs. 29,071.26
  - Shopping: Rs. 17,437.85
  - Loan/EMI: Rs. 14,410.00
  - Health: Rs. 14,057.00
  - Groceries: Rs. 8,042.43
  - Travel: Rs. 2,477.00

> How many transactions did I make in June?
There were 80 matching transaction(s).

> What percentage of my spending was on Investments?
You spent Rs. 945,658.00 on Investments, which is 31.88% of your total spend of Rs. 2,966,606.58 in that range.

> How much more did I spend in June compared to May?
You spent Rs. 1,342,522.75 in the more recent period vs Rs. 34,647.79 before -- that's Rs. 1,307,874.96 more (3774.8%).
```

## Month-on-month table (categories x months, Jul 2025 - Jul 2026)

```
+---------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+--------------+------------+--------------+
| Category      |   Jul 2025 |   Aug 2025 |   Sep 2025 |   Oct 2025 |   Nov 2025 |   Dec 2025 |   Jan 2026 |   Feb 2026 |   Mar 2026 |   Apr 2026 |   May 2026 |     Jun 2026 |   Jul 2026 |        Total |
+===============+============+============+============+============+============+============+============+============+============+============+============+==============+============+==============+
| Bills         |     499.07 |     464.99 |  13,500.00 |       0.00 |      17.98 |       6.00 | 512,688.16 |      55.00 |      40.00 |       0.00 |     280.00 |   100,000.00 |      45.00 |   627,596.20 |
| Dining        |     210.00 |  13,521.74 |   4,129.95 |   2,401.28 |  20,360.75 |   5,269.13 |   2,832.46 |   4,068.65 |   8,813.19 |   9,177.43 |   9,381.94 |     5,338.75 |  17,911.00 |   103,416.27 |
| Entertainment |      90.00 |     217.00 |     258.00 |     598.00 |   7,084.24 |   4,092.00 |     814.00 |   1,550.15 |  14,437.78 |   1,913.00 |  11,827.00 |       708.00 |      10.00 |    43,599.17 |
| Groceries     |       0.00 |      36.00 |     207.00 |     134.00 |      20.00 |     155.00 |       0.00 |       0.00 |   5,890.00 |     231.00 |   1,137.43 |         0.00 |     232.00 |     8,042.43 |
| Health        |       0.00 |     180.00 |   2,100.00 |  10,079.00 |      50.00 |       0.00 |     607.00 |       0.00 |       0.00 |       0.00 |     538.00 |         0.00 |     503.00 |    14,057.00 |
| Investments   |       0.00 |   6,078.00 |  14,000.00 |  14,000.00 | 141,000.00 |   8,500.00 |  14,000.00 |  10,000.00 |  10,000.00 |  10,025.00 |   4,040.00 |   210,000.00 | 504,015.00 |   945,658.00 |
| Loan/EMI      |     200.00 |     120.00 |   5,590.00 |   7,000.00 |   1,500.00 |       0.00 |       0.00 |       0.00 |       0.00 |       0.00 |       0.00 |         0.00 |       0.00 |    14,410.00 |
| Other         |      35.00 |   3,352.00 |   1,105.00 |   3,196.00 |   7,187.00 |  25,659.00 |     639.00 |   5,344.00 |   3,304.80 |   1,210.00 |     224.00 | 1,004,297.00 |     468.00 | 1,056,020.80 |
| Shopping      |       0.00 |   5,002.00 |      54.00 |   1,130.00 |      15.00 |   4,844.50 |   1,139.70 |   1,763.75 |   1,368.00 |     390.00 |   1,708.90 |        22.00 |       0.00 |    17,437.85 |
| Transfers     |  10,891.90 |  12,982.00 |   6,091.00 |   4,021.00 |  12,260.00 |   3,497.00 |  10,965.00 |  14,157.00 |   2,695.98 |  19,465.72 |   4,332.00 |     2,952.00 |     510.00 |   104,820.60 |
| Transport     |     487.90 |   3,343.00 |      72.00 |   1,014.00 |     239.00 |       0.00 |     571.99 |   2,047.00 |     562.85 |       0.00 |   1,178.52 |    19,205.00 |     350.00 |    29,071.26 |
| Travel        |       0.00 |       0.00 |       0.00 |       0.00 |   2,417.00 |       0.00 |      60.00 |       0.00 |       0.00 |       0.00 |       0.00 |         0.00 |       0.00 |     2,477.00 |
| Total         |  12,413.87 |  45,296.73 |  47,106.95 |  43,573.28 | 192,150.97 |  52,022.63 | 544,317.31 |  38,985.55 |  47,112.60 |  42,412.15 |  34,647.79 | 1,342,522.75 | 524,044.00 | 2,966,606.58 |
+---------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+--------------+------------+--------------+
```

Every Q&A answer above was hand-cross-checked against this table (e.g.
Dining/Jun 2026 = 5,338.75 matches the "How much did I spend on dining
last month?" answer exactly; May's Total column = 34,647.79 matches the
"change" comparison's prior-period figure).
