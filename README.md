# Expense Tracker Agent

An agent that reads bank transaction CSVs, categorizes spending using an
LLM (no hardcoded merchant->category lookup table), and answers
natural-language questions about the data in a simple Q&A loop.

Built for ICICI/Union Bank style statements, but works with any CSV in
the expected format.

## What it does

- Takes a CSV of transactions (`date`, `bank`, `merchant`, `amount`, `description`)
- Categorizes each transaction via an LLM (Groq, `llama-3.1-8b-instant`)
  into Dining, Groceries, Transport, Shopping, Bills, Entertainment,
  Health, Travel, Investments, Transfers, Loan/EMI, or Other -- inferred
  from the merchant/description text, not a fixed lookup table
- Keeps everything in memory as a pandas DataFrame
- Runs an interactive loop where you can ask things like:
  - `How much did I spend on dining last month?`
  - `What was my biggest expense category in March?`
  - `List all transactions above 5000.`
  - `What percentage of my spending was on Investments?`
  - `How much more did I spend in June compared to May?`
  - `And what about the month before?` (follow-ups carry context forward)
- Translates each question into a structured filter/aggregation, runs it
  with pandas, and answers in plain English

## Stretch features implemented

- **Chart** -- a matplotlib bar chart of spending by category (`--chart`)
- **Budget alerts** -- flag categories that exceed a set limit (`--budget Category=limit`, repeatable)
- **Follow-up questions** -- "and what about the month before?" resolves relative to the previous answer
- **Month-on-month table** -- categories x months, printed as a bordered grid or asked for directly in the Q&A loop (`--table`, or just ask)
- **Excel export** -- a categories x months workbook with live `SUMIFS` formulas, not static numbers (`--excel`)
- **Percentage & period-comparison math** -- "what percent of my spending was on X" and "how much more did I spend in X vs Y", computed exactly in pandas rather than left to the LLM to guess at

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key from [console.groq.com/keys](https://console.groq.com/keys), then:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your-key-here"

# macOS/Linux
export GROQ_API_KEY="your-key-here"
```

## Usage

```bash
# Basic run -- categorizes and drops you into the Q&A prompt
python expense_tracker.py sample_transactions.csv

# With the extras
python expense_tracker.py sample_transactions.csv --table --excel --chart --budget Dining=2500

# Non-interactive (batch questions, useful for testing/demos)
python expense_tracker.py sample_transactions.csv --questions "How much did I spend on dining last month?" "What was my biggest expense category in March?"

# Offline/no API key needed (heuristic stand-in for the LLM, for testing the pipeline)
python expense_tracker.py sample_transactions.csv --mock
```

Categorization results are cached to `.category_cache.json` (keyed by
merchant+description), so re-running the same CSV doesn't re-spend API
tokens -- this also means an interrupted run (e.g. hitting a rate limit)
resumes instead of starting over.

## Files

| File | What it is |
|---|---|
| `expense_tracker.py` | The whole thing -- one script |
| `sample_transactions.csv` | Synthetic test data (60 fabricated transactions formatted to look like real ICICI/Union Bank statement exports) |
| `transcript.md` | Real Q&A output from a live run, cross-checked against the underlying data |
| `NOTES.md` | What was tricky and how it was solved -- includes a real debugging story: parsing a real bank statement PDF, switching LLM providers after hitting a billing wall, and catching/fixing a case where the LLM hallucinated numbers when phrasing answers |

To use your own bank statement instead of the sample data, reshape it
into the same five columns (`date, bank, merchant, amount, description`)
and point the script at it. Real bank CSV exports are never committed to
this repo -- see `.gitignore`.

## Design notes

Questions are turned into a small structured JSON "query plan"
(category, date range, amount bounds, aggregation type) by the LLM, then
executed locally with plain pandas -- rather than asking the LLM to write
and run arbitrary code, or to restate computed numbers in prose (both
were tried and both turned out to be unreliable; see `NOTES.md` for the
specifics). The LLM's job is limited to the part it's actually good at:
turning a fuzzy question into a filter. Everything numeric is computed
deterministically.
