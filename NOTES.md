# Notes

**On the data:** two datasets are included. `sample_transactions.csv` is
60 fabricated transactions formatted to look like real ICICI/Union Bank
statement exports, used for initial development. `icici_expenses_jul25_jul26.csv`
is **real data** — 1,189 actual spend transactions parsed from a real
ICICI Bank statement PDF (Jul 2025 - Jul 2026), with names/account details
kept out of this write-up.

**On the LLM provider:** built against the Anthropic API originally, then
switched to Groq's free tier after hitting an Anthropic billing wall.
Within Groq, also downgraded from `llama-3.3-70b-versatile` to the
lighter `llama-3.1-8b-instant` after the 70b model's daily token quota
(100k/day on the free tier) got exhausted partway through a 1,235-row
categorization run. The lighter model has a much higher daily allowance,
at some cost to reliability (see below).

**What was tricky, in the order they came up:**

1. **Extracting structured transactions from a real bank statement PDF.**
   ICICI's e-statement layout has no visible cell borders, so
   `pdfplumber`'s default table detection only finds the header row. Had
   to reconstruct rows from raw word positions instead: anchor on the
   "S.No" column (sequential integers, unambiguous), then window the
   surrounding text by vertical position to capture each row's multi-line
   remarks. One layout quirk took a while to spot: the transaction's
   "payee name" line prints *above* its own S.No/date/amount line, not
   below it — a fixed ~6pt offset in the windowing logic accounts for
   this. Validated the whole 110-page, 1,451-row extraction by checking
   that every row's balance recalculates correctly from the previous
   balance ± the transaction amount — zero mismatches across the full
   statement, including a large-balance edge case that broke an earlier,
   fixed-pixel-column version of the parser (7-digit balances shift the
   column position enough to cross a hardcoded threshold; fixed by
   determining withdrawal/deposit direction from the running balance
   itself instead of column position).

2. **Turning free-text questions into reliable filters.** Rather than
   asking the LLM to write and run arbitrary pandas code per question, it
   emits a small structured JSON "query plan" (category, date range,
   amount bounds, aggregation type), executed locally with plain pandas.

3. **The LLM was hallucinating numbers when phrasing final answers.**
   Initially, the computed result (an exact dict from pandas) was handed
   back to the LLM with a "phrase this in plain English" prompt. On the
   real 1,189-row dataset this produced clearly wrong output — e.g.
   reporting Rs. 20,525 for a category/month combination that actually
   totalled Rs. 5,338.75, and a "Rs. 0 difference" for two months whose
   real difference was over Rs. 13 lakh. Root cause: a small, fast model
   isn't reliable at transcribing exact numbers out of a JSON blob it was
   just handed. Fixed by never letting the LLM restate computed numbers —
   answers are now always formatted deterministically in Python, straight
   from the result dict. The LLM's role is limited to translating the
   question into a filter, which is the part that actually benefits from
   its judgement.

4. **Date arithmetic was unreliable from the LLM planner.** Follow-up
   questions like "what about the month before?" occasionally resolved
   to nonsensical dates (one run returned "May 2022" out of nowhere), and
   bare month names sometimes landed on a year with zero data. Fixed the
   same way as the answer-hallucination issue: date resolution for
   explicit cues ("last month", "the month before", bare month names, "X
   vs Y" comparisons) is now done deterministically in Python and forced
   onto the plan as a hard override, regardless of what the LLM's JSON
   said. Verified with unit-style tests that simulate the LLM producing
   exactly the wrong output observed in production and confirm the
   override corrects it.

5. **Follow-up questions dropped filters inconsistently.** A related bug:
   "and what about the month before?" correctly carried the *date range*
   forward (that part goes through the deterministic override above) but
   the LLM sometimes silently dropped the *category* filter from the
   previous turn, answering with total spend instead of the
   category-scoped figure. Fixed with the same pattern — if the question
   looks like a follow-up and doesn't name a different category, the
   previous turn's category is forced onto the new plan rather than
   trusting the LLM to retain it.

6. **Batching categorization without hardcoding a lookup table.** ICICI
   and Union Bank format transaction descriptions very differently, so
   categorization leans on the model's judgement rather than string
   matching. Real statements also include large transfers that don't fit
   "everyday spending" categories — fixed-deposit transfers, mutual
   fund/SIP contributions, loan-style auto-debits, person-to-person
   transfers — so three extra categories (Investments, Transfers,
   Loan/EMI) were added alongside the original spending buckets, with
   explicit guidance in the categorization prompt for recognizing them.

7. **Groq's free-tier daily token limit.** A 1,235-transaction
   categorization run can exhaust the smaller free-tier quota partway
   through. Added disk-backed caching (`.category_cache.json`, keyed by a
   hash of merchant+description, saved after every batch) so an
   interrupted run resumes instead of re-spending tokens on transactions
   already categorized — verified by simulating a mid-run failure and
   confirming the next run picks up exactly where it left off.

8. **Messy CSV data in the wild.** The real statement CSV ended up with
   45 rows with a genuinely empty date cell (not a formatting issue --
   the cell itself is blank) after being handled outside the script.
   Rather than crash (an early version did, with a cryptic
   `NaTType does not support strftime`), the loader now does multi-format
   date parsing, drops only the genuinely unparseable rows, and reports
   exactly which transactions were affected so they're auditable rather
   than silently missing.

**Stretch goals implemented:** budget alerts (`--budget Category=limit`,
repeatable), a matplotlib bar chart (`--chart`), a category-by-month
Excel export with live `SUMIFS` formulas (`--excel`), the same table
printed in-terminal with proper grid borders (`--table`, or just ask for
it in the Q&A loop), percentage-of-total and period-over-period change
math, and follow-up question support (see point 5 above for a caveat
that was found and fixed).
