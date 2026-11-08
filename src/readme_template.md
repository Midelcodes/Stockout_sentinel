# Stockout Sentinel

Forecasts facility contraceptive demand, flags stockout risk, suggests within-district transfers and briefs a district pharmacist in English or French. **Historical data as of {as_of}. Demonstration only; not for operational use.** {live_line} [2-minute video](https://youtu.be/EYA5Bn-vclg) · [LLM playbook for programme teams](docs/llm-playbook.md) · MIT licence.

**Headline:** {headline}

## Problem
Facilities stock out while supply sits nearby, and logistics capacity is under strain as external funding falls. Can a model beat the 3-month average (AMC) logistics systems use today?

## Data
[USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data): a Côte d'Ivoire eLMIS extract of {products} contraceptives at {sites} sites, {first_month} to {last_month}, {rows} rows. Côte d'Ivoire has used AI forecasting since 2021 with Pendulum, supported by USAID and the Gates Foundation; a 2023 malaria proof of concept [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (a reported result, not an independent finding).

## Method
Rows are flagged, not fixed: {fake_zero} fake zeros, {all_zero} all-zero, {censored} censored, {balance} balance errors; {valid_y} valid targets remain. Direct multi-horizon LightGBM with p50/p90 quantiles, backtested at T-9, T-6 and T-3. p90 is widened using errors from earlier backtest folds only. Labels use months of stock from the mean forecast; an all-zero or fake-zero last report is "Unverified zero", and reports up to 3 months old count, with their age shown. Greedy within-district transfers use **SYNTHETIC** expiry dates. Briefs by {llm_model} pass a number, range and format guardrail or fall back to a template ({brief_status}). The app makes no LLM calls. Free-tier Gemini prompts may be used by Google to improve its products; they contain only public, aggregated facility data.

## Results
{results_table}

All models share the same rows. The stocking comparison is on the Model card.

## Budget prioritisation
When a budget cannot fund every resupply order, which come first? Each site and product labelled Stockout risk gets an order covering the larger of p90 demand over the lead time and 2 months of mean demand. A mixed-integer linear program (SciPy HiGHS) resolves as many risks as the budget allows, breaking ties toward more demand, and is compared with two hand rules. {budget_line} The data has no prices, so costs are 1 unit per stock unit until a national price list is added in `config.yaml`. Every order needs human review.

## Ask the docs
The app's "Ask the docs" page retrieves the passages of this README, the playbook and the guardrail notes that answer a question (TF-IDF over sections, with a relevance floor), and shows each with its source. Nothing is generated, so every answer can be checked.

## Limitations
Censored demand understates true need, and `stock_stockout_days` is 0 in every row. Fake-zero rules are heuristics, and an all-zero month can be a real stockout. Stale stock figures are not adjusted for use since. Expiry data is SYNTHETIC. No supplier lead times or order cycles. 2016 to 2019 data from one country and one commodity class (contraceptives, not all essential medicines). Transfers reach {served} of {recipients} series in need; {single_site} of {districts} districts have one site.

## Bias
{bias_text} `site_type` does not support a rural/urban claim. The guardrail stops invented numbers and wrong ranges but not every misplaced number, so a passed brief is not a verified one. Every transfer needs human review.

## Reuse
- [`llm_number_guardrail`](llm_number_guardrail/README.md): the number check, retry and fallback loop as a standalone, dependency-free package for any project where an LLM must not invent figures.
- [Playbook](docs/llm-playbook.md): seven rules, prompt patterns, a two-minute output check and a 60-minute workshop for programme teams using LLMs on health data.
- MIT licence, so ministries and partners can adopt and adapt it as an open digital public good.

## Next
Trace funded orders through procurement and cash release, real prices and budget ceilings, DHIS2 and eLMIS integration, Nigeria's NHLMIS, real batch and expiry data, lead-time-aware reorder logic.

## Run it
Python 3.11. Put the CSVs in `data/raw/` and optionally set `GEMINI_API_KEY` in `.env`, then `pip install -r requirements.txt`, `python -m src.pipeline --all`, `streamlit run app/Home.py`.
