# Stockout Sentinel

Forecasts facility contraceptive demand, flags stockout risk, suggests within-district transfers and briefs a district pharmacist in English or French. **Historical data as of {as_of}. Demonstration only; not for operational use.** {live_line}

**Headline:** {headline}

## Problem
Facilities stock out while supply sits nearby, and logistics capacity is under strain as external funding falls. Can a model beat the 3-month average (AMC) logistics systems use today?

## Data
[USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data): a Côte d'Ivoire eLMIS extract of {products} contraceptives at {sites} sites, {first_month} to {last_month}, {rows} rows. Côte d'Ivoire has used AI forecasting since 2021 with Pendulum, supported by USAID and the Gates Foundation; a 2023 malaria proof of concept [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (a reported result, not an independent finding).

## Method
Rows are flagged, not fixed: {fake_zero} fake zeros, {all_zero} all-zero, {censored} censored, {balance} balance errors; {valid_y} valid targets remain. Direct multi-horizon LightGBM with p50/p90 quantiles, backtested at T-9, T-6 and T-3. Labels use months of stock and p90. Greedy within-district transfers use **SYNTHETIC** expiry dates. Briefs by {llm_model} pass a number, range and format guardrail or fall back to a template ({brief_status}). The app makes no LLM calls.

## Results
{results_table}

All models share the same rows; the p90 coverage target is about 90%.

## Limitations
Censored demand understates true need, and `stock_stockout_days` is 0 in every row. Fake-zero rules are heuristics. Expiry data is SYNTHETIC. No supplier lead times or order cycles. 2016 to 2019 data from one country and one commodity class (contraceptives, not all essential medicines). Transfers reach {served} of {recipients} series in need; {single_site} of {districts} districts have one site.

## Bias
{bias_text} `site_type` does not support a rural/urban claim. The guardrail stops invented numbers but not every misplaced one, so a passed brief is not a verified one. Every transfer needs human review.

## Next
DHIS2 and eLMIS integration, Nigeria's NHLMIS, real batch and expiry data, lead-time-aware reorder logic.

## Run it
Python 3.11. Put the CSVs in `data/raw/` and optionally set `GEMINI_API_KEY` in `.env`, then `pip install -r requirements.txt`, `python -m src.pipeline --all`, `streamlit run app/Home.py`.
