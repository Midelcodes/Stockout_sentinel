# Stockout Sentinel

Decision-support prototype: forecasts facility-level contraceptive demand, flags stockout risk, suggests transfers within a district, and explains each facility to a district pharmacist in English or French. **Historical data as of {as_of}. Demonstration only; not for operational use.** {live_line}

**Headline:** {headline}

## Problem
Facilities run out of contraceptives while stock sits nearby, and logistics capacity is under strain as external funding falls. Systems usually plan from a 3-month average monthly consumption (AMC). This project tests whether a model does better, and reports it plainly if it does not.

## Data
[USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data): an extract from Côte d'Ivoire's eLMIS covering {products} contraceptives at {sites} sites, {first_month} to {last_month}, {rows} reported rows. July to September 2019 were withheld by the competition. Raw data is not in this repo. Côte d'Ivoire's Ministry of Health has used AI forecasting since 2021 with Pendulum, supported by USAID and the Gates Foundation; a 2023 malaria-commodity proof of concept [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (a reported result, not an independent finding).

## Method
- **Cleaning:** rows are flagged, never silently fixed: fake zeros ({fake_zero}), all-zero rows ({all_zero}), censored stockout months ({censored}) and balance errors ({balance}). Missing months stay NaN. {valid_y} rows keep a valid target.
- **Forecast:** direct multi-horizon LightGBM (1 to 3 months) with Tweedie point and p50/p90 quantile models, scored on rolling origins at T-9, T-6 and T-3. Settings were fixed before seeing results.
- **Stock labels:** months of stock from p50; stockout risk from p90 over the lead time.
- **Redistribution:** greedy, within district and product. Expiry dates are **SYNTHETIC**.
- **Briefs:** written by {llm_model} from a facts-only JSON. A guardrail rejects any number not in the facts, a wrong product range or a brief of the wrong shape, retries, then falls back to a template. Status: {brief_status}. The deployed app makes no LLM calls.

## Results
{results_table}

{rows_note} p90 coverage target is about 90%. {shortfall_note}

## Limitations
Censored demand understates true need, and `stock_stockout_days` is 0 in every row, so only months ending at zero stock are detected. Fake-zero rules are heuristics. Expiry data is SYNTHETIC. No supplier lead times or order cycles. 2016 to 2019 data from one country and one commodity class (contraceptives, not all essential medicines). {single_site} of {districts} districts have one site, so transfers reach only {served} of {recipients} sites in need.

## Bias
{bias_table}

Regional WAPE runs from {region_best} to {region_worst} (regions with at least {region_min_rows} scored rows). `site_type` separates only hospitals, health centers and university institutes, so no rural/urban gap is claimed. LLM risk: the guardrail catches invented numbers but not a real number attached to the wrong product, so a "passed" brief is not a verified one. Every transfer needs human review.

## Next
DHIS2 and eLMIS integration, Nigeria's national logistics system (NHLMIS), real batch and expiry data, and lead-time-aware reorder logic.

## Run it
Python 3.11. Put the raw CSVs in `data/raw/`, and optionally set `GEMINI_API_KEY` in `.env` (see `.env.example`).
```
pip install -r requirements.txt
python -m src.pipeline --all
streamlit run app/Home.py
```
