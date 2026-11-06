# Stockout Sentinel

Forecasts facility contraceptive demand, flags stockout risk, suggests within-district transfers and briefs a district pharmacist in English or French. **Historical data as of June 2019. Demonstration only; not for operational use.** [Live demo](https://stockoutsentinel.up.railway.app/).

**Headline:** LightGBM beats AMC-3 on forecast error: WAPE 0.593 to 0.545 (-8.0%, 95% interval -15.2% to -2.6%). Its shortfall share is higher than AMC-3's (33.2% against 31.8%), and p90 coverage is 85.9%.

## Problem
Facilities stock out while supply sits nearby, and logistics capacity is under strain as external funding falls. Can a model beat the 3-month average (AMC) logistics systems use today?

## Data
[USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data): a Côte d'Ivoire eLMIS extract of 11 contraceptives at 155 sites, January 2016 to June 2019, 35,753 rows. Côte d'Ivoire has used AI forecasting since 2021 with Pendulum, supported by USAID and the Gates Foundation; a 2023 malaria proof of concept [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (a reported result, not an independent finding).

## Method
Rows are flagged, not fixed: 1,228 fake zeros, 9,820 all-zero, 2,003 censored, 0 balance errors; 22,702 valid targets remain. Direct multi-horizon LightGBM with p50/p90 quantiles, backtested at T-9, T-6 and T-3. Labels use months of stock and p90. Greedy within-district transfers use **SYNTHETIC** expiry dates. Briefs by `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite` pass a number, range and format guardrail or fall back to a template (74.5% passed, 17.4% passed_after_retry, 8.1% template_fallback). The app makes no LLM calls.

## Results
| Model | WAPE | MASE | Under-forecast | Shortfall | p90 coverage | Rows scored |
| --- | --- | --- | --- | --- | --- | --- |
| **LightGBM** | 0.545 | 1.001 | 41.8% | 33.2% | 85.9% | 5,649 |
| AMC-3 | 0.593 | 1.025 | 41.4% | 31.8% | n/a | 5,649 |
| eLMIS AMC | 0.575 | 1.036 | 44.1% | 36.2% | n/a | 5,649 |
| Naive | 0.682 | 1.161 | 42.4% | 35.9% | n/a | 5,649 |
| Seasonal naive | 0.719 | 1.245 | 42.5% | 40.8% | n/a | 5,649 |

All models share the same rows; the p90 coverage target is about 90%.

## Limitations
Censored demand understates true need, and `stock_stockout_days` is 0 in every row. Fake-zero rules are heuristics. Expiry data is SYNTHETIC. No supplier lead times or order cycles. 2016 to 2019 data from one country and one commodity class (contraceptives, not all essential medicines). Transfers reach 89 of 308 series in need; 50 of 80 districts have one site.

## Bias
LightGBM WAPE by site volume, lowest to highest: 0.698, 0.578, 0.494. By reporting quality, best to worst: 0.479, 0.559, 0.628. By region (50+ scored rows): 0.439 to 0.842. `site_type` does not support a rural/urban claim. The guardrail stops invented numbers but not every misplaced one, so a passed brief is not a verified one. Every transfer needs human review.

## Next
DHIS2 and eLMIS integration, Nigeria's NHLMIS, real batch and expiry data, lead-time-aware reorder logic.

## Run it
Python 3.11. Put the CSVs in `data/raw/` and optionally set `GEMINI_API_KEY` in `.env`, then `pip install -r requirements.txt`, `python -m src.pipeline --all`, `streamlit run app/Home.py`.
