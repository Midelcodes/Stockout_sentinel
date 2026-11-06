# Stockout Sentinel

Decision-support prototype: forecasts facility-level contraceptive demand, flags stockout risk, suggests transfers within a district, and explains each facility to a district pharmacist in English or French. **Historical data as of June 2019. Demonstration only; not for operational use.** 

**Headline:** LightGBM beats AMC-3 on forecast error: WAPE 0.593 to 0.545 (-8.0%, 95% interval -15.2% to -2.6%). Its shortfall share is higher than AMC-3's (33.2% against 31.8%), and p90 coverage is 85.9%.

## Problem
Facilities run out of contraceptives while stock sits nearby, and logistics capacity is under strain as external funding falls. Systems usually plan from a 3-month average monthly consumption (AMC). This project tests whether a model does better, and reports it plainly if it does not.

## Data
[USAID's Intelligent Forecasting Challenge](https://zindi.africa/competitions/usaids-intelligent-forecasting-challenge-model-future-contraceptive-use/data): an extract from Côte d'Ivoire's eLMIS covering 11 contraceptives at 155 sites, January 2016 to June 2019, 35,753 reported rows. July to September 2019 were withheld by the competition. Raw data is not in this repo. Côte d'Ivoire's Ministry of Health has used AI forecasting since 2021 with Pendulum, supported by USAID and the Gates Foundation; a 2023 malaria-commodity proof of concept [reported](https://community.dhis2.org/t/leveraging-dhis2-data-to-create-ai-driven-health-facility-forecasts-in-cote-d-ivoire/58136) an average 37% reduction in forecast error (a reported result, not an independent finding).

## Method
- **Cleaning:** rows are flagged, never silently fixed: fake zeros (1,228), all-zero rows (9,820), censored stockout months (2,003) and balance errors (0). Missing months stay NaN. 22,702 rows keep a valid target.
- **Forecast:** direct multi-horizon LightGBM (1 to 3 months) with Tweedie point and p50/p90 quantile models, scored on rolling origins at T-9, T-6 and T-3. Settings were fixed before seeing results.
- **Stock labels:** months of stock from p50; stockout risk from p90 over the lead time.
- **Redistribution:** greedy, within district and product. Expiry dates are **SYNTHETIC**.
- **Briefs:** written by `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite` from a facts-only JSON. A guardrail rejects any number not in the facts, a wrong product range or a brief of the wrong shape, retries, then falls back to a template. Status: 74.5% passed, 17.4% passed_after_retry, 8.1% template_fallback. The deployed app makes no LLM calls.

## Results
| Model | WAPE | MASE | Under-forecast | Shortfall | p90 coverage | Rows scored |
| --- | --- | --- | --- | --- | --- | --- |
| **LightGBM** | 0.545 | 1.001 | 41.8% | 33.2% | 85.9% | 5,649 |
| AMC-3 | 0.593 | 1.025 | 41.4% | 31.8% | n/a | 5,649 |
| eLMIS AMC | 0.575 | 1.036 | 44.1% | 36.2% | n/a | 5,649 |
| Naive | 0.682 | 1.161 | 42.4% | 35.9% | n/a | 5,649 |
| Seasonal naive | 0.719 | 1.245 | 42.5% | 40.8% | n/a | 5,649 |

All models are scored on the same 5,649 rows. p90 coverage target is about 90%. eLMIS AMC's WAPE sits close to the model's partly because it forecasts low: WAPE rewards that, while the shortfall column shows its cost.

## Limitations
Censored demand understates true need, and `stock_stockout_days` is 0 in every row, so only months ending at zero stock are detected. Fake-zero rules are heuristics. Expiry data is SYNTHETIC. No supplier lead times or order cycles. 2016 to 2019 data from one country and one commodity class (contraceptives, not all essential medicines). 50 of 80 districts have one site, so transfers reach only 89 of 308 sites in need.

## Bias
| Slice | LightGBM WAPE | AMC-3 WAPE | Shortfall | p90 coverage | Rows |
| --- | --- | --- | --- | --- | --- |
| Volume tier: lowest | 0.698 | 0.726 | 51.0% | 84.5% | 1,720 |
| Volume tier: middle | 0.578 | 0.620 | 35.8% | 86.5% | 1,951 |
| Volume tier: highest | 0.494 | 0.550 | 27.9% | 86.4% | 1,978 |
| Reporting-quality tier: best | 0.479 | 0.501 | 30.0% | 84.8% | 1,965 |
| Reporting-quality tier: middle | 0.559 | 0.645 | 34.0% | 85.8% | 1,976 |
| Reporting-quality tier: worst | 0.628 | 0.661 | 37.1% | 87.1% | 1,708 |

Regional WAPE runs from 0.439 to 0.842 (regions with at least 50 scored rows). `site_type` separates only hospitals, health centers and university institutes, so no rural/urban gap is claimed. LLM risk: the guardrail catches invented numbers but not a real number attached to the wrong product, so a "passed" brief is not a verified one. Every transfer needs human review.

## Next
DHIS2 and eLMIS integration, Nigeria's national logistics system (NHLMIS), real batch and expiry data, and lead-time-aware reorder logic.

## Run it
Python 3.11. Put the raw CSVs in `data/raw/`, and optionally set `GEMINI_API_KEY` in `.env` (see `.env.example`).
```
pip install -r requirements.txt
python -m src.pipeline --all
streamlit run app/Home.py
```
