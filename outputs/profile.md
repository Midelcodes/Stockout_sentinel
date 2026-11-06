# Data profile

## Schema check

All headers match the spec.

## Files

### Train.csv

35,753 rows, encoding `utf-8-sig`

| column | dtype | nulls | distinct |
| --- | --- | --- | --- |
| year | int64 | 0 | 4 |
| month | int64 | 0 | 12 |
| region | str | 0 | 20 |
| district | str | 0 | 80 |
| site_code | str | 0 | 155 |
| product_code | str | 0 | 11 |
| stock_initial | int64 | 0 | 798 |
| stock_received | int64 | 0 | 259 |
| stock_distributed | int64 | 0 | 338 |
| stock_adjustment | int64 | 0 | 389 |
| stock_end | int64 | 0 | 804 |
| average_monthly_consumption | int64 | 0 | 313 |
| stock_stockout_days | int64 | 0 | 1 |
| stock_ordered | float64 | 763 | 333 |

### service_delivery_site_data.csv

156 rows, encoding `utf-8-sig`

| column | dtype | nulls | distinct |
| --- | --- | --- | --- |
| site_code | str | 0 | 156 |
| site_type | str | 0 | 3 |
| site_region | str | 0 | 20 |
| site_district | str | 0 | 81 |
| site_latitude | float64 | 0 | 156 |
| site_longitude | float64 | 0 | 156 |

### product.csv

11 rows, encoding `utf-8-sig`

| column | dtype | nulls | distinct |
| --- | --- | --- | --- |
| product_code | str | 0 | 11 |
| product_type | str | 0 | 7 |
| product_name | str | 0 | 11 |

### data_dictionary.csv

48 rows, encoding `utf-8-sig`

| column | dtype | nulls | distinct |
| --- | --- | --- | --- |
| File  | str | 0 | 4 |
| Field | str | 0 | 46 |
| Description | str | 0 | 48 |

## Data dictionary

| File  | Field | Description |
| --- | --- | --- |
| contraceptive_logistics_data.csv | year | The year of the observation |
| contraceptive_logistics_data.csv | month | The month of the observation |
| contraceptive_logistics_data.csv | region | The higher level geographical area within Côte d’Ivoire |
| contraceptive_logistics_data.csv | district | The lower level geographical area within Côte d’Ivoire |
| contraceptive_logistics_data.csv | site_code | The actual health service delivery site |
| contraceptive_logistics_data.csv | product_code | The unique identifier for each commodity |
| contraceptive_logistics_data.csv | stock_initial | Stock in hand of previous period. This is quantified in dispensing units.  |
| contraceptive_logistics_data.csv | stock_received | Total quantity received in last period. This is quantified in dispensing units.  |
| contraceptive_logistics_data.csv | stock_distributed* | Quantity dispensed/consumed in last reporting period. This is quantified in dispensing units. Note: This is the target variable.  |
| contraceptive_logistics_data.csv | stock_adjustment | All kinds of losses/ adjustments made at the facility |
| contraceptive_logistics_data.csv | stock_end | Current physical count of stock on hand. This is quantified in dispensing units. |
| contraceptive_logistics_data.csv | average_monthly_consumption | Average monthly consumption, for last three months. This is quantified in dispensing units.  Note: Average monthly consumption in the system actually provides an average of the most recently available Normalized Consumption values. Normalized Consumption is the stock_distributed value adjusted to reflect any stock_stockout_days. Because months are approximated to have 30 days, one which had 15 stockout days would yield a Normalized Consumption of exactly twice its Monthly Consumption. |
| contraceptive_logistics_data.csv | stock_stockout_days | Total number of days facility was out of stock. |
| contraceptive_logistics_data.csv | stock_ordered | The requested quantity. This is quantified in dispensing units. |
| contraceptive_case_data_(annual, monthly).csv | implant_women_old | Number of women receiving implants during the month, who used contraception previously |
| contraceptive_case_data_(annual, monthly).csv | implant_women_new | Number of women receiving implants during the month, who did not use contraception previously |
| contraceptive_case_data_(annual, monthly).csv | injection2_women_old | Number of women receiving injection (2 months) during the month, who used contraception previously |
| contraceptive_case_data_(annual, monthly).csv | injection2_women_new | Number of women receiving injection (2 months) during the month, who did not use contraception previously |
| contraceptive_case_data_(annual, monthly).csv | injection3_women_old | Number of women receiving injection (3 months) during the month, who used contraception previously |
| contraceptive_case_data_(annual, monthly).csv | injection3_women_new | Number of women receiving injection (3 months) during the month, who did not use contraception previously |
| contraceptive_case_data_(annual, monthly).csv | pill_women_old | Number of women receiving oral contraceptives during the month, who used contraception previously |
| contraceptive_case_data_(annual, monthly).csv | pill_women_new | Number of women receiving oral contraceptives during the month, who did not use contraception previously |
| contraceptive_case_data_(annual, monthly).csv | iud_women_old | Number of women receiving IUDs during the month, who used contraception previously |
| contraceptive_case_data_(annual, monthly).csv | iud_women_new | Number of women receiving IUDs during the month, who did not use contraception previously |
| contraceptive_case_data_(annual, monthly).csv | iud_number_dispensed | Number of IUDs given to women during that month |
| contraceptive_case_data_(annual, monthly).csv | implant_number_dispensed | Number of implants given to women during that month |
| contraceptive_case_data_(annual, monthly).csv | injection2_number_dispensed | Number of injection (2 months) given to women during that month |
| contraceptive_case_data_(annual, monthly).csv | injection3_number_dispensed | Number of injection (3 months) given to women during that month |
| contraceptive_case_data_(annual, monthly).csv | pill_number_dispensed | Number of oral contraceptives (one cycle) given to women during that month |
| contraceptive_case_data_(annual, monthly).csv | iud_number_received | Number of IUD received in that month |
| contraceptive_case_data_(annual, monthly).csv | implant_number_received | Number of implants received in that month |
| contraceptive_case_data_(annual, monthly).csv | injection2_number_received | Number of injection (2 months) received in that month |
| contraceptive_case_data_(annual, monthly).csv | injection3_number_received | Number of injection (3 months) received in that month |
| contraceptive_case_data_(annual, monthly).csv | pill_number_received | Number of oral contraceptive (one cycle) received in that month |
| contraceptive_case_data_(annual, monthly).csv | iud_stock_end | The amount of IUD stock available at the end of that month |
| contraceptive_case_data_(annual, monthly).csv | implant_stock_end | The amount of implant stock available at the end of that month |
| contraceptive_case_data_(annual, monthly).csv | injection2_stock_end | The amount of injection (2 months) stock available at the end of that month |
| contraceptive_case_data_(annual, monthly).csv | injection3_stock_end | The amount of injection (3 months) stock available at the end of that month |
| contraceptive_case_data_(annual, monthly).csv | pill_stock_end | The amount of pill stock available at the end of that month |
| service_delivery_site_data.csv | site_code | The unique code of the actual health service delivery site |
| service_delivery_site_data.csv | site_type | The kind of health delivery location including hospital, health center and university hospital/national institute |
| service_delivery_site_data.csv | site_region | The higher level geographical area within Cote d’Ivoire |
| service_delivery_site_data.csv | site_district | The lower level geographical area within Cote d’Ivoire |
| service_delivery_site_data.csv | site_latitude | The site's latitudinal coordinates |
| service_delivery_site_data.csv | site_longitude | The site's longitudinal coordinates |
| product.csv | product_code | The unique product code |
| product.csv | product_type | The contraceptive category for each product |
| product.csv | product_name | The product name, dose, and presentation |

Columns not named anywhere in the dictionary: none

## Summary

| metric | observed | spec |
| --- | --- | --- |
| rows | 35,753 |  |
| sites | 155 | 156 |
| products | 11 | 11 |
| regions | 20 |  |
| districts | 80 |  |
| first month | 2016-01 | 2016-01 |
| last month | 2019-06 | 2019-09 |
| rows with unreadable year or month | 0 |  |
| duplicate site x product x month rows | 0 |  |
| sites listed under more than one district | 0 |  |
| districts with a single site | 50 |  |

## Series (site x product)

Series count: 1,343

| distribution | min | p25 | median | p75 | max | mean |
| --- | --- | --- | --- | --- | --- | --- |
| reported months per series | 1.0 | 11.0 | 32.0 | 41.0 | 42.0 | 26.6 |

Missing months are counted apart: internal gaps fall between a series' first and last report; trailing months fall after its last report, up to the dataset's last month.

| metric | value |
| --- | --- |
| series with internal gaps | 330 of 1,343 (24.6%) |
| internal missing months, total | 2,353 |
| internal missing months per series, median | 0.0 |
| internal missing months per series, p90 | 5.0 |
| internal missing months per series, max | 35.0 |
| series ending before the last month | 311 of 1,343 (23.2%) |
| trailing missing months, total | 6,062 |

## Data quality

| check | value |
| --- | --- |
| `stock_distributed` = 0 | 17,738 of 35,753 (49.6%) |
| of those, `stock_initial` > 0 | 6,741 of 17,738 (38.0%) |
| `stock_stockout_days` > 0 | 0 of 35,753 (0.0%) |
| `stock_stockout_days` null | 0 of 35,753 (0.0%) |
| stock balance off by more than 1 | 0 of 35,753 (0.0%) |
| stock balance off by any amount | 0 of 35,753 (0.0%) |
| off by more than 1 if adjustment were subtracted instead | 1,740 of 35,753 (4.9%) |
| rows skipped by the balance check (a stock field is null) | 0 |
| rows with every stock field 0 | 9,820 of 35,753 (27.5%) |
| series with no stock activity in any month | 354 of 1,343 (26.4%) |
| recorded AMC within 1 unit of the plain mean of months t-2 to t | 32,486 of 32,531 (99.9%) |

## Joins

Product codes with no row in product.csv: 0

Site codes with no row in service_delivery_site_data.csv: 0

Sites in service_delivery_site_data.csv with no logistics rows: 1 (C5076)
