# Using LLMs safely on health programme data: a playbook

A practical guide for programme, finance and supply-chain teams who want to use large language models (LLMs) such as Gemini, Claude or ChatGPT in their day-to-day work. It comes from building [Stockout Sentinel](https://stockoutsentinel.up.railway.app/), where an LLM writes stock briefs for district pharmacists, and from what went wrong along the way.

Author: Joshua Osayamen. Licence: MIT, like the rest of this repository. Reuse and adapt freely.

## The one idea

**Let code compute the numbers. Let the LLM only phrase them. Then check.**

LLMs are fluent, not reliable with figures. In this project the model wrote short briefs from facts it had been given, and an automatic check tested every draft for invented numbers, wrong ranges and the wrong format. Even with up to three attempts, each told exactly what was wrong, 51.6% of briefs still failed and were replaced by a plain template. A fluent brief with one wrong stock figure is worse than a plain correct one, because people trust fluent text.

## What LLMs are good and bad at in programme work

| Good at | Bad at |
| --- | --- |
| Turning computed facts into plain language for a specific reader | Arithmetic, totals, percentages, dates |
| Translating and adapting tone (e.g. English to French for a pharmacist) | Knowing your country's current policy, prices or stock levels |
| Summarising long reports you paste in, and drafting first versions | Telling you when it does not know |
| Extracting fields from messy text into a table you then check | Consistency: the same prompt can give different answers |
| Suggesting questions, risks and options to consider | Making decisions that need accountability |

## Seven rules

1. **Compute first, phrase second.** Do sums, forecasts and rankings in a spreadsheet or code. Give the LLM only the results.
2. **Give facts, not raw data.** Send the few numbers the reader needs, labelled, not the whole dataset. Less input means fewer places to go wrong, and less data leaves your organisation.
3. **Check every number.** Every figure in the output must appear in the facts you supplied. In this project a short program does this automatically (the [`llm_number_guardrail`](../llm_number_guardrail/README.md) package); by hand, tick each number against your source.
4. **Have a fallback.** Decide in advance what happens when the output fails the check: retry once with the list of problems, then use a plain template. Never ship the failed draft.
5. **Measure and publish the failure rate.** Count how often the check fails. Here 51.6% of briefs ended in the template. Reporting that number openly is what makes the rest credible.
6. **A person decides.** LLM output is a suggestion. Transfers, orders, budgets and anything clinical need a named human to approve them.
7. **Keep sensitive data out.** No patient names, IDs or health records in a public LLM tool. Check the provider's terms: free tiers often allow your inputs to be used to improve the product. Follow your national data-protection law (in Nigeria, the Nigeria Data Protection Act 2023).

## Before you paste anything: a data checklist

- [ ] Does it contain names, phone numbers, ID numbers or any patient information? Remove or replace them.
- [ ] Is it confidential to a government partner or grantee? Ask before sharing it with an external tool.
- [ ] Do you know whether this tool's terms allow it to keep or train on your input?
- [ ] Could the same task be done with aggregated figures instead of row-level data? Use those.

## Prompt patterns that work

Each pattern names the reader, gives the facts, and fixes the format.

**Brief from computed facts**
```
You write for a district pharmacist. Use only the facts below. Copy every
number exactly as written; do no arithmetic and add no numbers. Write 3 to 5
sentences, then up to 3 bullet actions. Say when a figure is a range.

FACTS:
<paste the labelled figures>
```

**Summarise a report**
```
Summarise this report for a programme officer in 150 words: the decision it
supports, the 3 main findings with their page numbers, and anything it says is
uncertain. Quote figures exactly and give the page for each.

REPORT:
<paste the text>
```

**Extract a table from messy text**
```
Extract every facility, product and quantity mentioned below into a table with
columns facility | product | quantity | sentence quoted. Leave a cell empty if
the text does not state it. Do not infer.
```
Then check a sample of rows against the source.

**Translate for a francophone reader**
```
Translate into French for a district pharmacist in Côte d'Ivoire. Keep every
number exactly as written, use French decimal commas, and keep product names
unchanged.
```

## Checking an output in two minutes

1. Circle every number. Find each one in your facts. Any you cannot find: reject.
2. Check each number sits beside the right item (a correct figure attached to the wrong product is the most common slip that passes a quick read).
3. Check nothing important from the facts was left out.
4. Read it as the intended reader would: is the action clear, and is uncertainty stated?

## Judging an AI tool before you adopt it

Ask the team that built it:

- **What does it beat?** Compare against the method people use today (here, the 3-month average consumption), on the same data, with the result stated in plain units.
- **Where does it do worse?** Ask for errors broken down by site size, data quality and region. In this project error was highest at low-volume sites and sites that report poorly, which are often the ones most in need.
- **How often is it wrong, and what happens then?** Look for a published failure rate and a defined fallback.
- **What does it cost to run?** A tool a ministry cannot afford after the pilot is not a solution. Precomputing outputs can keep running costs near zero.
- **Who checks it?** Every suggestion should have a named human reviewer.

## A 60-minute team workshop

| Minutes | Activity |
| --- | --- |
| 0–10 | What LLMs are good and bad at (the table above), with one live example of a wrong number. |
| 10–25 | Hands-on: each person writes a brief from the same five facts using the "brief from computed facts" pattern. |
| 25–40 | Swap and check each other's outputs with the two-minute check. Count the errors found. |
| 40–50 | The data checklist: what may and may not go into a public tool, with examples from our own work. |
| 50–60 | Each person names one task they will try this week, and how they will check it. |

## Glossary

- **LLM:** large language model, a system that generates text from a prompt.
- **Guardrail:** an automatic check on LLM output before anyone sees it.
- **Fallback:** the safe output used when a draft fails the check.
- **Retrieval:** finding the passages of your own documents that answer a question, so answers can cite a source. See the "Ask the docs" page in the app.
- **p50 / p90:** a likely amount, and an amount demand should stay under about 9 times in 10.
- **AMC:** average monthly consumption, the usual basis for reorders in logistics systems.
