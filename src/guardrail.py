"""Number guardrail for LLM briefs, and the deterministic template it falls back to."""
import json
import re
from decimal import Decimal

SPACES = "    "
NUMBER_PATTERNS = {
    "en": re.compile(rf"(?<!\d)\d{{1,3}}(?:[,{SPACES}]\d{{3}})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?"),
    "fr": re.compile(rf"(?<!\d)\d{{1,3}}(?:[.{SPACES}]\d{{3}})+(?:,\d+)?(?!\d)|\d+(?:[.,]\d+)?"),
}
DOT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
FIELD_NAMES = re.compile(
    r"\b(stock_end|months_of_stock|days_to_stockout|low_data_quality|label_counts|product_name|other_site)\b"
)
RANGE = re.compile(r"(\d+)\s*(?:to|à|–)\s*(\d+)")
NULL_VALUE = re.compile(r"\bnull\b", re.I)
ENGLISH_IN_FRENCH = re.compile(r"\b(p50 to p90|human review|expiry|stockout|units|June)\b", re.I)


def parse_number(token, lang):
    """Read one number token. English decimals use '.', French decimals use ','."""
    token = re.sub(f"[{SPACES}]", "", token)
    if lang == "en":
        return Decimal(token.replace(",", ""))
    if "," in token:
        return Decimal(token.replace(".", "").replace(",", "."))
    if DOT_THOUSANDS.fullmatch(token):
        return Decimal(token.replace(".", ""))
    return Decimal(token)


def extract_numbers(text, lang):
    return [(m.group(), parse_number(m.group(), lang)) for m in NUMBER_PATTERNS[lang].finditer(text)]


def allowed_numbers(facts):
    """Every number anywhere in the serialized FACTS, including inside names and dates."""
    serialized = json.dumps(facts, ensure_ascii=False)
    return {Decimal(token) for token in re.findall(r"\d+(?:\.\d+)?", serialized)}


def find_offending(text, allowed, lang):
    """Numbers in the text that are not in the allowed set, in order, without repeats."""
    offending = []
    for token, value in extract_numbers(text, lang):
        if value not in allowed and token not in offending:
            offending.append(token)
    return offending


def sentence_count(prose, facts):
    """Sentences in the prose, ignoring the periods inside product names and decimals."""
    for product in facts["products"]:
        prose = prose.replace(product["product_name"], "PRODUCT")
    prose = re.sub(r"\d+[.,]\d+", "N", prose)
    return len([s for s in re.split(r"(?<=[.!?])\s+", prose.strip()) if s])


def find_shape_problems(text, facts, lang):
    """Reasons a draft does not fit the brief format, written for the model to fix."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith("- ")]
    prose = " ".join(line for line in lines if not line.startswith("- "))
    problems = []
    sentences = sentence_count(prose, facts)
    if not 3 <= sentences <= 5:
        problems.append(f"the paragraph has {sentences} sentences; it must have 3 to 5")
    if len(bullets) > 3:
        problems.append(f"there are {len(bullets)} bullets; the limit is 3")
    problems += [f"field name in the text: {name}" for name in sorted(set(FIELD_NAMES.findall(text)))]
    if NULL_VALUE.search(text):
        problems.append("the word null appears; leave unknown values out instead")
    if not (facts["transfers_in"] or facts["transfers_out"]) and "transfer" in text.lower():
        problems.append("this site has no transfers, so do not mention any")
    if lang == "fr":
        words = sorted({word.lower() for word in ENGLISH_IN_FRENCH.findall(text)})
        problems += [f"English left in the French text: {word}" for word in words]
    return problems


def find_range_problems(text, facts):
    """Ranges written after a product name that are not that product's p50 to p90."""
    mentions = sorted(
        (match.start(), match.end(), i)
        for i, product in enumerate(facts["products"])
        for match in re.finditer(re.escape(product["product_name"]), text)
    )
    problems = []
    for position, (_, end, index) in enumerate(mentions):
        product = facts["products"][index]
        stop = mentions[position + 1][0] if position + 1 < len(mentions) else len(text)
        valid = set(zip(product["p50"], product["p90"]))
        for low, high in RANGE.findall(text[end:stop]):
            if (int(low), int(high)) not in valid:
                problems.append(
                    f"wrong range for {product['product_name']}: {low} to {high} is not its p50 to p90"
                )
    return problems


def find_problems(text, facts, allowed, lang):
    numbers = [f"number not in FACTS: {n}" for n in find_offending(text, allowed, lang)]
    return numbers + find_shape_problems(text, facts, lang) + find_range_problems(text, facts)


def guarded_brief(facts, lang, generate, max_attempts):
    """Generate until a draft passes, feeding its problems back to the model.

    generate takes None or (draft, problems) and returns a draft, or None if
    the attempt produced nothing usable.
    """
    allowed = allowed_numbers(facts)
    feedback = None
    for attempt in range(1, max_attempts + 1):
        draft = generate(feedback)
        if not draft:
            continue
        problems = find_problems(draft, facts, allowed, lang)
        if not problems:
            status = "passed" if attempt == 1 else "passed_after_retry"
            return {"text": draft, "guardrail_status": status, "attempts": attempt}
        feedback = (draft, problems)
    return {
        "text": template_brief(facts, lang),
        "guardrail_status": "template_fallback",
        "attempts": max_attempts,
    }


TEMPLATE_TEXT = {
    "en": {
        "opening": "Site {site} in {district}, as of {as_of}, has these stock labels: {counts}.",
        "product": (
            "{name} has {stock} on hand against expected demand next month of "
            "{p50} to {p90} units (p50 to p90), about {days} of stock."
        ),
        "no_product": "No product at this site has both current stock and expected demand to compare.",
        "ranges": "Forecasts are ranges, not certainties.",
        "transfers": "Transfers below are suggestions for human review, and their expiry dates are simulated.",
        "quality": "Reporting quality at this site is in the lowest tier, so treat these figures with caution.",
        "receive": "- Review a suggested transfer: receive {quantity} of {name} from {site} (simulated expiry {expiry}).",
        "send": "- Review a suggested transfer: send {quantity} of {name} to {site} (simulated expiry {expiry}).",
        "resupply": "- Check resupply for {name}.",
        "verify": "- Confirm whether the zero report for {name} is a real stockout.",
        "monitor": "- Continue routine stock monitoring.",
    },
    "fr": {
        "opening": "Le site {site} à {district}, à la date de {as_of}, présente ces statuts de stock : {counts}.",
        "product": (
            "{name} : {stock} en stock pour une demande attendue le mois prochain de "
            "{p50} à {p90} unités (p50 à p90), soit environ {days} de stock."
        ),
        "no_product": "Aucun produit de ce site ne permet de comparer stock actuel et demande attendue.",
        "ranges": "Les prévisions sont des fourchettes, pas des certitudes.",
        "transfers": "Les transferts ci-dessous sont des suggestions à valider par une personne, et leurs dates de péremption sont simulées.",
        "quality": "La qualité des rapports de ce site est dans le groupe le plus faible : ces chiffres sont à prendre avec prudence.",
        "receive": "- Examiner un transfert suggéré : recevoir {quantity} de {name} depuis {site} (péremption simulée {expiry}).",
        "send": "- Examiner un transfert suggéré : envoyer {quantity} de {name} vers {site} (péremption simulée {expiry}).",
        "resupply": "- Vérifier le réapprovisionnement de {name}.",
        "verify": "- Confirmer si le zéro déclaré pour {name} est une vraie rupture.",
        "monitor": "- Poursuivre le suivi de routine des stocks.",
    },
}
FRENCH_LABELS = {
    "No current data": "Pas de données récentes",
    "Not stocked": "Non stocké",
    "Ordered, never received": "Commandé, jamais reçu",
    "Unverified zero": "Zéro non vérifié",
    "No expected demand": "Pas de demande attendue",
    "Stockout risk": "Risque de rupture",
    "Understocked": "Sous-stocké",
    "Overstocked": "Surstocké",
    "Adequate": "Adéquat",
}
FRENCH_MONTHS = {
    "January": "janvier", "February": "février", "March": "mars", "April": "avril",
    "May": "mai", "June": "juin", "July": "juillet", "August": "août",
    "September": "septembre", "October": "octobre", "November": "novembre", "December": "décembre",
}


def _count(n, noun, lang):
    """A number with its noun. French treats 0 and 1 as singular, English only 1."""
    singular = {
        "en": {"unit": "unit", "day": "day"},
        "fr": {"unit": "unité", "day": "jour"},
    }[lang][noun]
    plural = {"en": {"unit": "units", "day": "days"}, "fr": {"unit": "unités", "day": "jours"}}[lang][noun]
    one = n in (0, 1) if lang == "fr" else n == 1
    return f"{n} {singular if one else plural}"


def local_month(label, lang):
    if lang == "en":
        return label
    month, year = label.split()
    return f"{FRENCH_MONTHS[month]} {year}"


def _local_label(label, lang):
    return label if lang == "en" else FRENCH_LABELS[label]


def _urgent_products(facts):
    measurable = [p for p in facts["products"] if p["days_to_stockout"] is not None]
    return sorted(measurable, key=lambda p: (p["label"] != "Stockout risk", p["days_to_stockout"]))


def template_sentences(facts, lang):
    """3 to 5 sentences. The transfer and data-quality notes always fit; products give way."""
    text = TEMPLATE_TEXT[lang]
    counts = ", ".join(
        f"{_local_label(label, lang)} {count}" for label, count in facts["label_counts"].items()
    )
    notes = [text["ranges"]]
    if facts["transfers_in"] or facts["transfers_out"]:
        notes.append(text["transfers"])
    if any(p["low_data_quality"] for p in facts["products"]):
        notes.append(text["quality"])

    opening = text["opening"].format(
        site=facts["site_code"], district=facts["district"],
        as_of=local_month(facts["as_of_label"], lang), counts=counts,
    )
    products = [
        text["product"].format(
            name=p["product_name"], stock=_count(p["stock_end"], "unit", lang),
            p50=p["p50"][0], p90=p["p90"][0], days=_count(p["days_to_stockout"], "day", lang),
        )
        for p in _urgent_products(facts)[: 4 - len(notes)]
    ]
    return [opening] + (products or [text["no_product"]]) + notes


def template_brief(facts, lang):
    """A deterministic brief built only from FACTS."""
    text = TEMPLATE_TEXT[lang]
    urgent = _urgent_products(facts)
    transfers = (
        [("receive", t) for t in facts["transfers_in"]]
        + [("send", t) for t in facts["transfers_out"]]
    )
    bullets = [
        text[kind].format(
            quantity=_count(t["quantity"], "unit", lang), name=t["product_name"], site=t["other_site"],
            expiry=t["SYNTHETIC_expiry_date"],
        )
        for kind, t in transfers[:3]
    ]
    at_risk = [p for p in urgent if p["label"] == "Stockout risk"]
    bullets += [text["resupply"].format(name=p["product_name"]) for p in at_risk[: 3 - len(bullets)]]
    unverified = [p for p in facts["products"] if p["label"] == "Unverified zero"]
    bullets += [text["verify"].format(name=p["product_name"]) for p in unverified[: 3 - len(bullets)]]
    prose = " ".join(template_sentences(facts, lang))
    return prose + "\n\n" + "\n".join(bullets or [text["monitor"]])
