import re
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

# ---------------------------
# Load spaCy model
# ---------------------------
nlp = spacy.load("en_core_web_sm")

# Add EntityRuler BEFORE spaCy's default NER
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Custom financial + quarter patterns
patterns = [
    {"label": "QUARTER", "pattern": "Q1"},
    {"label": "QUARTER", "pattern": "Q2"},
    {"label": "QUARTER", "pattern": "Q3"},
    {"label": "QUARTER", "pattern": "Q4"},
    {"label": "QUARTER", "pattern": "quarter"},
    {"label": "QUARTER", "pattern": "first quarter"},
    {"label": "QUARTER", "pattern": "second quarter"},
    {"label": "QUARTER", "pattern": "third quarter"},
    {"label": "QUARTER", "pattern": "fourth quarter"},
    {"label": "FIN_METRIC", "pattern": "EPS"},
]
ruler.add_patterns(patterns)

# ---------------------------
# Label mapping (expand short codes to full forms)
# ---------------------------
label_mapping = {
    "GPE": "Geo-Political Entity",
    "PERSON": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Financial Amount",
    "PERCENT": "Percentage",
    "CARDINAL": "Cardinal Number",
    "ORDINAL": "Ordinal Number",
    "QUARTER": "Quarter Reference",
    "FIN_METRIC": "Earnings Per Share",
}

# ---------------------------
# UNIVERSAL FUNCTION
# ---------------------------
def extract_entities(text):
    doc = nlp(text)
    final_entities = []

    for ent in doc.ents:
        full_label = label_mapping.get(ent.label_, ent.label_)

        # Override: ensure quarters are always Quarter Reference
        if ent.text.lower() in [
            "quarter", "first quarter", "second quarter", "third quarter", "fourth quarter"
        ] or re.fullmatch(r"Q[1-4]", ent.text):
            final_entities.append((ent.text, "Quarter Reference", "Custom Override"))
        # EPS override
        elif ent.text.upper() == "EPS":
            final_entities.append((ent.text, "Earnings Per Share", "Custom Override"))
        # "3.12" without $ should be Cardinal Number, not Financial Amount
        elif re.fullmatch(r"\d+(\.\d+)?", ent.text):
            final_entities.append((ent.text, "Cardinal Number", "Custom Override"))
        else:
            final_entities.append((ent.text, full_label, "spaCy"))

    # Regex: capture ONLY $ amounts as Financial Amounts
    for amt in re.findall(r"\$[0-9]+(?:\.[0-9]+)?(?:\s?(?:billion|million))?", text, flags=re.I):
        if not any(e[0] == amt for e in final_entities):
            final_entities.append((amt, "Financial Amount", "Custom Regex"))

    df = pd.DataFrame(final_entities, columns=["Entity", "Label", "Source"])
    return df

# ---------------------------
# MASTER TEXT CONTAINER
# ---------------------------
text = """  """

# Run extraction
df = extract_entities(text)

# Display neatly styled table
display(df.style.set_table_styles(
    [{'selector': 'th', 'props': [('border', '1px solid black'), ('background-color', '#f2f2f2')]},
     {'selector': 'td', 'props': [('border', '1px solid black')]}]
))