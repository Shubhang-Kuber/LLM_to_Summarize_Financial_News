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

# Custom financial + location patterns
patterns = [
    {"label": "QUARTER", "pattern": [{"TEXT": {"REGEX": r"Q[1-4]"}}]},
    {"label": "FIN_METRIC", "pattern": [{"TEXT": "EPS"}]},
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

    # spaCy + custom overrides
    for ent in doc.ents:
        full_label = label_mapping.get(ent.label_, ent.label_)
        final_entities.append((ent.text, full_label, "spaCy/Custom"))

    # Regex for financial amounts (extra safeguard)
    for amt in re.findall(r"\$[0-9]+(?:\.[0-9]+)?(?:\s?(?:billion|million))?", text, flags=re.I):
        if not any(e[0] == amt for e in final_entities):
            final_entities.append((amt, "Financial Amount", "Custom Regex"))

    # Convert to DataFrame
    df = pd.DataFrame(final_entities, columns=["Entity", "Label", "Source"])
    return df

# ---------------------------
# MASTER TEXT CONTAINER (replace with your own)
# ---------------------------
text = """Tesla reported record third-quarter deliveries on October 2, 2025,
with 435,000 vehicles shipped. Analysts expect Q4 revenue to cross $25 billion,
while CEO Elon Musk hinted at expanding production in Berlin. EPS for the quarter was $3.12."""

# Run extraction
df = extract_entities(text)

# Display neatly styled table
display(df.style.set_table_styles(
    [{'selector': 'th', 'props': [('border', '1px solid black'), ('background-color', '#f2f2f2')]},
     {'selector': 'td', 'props': [('border', '1px solid black')]}]
))
