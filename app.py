from flask import Flask, request, render_template
import pandas as pd
import stanza
import unicodedata
import os

# Load Stanza Polish model
stanza.download("pl")
nlp = stanza.Pipeline(lang="pl", processors="tokenize,lemma", use_gpu=False)

# Load tag list from CSV
file_path = os.path.join(os.getcwd(), "TagListPL.csv")
df = pd.read_csv(file_path, encoding="utf-8", sep=",")
tag_dict = dict(zip(df["word"], df["tag"]))

# Manually correct known lemmatization issues
manual_lemma_corrections = {
    "dostałem": "dostać",
    "dostałeś": "dostać",
    "dostaliśmy": "dostać",
    "dostaliście": "dostać"
}

# Conditional endings that should be removed
conditional_endings = {"bym", "byś", "by", "byśmy", "byście"}

def normalize_text(text):
    """Normalize text to avoid encoding issues."""
    return unicodedata.normalize("NFKC", text)

def lemmatize_and_tag(text):
    """Lemmatize words and match them with tags, ensuring conditionals are reduced to their base verb."""
    text = normalize_text(text)
    doc = nlp(text)
    tagged_output = []
    
    words = [(word.text.lower(), manual_lemma_corrections.get(word.text.lower(), word.lemma.lower()), word.upos) 
             for sentence in doc.sentences for word in sentence.words]

    i = 0
    while i < len(words):
        word_text, lemma, pos = words[i]

        # 🛑 Remove wrongly inserted "być"
        if lemma == "być":
            i += 1  # Skip "być"
            continue

        # ✅ Fix Conditional Verbs: Convert to base form and remove "by"
        if i > 0 and word_text in conditional_endings and words[i - 1][2] == "VERB":
            base_verb = words[i - 1][1]  # Extract the verb lemma
            tagged_output[-1] = (base_verb, tag_dict.get(base_verb, "UNKNOWN"))  # Replace last verb entry
            i += 1  # Skip "by" entirely
            continue

        # ✅ Fix Reflexive Verbs (zrobiłem się → zrobić się)
        if word_text == "się":
            if i > 0 and words[i - 1][2] == "VERB":
                base_verb = words[i - 1][1]
                reflexive_verb = base_verb + " się"
                if reflexive_verb in tag_dict:
                    tag = tag_dict[reflexive_verb]
                    tagged_output[-1] = (reflexive_verb, tag)  # Replace last verb entry
                else:
                    tagged_output[-1] = (reflexive_verb, "UNKNOWN")
                i += 1
                continue

        # Default: Process normally
        tag = tag_dict.get(lemma, "UNKNOWN")
        tagged_output.append((lemma, tag))
        i += 1

    return tagged_output

# ✅ FIX: Make sure Flask app is defined!
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        result = lemmatize_and_tag(text)
        return render_template("index.html", text=text, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)















