import cloudscraper
from bs4 import BeautifulSoup
import wikipediaapi
import ollama
import re
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=ResourceWarning)

# ================== MODÈLES ==================
MODEL_REASONING = "llama3.1"   # IA 1 : raisonnement
MODEL_FACTCHECK = "mistral"    # IA 2 : référence (plus prudente)

scraper = cloudscraper.create_scraper()

wiki = wikipediaapi.Wikipedia(
    language="fr",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="FactCheckerBot/1.0 (contact: ton_email@example.com)"
)

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "est", "sont", "ne", "pas",
    "que", "qui", "dans", "sur", "avec", "pour", "par", "ce", "cela"
}

CAUSAL_WORDS = [
    "car", "parce que", "en raison", "provoque", "résulte", "cause"
]

# --------------------------------------------------

def extract_concepts(claim):
    words = re.findall(r"\b[a-zA-Zéèêàçùîôû]+\b", claim.lower())
    return list({w for w in words if w not in STOPWORDS and len(w) > 3})[:5]

# --------------------------------------------------

def search_wikipedia(concepts):
    print("📚 Recherche encyclopédique...")
    texts = []

    for concept in concepts:
        page = wiki.page(concept)
        if page.exists():
            print(f"📖 Page trouvée : {page.title}")
            texts.append(page.text[:5000])

    return texts

# --------------------------------------------------

def relevance_score(text, concepts):
    text_lower = text.lower()
    counter = Counter(text_lower.split())

    score = sum(counter[c] for c in concepts)
    causal_bonus = sum(1 for w in CAUSAL_WORDS if w in text_lower)

    return score + causal_bonus * 2

# --------------------------------------------------
# IA 1 — raisonnement (inchangée dans l’esprit)

def final_verdict(original, concepts, context):
    print("⚖️ Analyse IA 1 (raisonnement)...")

    prompt = f"""
    CONTEXTE (encyclopédique) :
    {context}

    En te basant STRICTEMENT sur ce contexte :
    - explique le mécanisme réel lié aux concepts : {', '.join(concepts)}
    - indique si l'affirmation est vraie ou fausse
    - si l'information est insuffisante, dis-le clairement

    AFFIRMATION : "{original}"

    Réponse courte, factuelle et causale.
    """

    res = ollama.chat(
        model=MODEL_REASONING,
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]

# --------------------------------------------------
# IA 2 — fact-checking strict (résultat attendu)

def expected_result_fact_check(claim):
    print("🧪 Fact-checking IA 2 (référence)...")

    prompt = f"""
Tu es un système de fact-checking très strict.

AFFIRMATION :
"{claim}"

Règles :
- Réponds UNIQUEMENT par l’une des valeurs suivantes :
  RESULTAT_ATTENDU: VRAI
  RESULTAT_ATTENDU: FAUX
  RESULTAT_ATTENDU: INCERTAIN
- N’ajoute aucune justification
- N’ajoute aucun autre texte
- Si le doute existe, réponds INCERTAIN
"""

    res = ollama.chat(
        model=MODEL_FACTCHECK,
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]

# --------------------------------------------------

def main():
    print("\n🛡️ CORRECTEUR DE VÉRITÉ — DOUBLE IA\n")

    claim = input("👉 Entre une affirmation : ")

    concepts = extract_concepts(claim)
    print(f"🧠 Concepts détectés : {concepts}")

    wiki_texts = search_wikipedia(concepts)

    if not wiki_texts:
        print("❌ Aucune information encyclopédique trouvée.")
        return

    valid_texts = []

    for text in wiki_texts:
        score = relevance_score(text, concepts)
        if score >= 2:   # seuil assoupli
            valid_texts.append(text)

    if not valid_texts:
        print("❌ Informations trouvées mais insuffisantes.")
        return

    result_ia1 = final_verdict(claim, concepts, "\n\n".join(valid_texts))
    result_expected = expected_result_fact_check(claim)

    print("\n" + "=" * 20 + " IA 1 : RAISONNEMENT " + "=" * 20)
    print(result_ia1)

    print("\n" + "=" * 18 + " IA 2 : RÉSULTAT ATTENDU " + "=" * 18)
    print(result_expected)

    if ("VRAI" in result_ia1 and "VRAI" in result_expected) or \
       ("FAUX" in result_ia1 and "FAUX" in result_expected):
        print("\n✅ Concordance : l’IA 1 donne le bon résultat")
    else:
        print("\n⚠️ Désaccord : l’IA 1 est potentiellement erronée")

    print("=" * 60)

# --------------------------------------------------

if __name__ == "__main__":
    main()

