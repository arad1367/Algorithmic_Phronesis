
import json
from collections import Counter
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Note by Pejman: Very basic thematic coding proxy script (lexical analysis to simulate thematic coding)
# In reality, thematic coding for A* journals is done manually (e.g., via NVivo or MAXQDA),
# but this script provides a data extraction framework to support human-led coding and 
# visualizes keyword density within theoretical domains

def analyze_interviews(filepath='Interviews.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            interviews = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    print(f"Loaded {len(interviews)} interviews.")

    # 1. Define Theoretical Dictionaries for Directed Thematic Analysis
    # Based on the Theoretical Background section of the paper
    themes = {
        'Algorithmic_Phronesis_Simulation': ['simulate', 'wisdom', 'judgment', 'plausible', 'confidence', 'certainty', 'rhetorical', 'vocabulary', 'fluency'],
        'Relational_Dissonance': ['dissonance', 'frustrated', 'unsettles', 'jealous', 'defensive', 'moral discomfort', 'friction', 'anxiety'],
        'Erosion_of_Epistemic_Agency': ['erosion', 'ground shifting', 'authority', 'redistributed', 'babysitter', 'obedience', 'narrowing', 'surrendered'],
        'Preservation_of_Human_Humanness': ['responsibility', 'accountable', 'hesitation', 'restraint', 'moral triage', 'consequences', 'stake', 'implicated']
    }

    # 2. Extract texts and categorize by phase
    all_texts = " ".join([i.get('Phase_1_Identity', '') + " " + i.get('Phase_2_Critical_Incident', '') + " " + i.get('Phase_3_Humanness', '') for i in interviews])

    # Text normalization
    words = re.findall(r'\w+', all_texts.lower())

    print("\n--- Thematic Prevalence (Keyword Matching) ---")
    theme_counts = {theme: 0 for theme in themes}
    for word in words:
        for theme, keywords in themes.items():
            if word in [k.lower() for k in keywords]:
                theme_counts[theme] += 1

    total_matches = sum(theme_counts.values())
    if total_matches > 0:
        for theme, count in theme_counts.items():
            print(f"{theme}: {count} occurrences ({count/total_matches:.1%})")
    else:
        print("No theoretical keywords detected in sample.")

    # 3. Extract Illustrative Quotations 
    # Finding sentences containing key theoretical constructs
    print("\n--- Extracted Exemplar Quotations for A* Qualitative Data Structure ---")
    sentences = re.split(r'(?<=[.!?]) +', all_texts)

    def extract_quotes(keyword_list, max_quotes=3):
        quotes = []
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keyword_list):
                quotes.append(sentence.strip())
                if len(quotes) >= max_quotes:
                    break
        return quotes

    print("\nConstruct 1: Relational Dissonance")
    for q in extract_quotes(themes['Relational_Dissonance']):
        print(f" - \"{q}\"")

    print("\nConstruct 2: Preservation of Humanness / Accountability")
    for q in extract_quotes(['accountable', 'responsibility', 'consequences', 'moral']):
        print(f" - \"{q}\"")

    print("\nConstruct 3: Algorithmic Phronesis")
    for q in extract_quotes(['wisdom', 'simulate', 'judgment', 'certainty']):
        print(f" - \"{q}\"")

    # 4. Generate Output for Coding Table
    # A standard "Gioia Method" or First-order / Second-order / Aggregate dimension table 
    # is required for Human Relations.
    print("\n--- Required Action for Researchers ---")
    print("To meet 'Human Relations' methodological rigor:")
    print("1. Map First-Order Concepts (e.g., 'Student mistook bot explanation for wisdom').")
    print("2. Group into Second-Order Themes (e.g., 'Algorithmic Simulation of Phronesis').")
    print("3. Synthesize into Aggregate Dimensions (e.g., 'Ontological Erosion of Expertise').")
    print("4. Present a robust Data Structure Figure.")

if __name__ == "__main__":
    analyze_interviews()
