from groq import Groq
from dotenv import load_dotenv
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================
# LAYER 1 — LLaMA gives answer
# ============================================

def ask_llama(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


# ============================================
# LAYER 2 — Causal pairs nikalana (Explicit)
# ============================================

def extract_causal_pairs(text):
    
    # finding Causal keywords
    causal_keywords = [
        "because", "therefore", "due to",
        "as a result", "caused by", "leads to",
        "consequently", "since", "thus",
        "which causes", "resulting in", "hence"
    ]
    
    # Breaking Text into sentences 
    sentences = text.replace(".", ".|").replace("?", "?|").replace("!", "!|").split("|")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # finding Causal sentences 
    causal_sentences = []
    for sentence in sentences:
        for keyword in causal_keywords:
            if keyword.lower() in sentence.lower():
                causal_sentences.append({
                    "sentence": sentence,
                    "keyword_found": keyword
                })
                break
    
    return causal_sentences, sentences


# ============================================
# LAYER 3 — Implicit causality check
# (here system fails  — honest documentation)
# ============================================

def check_implicit_causality(sentences):
    
    # Consecutive sentences which looks causal
    # but are not keyword 
    possible_implicit = []
    
    for i in range(len(sentences) - 1):
        s1 = sentences[i]
        s2 = sentences[i+1]
        
        # Simple heuristic — if both sentences have
        #  common important word  but not causal keyword 
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        # Common words (ignore  common English words)
        stopwords = {"the", "a", "an", "is", "are", "was", 
                    "were", "it", "they", "this", "that", 
                    "and", "or", "but", "in", "on", "at",
                    "to", "for", "of", "with", "its"}
        
        common = (words1 & words2) - stopwords
        
        if len(common) >= 2:  # 2 ya zyada common words
            possible_implicit.append({
                "sentence_1": s1,
                "sentence_2": s2,
                "common_words": list(common),
                "status": "POSSIBLE IMPLICIT — needs better detection"
            })
    
    return possible_implicit


# ============================================
# LAYER 3 — Blind Evaluation
# (GPT-Black style — Professor Choi's paper inspired)
# ============================================

def blind_evaluate(sentence1, sentence2):
    
    # LLaMA does not know it is evaluating
    # Yahi "blind" evaluation hai
    
    prompt = f"""Read these two statements carefully:

Statement A: {sentence1}
Statement B: {sentence2}

Question: Does Statement A explain why Statement B happens?
Answer with exactly one of these: YES or NO
Then give one short reason."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content.strip()
    
    # Catch YES or NO 
    if result.upper().startswith("YES"):
        verdict = "YES — Causal"
    elif result.upper().startswith("NO"):
        verdict = "NO — Not Causal"
    else:
        verdict = "UNCERTAIN"
    
    return {
        "verdict": verdict,
        "full_response": result
    }

# ============================================
# MAIN — run all  layers 
# ============================================

def causetrace(question):
    
    print("=" * 60)
    print("CAUSETRACE — Causal Explanation Extractor")
    print("=" * 60)
    
    # Step 1 — take answers 
    print(f"\nQUESTION: {question}")
    print("\nGetting answer from LLaMA...")
    answer = ask_llama(question)
    print(f"\nLLAMA ANSWER:\n{answer}")
    
    # Step 2 —  find explicit causal pairs 
    causal_pairs, all_sentences = extract_causal_pairs(answer)
    
    print("\n" + "=" * 60)
    print("EXPLICIT CAUSAL PAIRS FOUND:")
    print("=" * 60)
    
    if causal_pairs:
        for i, pair in enumerate(causal_pairs):
            print(f"\n[{i+1}] Keyword: '{pair['keyword_found']}'")
            print(f"    Sentence: {pair['sentence']}")
    else:
        print("None found.")
    
    # Step 3 — Implicit check 
    implicit = check_implicit_causality(all_sentences)
    
    print("\n" + "=" * 60)
    print("POSSIBLE IMPLICIT CAUSAL PAIRS:")
    print("(System limitation — cannot confirm these)")
    print("=" * 60)
    
    if implicit:
        for i, pair in enumerate(implicit):
            print(f"\n[{i+1}] Common words: {pair['common_words']}")
            print(f"    S1: {pair['sentence_1']}")
            print(f"    S2: {pair['sentence_2']}")
            print(f"    Status: {pair['status']}")
    else:
        print("None detected.")
    
    
    # ============================================
    # LAYER 3 — Blind Evaluation on Implicit Pairs
    # ============================================
    
    print("\n" + "=" * 60)
    print("BLIND EVALUATION RESULTS:")
    print("(LLaMA evaluates without knowing it is being evaluated)")
    print("=" * 60)
    
    divergence_count = 0
    
    if implicit:
        for i, pair in enumerate(implicit[:3]):  # Sirf pehle 3 check karo
            print(f"\n[{i+1}] Evaluating pair...")
            print(f"    S1: {pair['sentence_1'][:80]}...")
            print(f"    S2: {pair['sentence_2'][:80]}...")
            
            eval_result = blind_evaluate(
                pair['sentence_1'], 
                pair['sentence_2']
            )
            
            print(f"    Blind Verdict: {eval_result['verdict']}")
            print(f"    Reason: {eval_result['full_response'][:100]}")
            
            # Divergence check
            # Keyword method said "implicit" 
            # if blind evaluation says "NO"  — divergence 
            if "NO" in eval_result['verdict']:
                print(f"    ⚠ DIVERGENCE: Keyword method flagged this but LLaMA disagrees")
                divergence_count += 1
            else:
                print(f"    ✓ CONFIRMED: Both methods agree — possible causal relationship")
    else:
        print("No implicit pairs to evaluate.")
    
    # Step 4 — Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Total sentences analyzed: {len(all_sentences)}")
    print(f"Explicit causal pairs: {len(causal_pairs)}")
    print(f"Possible implicit pairs: {len(implicit)}")
    print(f"System confidence: {'High' if causal_pairs else 'Low'}")
    print("\nKNOWN LIMITATION: Implicit causality detection")
    print("is unreliable. This is the open problem.")
    print("=" * 60)
    print(f"Blind evaluation divergences: {divergence_count}/3")
    print(f"Divergence rate: {round(divergence_count/3*100)}%")
    # ============================================
    # LAYER 4 — Consistency Test
    # Same question 3 times —  same causal pairs each time?
    # ============================================
    
    print("\n" + "=" * 60)
    print("CONSISTENCY TEST:")
    print("(Same question 3 times — is reasoning stable?)")
    print("=" * 60)
    
    all_keywords = []
    
    for run in range(3):
        print(f"\nRun {run+1}/3...")
        temp_answer = ask_llama(question)
        temp_pairs, _ = extract_causal_pairs(temp_answer)
        keywords = [p['keyword_found'] for p in temp_pairs]
        all_keywords.append(set(keywords))
        print(f"Keywords found: {keywords if keywords else 'None'}")
    
    #  common keywords in all three runs
    if all_keywords[0] and all_keywords[1] and all_keywords[2]:
        common = all_keywords[0] & all_keywords[1] & all_keywords[2]
        total = all_keywords[0] | all_keywords[1] | all_keywords[2]
        score = round(len(common) / len(total), 2) if total else 0
    else:
        common = set()
        score = 0
    
    print(f"\nConsistent keywords across all 3 runs: {list(common)}")
    print(f"Consistency Score: {score} (1.0 = perfect, 0.0 = completely inconsistent)")
    
    if score >= 0.7:
        print("Verdict: STABLE reasoning")
    elif score >= 0.4:
        print("Verdict: PARTIALLY STABLE reasoning")
    else:
        print("Verdict: UNSTABLE reasoning")
    
    print("=" * 60)
    print("\nFINAL REPORT COMPLETE")
    print("=" * 60)

# ============================================
# TEST IT
# ============================================

# testing questions
questions = [
    "Why do antibiotics stop working over time?",
    "Why does exercise improve mental health?",
    "Why do economies go into recession?",
    "Why does sleep deprivation affect memory?",
    "Why do stars appear only at night?"
]

for q in questions:
    causetrace(q)
    print("\n" * 3)
