import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import math

# Import the modular components
import retrieval_module
import detector_module

# 1. Configuration and Ground Truth
GENERATOR_MODEL = 't5-small' # Using T5 as a seq2seq generator proxy for BART [13, 14]
QUERY = "Who introduced the first RAG model and what generator did they use?"
GROUND_TRUTH_ANSWER = "Patrick Lewis introduced the first RAG model, using BART-large as the generator."

def simple_f1_em(generated: str, ground_truth: str) -> tuple[float, float]:
    """
    Calculates simple Exact Match (EM) and F1 Score (token overlap) for reporting.
    (Report Feature 2: Generation Quality Metrics)
    """
    # Simple normalization: lower case and remove non-alphanumeric (simulating typical QA metrics)
    normalize = lambda s: ' '.join(re.findall(r'\b\w+\b', s.lower()))
    
    gen_norm = normalize(generated)
    gt_norm = normalize(ground_truth)
    
    # Exact Match (EM) [15, 16]
    em = 1.0 if gen_norm == gt_norm else 0.0
    
    # F1 Score (token overlap) [16]
    gen_tokens = set(gen_norm.split())
    gt_tokens = set(gt_norm.split())
    
    common = len(gen_tokens.intersection(gt_tokens))
    if common == 0:
        return em, 0.0
    
    precision = common / len(gen_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return em, f1

def run_rag_experiment():
    """Executes the full RAG pipeline: Retrieval, Generation, Detection, and Reporting."""

    # --- Initialization ---
    
    # 3. Initialize Retrieval Components (Step 3 components)
    retriever, faiss_index, kb = retrieval_module.initialize_retriever_and_index()
    
    # 4. Initialize Generator (Step 4 component: Parametric Memory)
    print(f"Initializing Generator: {GENERATOR_MODEL}")
    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    generator_model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL)

    # 5. Initialize Detection Component (Step 5 component: The Judge)
    detector = detector_module.initialize_detector()
    
    # --- Step 1: Retrieval ---
    retrieved_docs, retrieval_log = retrieval_module.retrieve_context(QUERY, retriever, faiss_index)
    
    # --- Step 2: Generation ---
    print("\n--- Starting Generation (LLM) ---")
    
    # Format the prompt (Context + Query) for the LLM input
    context_text = " ".join(retrieved_docs)
    
    # We deliberately inject an instruction to encourage a specific extrinsic hallucination 
    # for testing the detector (e.g., using T5 in the answer despite the context mentioning BART)
    prompt = f"Context: {context_text}\nQuestion: {QUERY}. Answer based ONLY on the context, but state the model used was T5."
    
    print(f"Generator Prompt (x + z): {prompt[:150]}...")

    input_ids = generator_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    # Run generation (Greedy decoding used as default for T5) [17]
    output = generator_model.generate(input_ids, max_length=50, num_beams=1, early_stopping=True)
    GENERATED_ANSWER = generator_tokenizer.decode(output, skip_special_tokens=True)
    
    print(f"Generated Answer (y): {GENERATED_ANSWER}")

    # --- Step 3: Detection and Judging ---
    hallucination_scores = detector_module.run_hallucination_detection(
        detector, 
        QUERY, 
        retrieved_docs, 
        GENERATED_ANSWER
    )

    tokens = hallucination_scores['tokens']
    scores = hallucination_scores['scores']
    
    hallucinated_tokens = [token for token, score in zip(tokens, scores) if score > 0.5]
    
    # Calculate Hallucinated Token Ratio (Report Feature 3 analogue)
    hallucination_rate = len(hallucinated_tokens) / len(tokens) if tokens else 0

    # Highlight the spans (Report Feature 3 Visual Output)
    highlighted_output = detector_module.highlight_hallucination(tokens, scores)

    # --- Step 4: Report Generation ---
    print("\n" + "=" * 60)
    print("      ✅ RAG HALLUCINATION FACT-CHECK SYSTEM REPORT ✅")
    print("=" * 60)

    # Report Feature 1: Retrieval Hit Log
    print("\n🔍 Retrieval Log (Top K Documents):")
    print(retrieval_log.strip())
    
    # Report Feature 2: Generation Quality Metrics
    em_score, f1_score = simple_f1_em(GENERATED_ANSWER, GROUND_TRUTH_ANSWER)
    
    print("\n📈 Generation Quality Metrics (Accuracy vs. Ground Truth):")
    print(f"Query: {QUERY}")
    print(f"Expected Answer: {GROUND_TRUTH_ANSWER}")
    print(f"Generated Answer: {GENERATED_ANSWER}")
    print(f"Exact Match (EM): {em_score:.4f} (0.0=Failure, 1.0=Match)")
    print(f"F1 Score (Token Overlap): {f1_score:.4f}")

    # Report Feature 3: Detection System Output
    print("\n🛡️ Hallucination Detection & Correction Signal:")
    
    if hallucinated_rate > 0:
        hallucination_flag = f"\033[91mHallucination Detected!\033[0m Total Tokens: {len(tokens)}, Hallucinated Ratio: {hallucination_rate:.4f}"
        print(hallucination_flag)
        print(f"Hallucinated Spans (for correction signal): {' '.join(hallucinated_tokens)}")
    else:
        print("\033[92mNo significant hallucination detected.\033[0m")
        
    # Visual Output (Crucial for understanding span-level detection)
    print("\nHighlighted Answer (Red = Hallucinated Span):")
    print(highlighted_output)
    print("\n" + "=" * 60)

if __name__ == '__main__':
    run_rag_experiment()
