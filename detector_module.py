from lettucedetect import LettuceDetect

# Model path for LettuceDetect (Base version is efficient: 150M parameters) [10]
DETECTOR_MODEL = 'KRLabsOrg/lettucedect-base-modernbert-en-v1'

def initialize_detector():
    """Initializes the Hallucination Detection Model."""
    print(f"Initializing Hallucination Detector: {DETECTOR_MODEL}")
    detector = LettuceDetect(DETECTOR_MODEL)
    return detector

def run_hallucination_detection(detector, query: str, context: list, response: str) -> dict:
    """
    Scores the generated response against the provided context (references).
    
    Args:
        detector: The initialized LettuceDetect model.
        query (str): The user question.
        context (list): The list of retrieved knowledge snippets (z).
        response (str): The generated answer (y).
        
    Returns:
        dict: The result dictionary containing tokens, scores, and classifications.
    """
    print("\n--- Running Hallucination Detection (Judge) ---")
    
    # LettuceDetect processes the (Context, Query, Response) triple [11]
    hallucination_scores = detector.score(
        question=query,
        references=context,
        response=response
    )
    
    return hallucination_scores

def highlight_hallucination(tokens: list, scores: list) -> str:
    """
    Report Feature: Formats the output to highlight hallucinated spans in red.
    (This function uses terminal color codes for visual highlighting.)
    """
    output = []
    
    # A score > 0.5 indicates a high confidence in hallucination [12]
    for token, score in zip(tokens, scores):
        if score > 0.5:
            # \033[91m sets color to red (hallucinated)
            output.append(f"\033[91m{token}\033[0m") 
        else:
            # \033[0m resets color (supported)
            output.append(token)
            
    # Rejoin tokens, adding spaces appropriately (Handling potential issues with sub-word tokenization)
    highlighted_text = " ".join(output)
    highlighted_text = highlighted_text.replace(" ##", "").replace(" ##", "") 
    return highlighted_text

# If running this file directly for testing:
if __name__ == '__main__':
    detector_model = initialize_detector()
    mock_context = ["RAG uses BART as generator.", "RAG was invented by Patrick Lewis."]
    mock_query = "Who made RAG?"
    # Mock Hallucination: Saying T5 instead of BART
    mock_response = "Patrick Lewis created RAG, which uses the T5 model."
    
    scores = run_hallucination_detection(detector_model, mock_query, mock_context, mock_response)
    highlighted = highlight_hallucination(scores['tokens'], scores['scores'])
    print(f"\nMock Highlighted Output:\n{highlighted}")