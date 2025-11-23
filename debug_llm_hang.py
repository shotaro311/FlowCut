
import logging
import sys
from src.llm.two_pass import TwoPassFormatter
from src.transcribe.base import WordTimestamp

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

def main():
    print("Starting debug_llm_hang.py")
    
    # Mock data - generate 1200 words to trigger chunking
    words = []
    for i in range(1200):
        words.append(WordTimestamp(word=f"word{i}", start=i*0.5, end=i*0.5+0.4, confidence=0.9))
    
    text = " ".join(w.word for w in words)
    
    formatter = TwoPassFormatter(
        llm_provider="openai",
        temperature=0.0,
        timeout=20.0
    )
    
    print(f"Calling TwoPassFormatter.run with {len(words)} words...")
    try:
        # chunk_size defaults to 500, so this should trigger 3 chunks
        result = formatter.run(text, words, max_chars=17.0, chunk_size=500)
        print(f"Result segments: {len(result.segments) if result else 0}")

    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
