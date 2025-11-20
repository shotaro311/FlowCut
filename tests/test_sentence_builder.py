from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.builders import sentences_from_words  # noqa: E402
from src.transcribe.base import WordTimestamp  # noqa: E402


def make_word(text: str, start: float) -> WordTimestamp:
    return WordTimestamp(word=text, start=start, end=start + 1.0)


def test_sentences_split_on_delimiters():
    words = [make_word("設定", 0), make_word("を", 1), make_word("開いて。", 2), make_word("ください", 3)]
    sentences = sentences_from_words(words)
    assert len(sentences) == 2
    assert sentences[0].text == "設定 を 開いて。"
    assert sentences[0].start == 0
    assert sentences[0].end == 3
    assert sentences[1].text == "ください"


def test_sentences_handle_empty_words_with_fallback():
    sentences = sentences_from_words([], fallback_text="テスト 用 テキスト")
    assert len(sentences) == 1
    assert sentences[0].text == "テスト 用 テキスト"


def test_sentences_strip_blank_tokens():
    words = [make_word(" ", 0), make_word("実行", 1)]
    sentences = sentences_from_words(words)
    assert len(sentences) == 1
    assert sentences[0].text == "実行"
