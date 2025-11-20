from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.splitter import BlockSplitter, Sentence


def make_sentence(text: str, start: float) -> Sentence:
    return Sentence(text=text, start=start, end=start + 5)


def test_splitter_respects_char_limit():
    sentences = [make_sentence("あいうえお", i * 5) for i in range(4)]
    splitter = BlockSplitter(max_chars=10, overlap_sentences=1)
    blocks = splitter.split(sentences)
    assert len(blocks) == 3
    assert blocks[0].text.startswith("あいうえお")
    # 確認: 2ブロック目冒頭は前ブロック末尾の文
    assert blocks[1].sentences[0].overlap is True


def test_splitter_respects_duration_limit():
    sentences = [make_sentence(f"seg{i}", i * 40) for i in range(3)]
    splitter = BlockSplitter(max_duration=30, overlap_sentences=0)
    blocks = splitter.split(sentences)
    assert len(blocks) == 3
    assert blocks[0].duration == 5


def test_splitter_skips_empty_text():
    sentences = [make_sentence("foo", 0), Sentence(text="  "), make_sentence("bar", 10)]
    splitter = BlockSplitter(max_chars=50)
    blocks = splitter.split(sentences)
    assert len(blocks) == 1
    assert blocks[0].text == "foobar"
