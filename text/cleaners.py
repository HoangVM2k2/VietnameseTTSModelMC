""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
import unicodedata
from unidecode import unidecode
from phonemizer import phonemize
from g2p_en.expand import normalize_numbers
from .symbols import _vi_punctuation
from .g2p_vi import G2p_vi


_g2p_vi = G2p_vi()

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def vietnamese_cleaners(text: str) -> list[str]:
    text = text.lower()
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFC', text)
                            if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub("[\'\"()]+", "", text)
    text = re.sub("[-]+", " ", text)
    text = re.sub(f" ?([{_vi_punctuation}]) ?", r"\1", text)  # !! -> !
    text = re.sub(f"([{_vi_punctuation}])+", r"\1", text)  # !! -> !

    text = re.sub(r"\bxhcn\b", "xã hội chủ nghĩa", text)
    text = re.sub(r"\btivi\b", "ti vi", text)
    text = re.sub(r"\bishimura\b", "i si mu ra", text)
    text = re.sub(r"\bunitology\b", "u ni tô ly", text)
    text = re.sub(r"\bnozel\b", "nô gieo", text)
    text = re.sub(r"\bbantam\b", "ban tam", text)
    text = re.sub(r"\bgha\b", "ga", text)
    text = re.sub(r"\brig\b", "rít", text)
    text = re.sub(r"\bsếc\b", "sếch", text)
    text = re.sub(r"\blangshan\b", "lang san", text)
    text = re.sub(r"\btribat\b", "tri bát", text)
    text = re.sub(r"\bvapet\b", "va pét", text)
    text = re.sub(r"\bshamo\b", "sa mô", text)

    text = re.sub(f"([{_vi_punctuation}])", r" \1 ", text)
    text = re.sub(rf"\s+", r" ", text)

    text =_g2p_vi(text)

    return text