import os
import random
from collections import defaultdict
from text.symbols import _vi_punctuation, _unk, _sep


class G2p_vi:
    lexicon_dir = os.path.join(os.path.dirname(__file__), "lexicon/en-vi")

    def __init__(self) -> None:
        self.add_sep = True
        self.lexicon = defaultdict(list)
        lexicon_dir = self.lexicon_dir
        print("| Finding lexicon from", lexicon_dir)
        fp = os.path.join(lexicon_dir, "probabilities.txt")
        _has_proba = True
        
        if not (os.path.exists(fp) and os.path.isfile(fp)):
            fp = os.path.join(lexicon_dir, "en-vi-lexicon.txt")

            if not (os.path.exists(fp) and os.path.isfile(fp)):
                raise ValueError("Not found lexicon")
            
            _has_proba = False

        if _has_proba:
            self.probabilities = defaultdict(list)

        print("| Load lexicon from", fp)

        with open(fp, encoding="utf-8") as f:
            for line in f.readlines():
                word_phones = line.split()
                word = word_phones[0]
                if _has_proba is False:
                    phones = " ".join(word_phones[1:])
                else:
                    phones = " ".join(word_phones[1:-1])
                    self.probabilities[word].append(float(word_phones[-1]))

                self.lexicon[word].append(phones)

    def __call__(self, text: str):
        phones = []
        for word in text.split():
            if self.lexicon.get(word) is None:
                if word in _vi_punctuation or word == _sep:
                    phones.append(word)
                else:
                    phones.append(_unk)
            else:
                if self.has_probabilities is False:
                    ph_idx = int(random.random() * len(self.lexicon[word]))
                else:
                    proba = random.random()
                    cum_proba, ph_idx = 0.0, 0
                    for idx, pr in enumerate(self.probabilities[word]):
                        if cum_proba < proba:
                            ph_idx = idx
                        else:
                            break
                        cum_proba += pr

                ph = self.lexicon[word][ph_idx]
                phones += ph.split()
            if self.add_sep:
                phones.append(_sep)
        return phones[:-1]
    
    @property
    def has_probabilities(self) -> bool:
        return hasattr(self, "probabilities")
    
    def generate_probabilities(self, texts: list[str]):
        pass
    
        