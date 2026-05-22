import random
# Requires: pip install nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

def paraphrase_ai_text(text: str) -> str:
    # Synonym replacement to mimic human edits
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
    return aug.augment(text)[0]

def add_typos(text: str) -> str:
    # Character level typos
    aug = nac.KeyboardAug(aug_char_p=0.05, aug_word_p=0.1)
    return aug.augment(text)[0]

def mix_content(human_text: str, ai_text: str) -> str:
    human_sents = human_text.split(". ")
    ai_sents = ai_text.split(". ")
    mixed = human_sents[:len(human_sents)//2] + ai_sents[len(ai_sents)//2:]
    random.shuffle(mixed)
    return ". ".join(mixed)
