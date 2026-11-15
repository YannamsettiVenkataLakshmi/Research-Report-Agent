import spacy
import re
from rag_service.src.technical_patterns import TECH_PATTERNS

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

class TechnicalPreprocessor:
    """
    Preprocesses technical documents for BM25 retrieval.
    Preserves technical tokens while lemmatizing normal English words.
    """

    def __init__(self): 
        pass 

    def is_technical(self, token):

        for pattern in TECH_PATTERNS:
            if pattern.fullmatch(token):
                return True
        return False

    def preprocess(self, text):
        """
        Tokenize text for BM25 indexing.
        Technical tokens are preserved while normal words are lemmatized.
        """

        tokens = []

        # finding normal words and technical words in the text
        raw_tokens = re.findall(
            r"/[A-Za-z0-9._/-]+|[A-Za-z0-9._#+-]+",
            text
        )

        for raw in raw_tokens:

            if self.is_technical(raw):
                tokens.append(raw.lower())
                continue

            doc = nlp(raw.lower())

            for token in doc:

                if token.is_stop or token.is_punct or token.is_space:
                    continue
                lemma = token.lemma_.strip()

                if lemma:
                    tokens.append(lemma)
        return tokens
