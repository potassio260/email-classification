# Imports
import re
import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Ensure reproducible results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("error")


class MultilingualTextCleaner:
    def __init__(self):
        self.characters_to_remove = list(r"<>&[]{}|\^~`*_#+$€£¥×÷=@")
        
        self.web_email_tokens = {
            "url": "<URL>", "http": "<URL>", "https": "<URL>",
            "www": "<URL>", ".com": "", ".org": "", ".net": "",
            "email": "<EMAIL>", "mailto": "<EMAIL>"
        }
        
        self.language_dictionaries = self._build_dictionaries()
    
    def _build_dictionaries(self) -> Dict:
        return {
            "en": {
                "contractions": {
                    "don't": "do not", "won't": "will not", "can't": "cannot",
                    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
                    "'d": " would", "'m": " am", "let's": "let us", "that's": "that is",
                    "there's": "there is", "here's": "here is", "what's": "what is",
                    "where's": "where is", "who's": "who is", "it's": "it is",
                    "he's": "he is", "she's": "she is", "we're": "we are",
                    "they're": "they are", "you're": "you are", "i'm": "i am",
                },
                "abbreviations": {
                    "u.s.": "united states", "u.k.": "united kingdom",
                    "etc.": "etcetera", "vs.": "versus", "vs": "versus",
                    "e.g.": "for example", "i.e.": "that is", "dr.": "doctor",
                    "mr.": "mister", "mrs.": "missus", "ms.": "miss",
                },
                "slang": {
                    "lol": "laugh out loud", "omg": "oh my god", "btw": "by the way",
                    "fyi": "for your information", "asap": "as soon as possible",
                    "aka": "also known as", "idk": "i do not know", "brb": "be right back",
                },
                "symbols": {
                    "&": "and", "@": "at", "%": "percent", "$": "dollar",
                },
            },
            "es": {
                "contractions": {
                    "al": "a el", "del": "de el", "pal": "para el",
                    "pa'": "para", "d'": "de", "qu'": "que", "m'": "me",
                },
                "abbreviations": {
                    "sr.": "señor", "sra.": "señora", "ud.": "usted",
                    "dr.": "doctor", "dra.": "doctora", "etc.": "etcétera",
                },
                "slang": {
                    "tqm": "te quiero mucho", "xq": "por qué", "k": "que",
                    "pq": "porque", "bn": "bien", "tb": "también", "x": "por",
                },
                "symbols": {
                    "&": "y", "@": "arroba", "%": "por ciento", "$": "dólar",
                },
            },
        }
    
    def clean(self, text: str, lang: str = "en") -> str:
        text = text.lower()
        
        # Handle URLs and emails
        for token, replacement in self.web_email_tokens.items():
            text = text.replace(token, replacement)
        
        # Remove unwanted characters
        for ch in self.characters_to_remove:
            text = text.replace(ch, " ")
        
        # Language-specific replacements
        lang_data = self.language_dictionaries.get(lang, self.language_dictionaries["en"])
        
        for old, new in lang_data["contractions"].items():
            text = text.replace(old, new)
        for old, new in lang_data["abbreviations"].items():
            text = text.replace(old, new)
        for old, new in lang_data["slang"].items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        for old, new in lang_data["symbols"].items():
            text = text.replace(old, f" {new} ")
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class MultilingualTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, auto_detect_lang: bool = True, default_lang: str = "en"):
        self.auto_detect_lang = auto_detect_lang
        self.default_lang = default_lang
        self.cleaner = MultilingualTextCleaner()
        
        # Load spaCy models
        try:
            self.models = {
                "en": spacy.load("en_core_web_sm"),
                "es": spacy.load("es_core_news_sm"),
            }
        except OSError as e:
            raise OSError(
                "SpaCy models not found. Install with:\n"
                "python -m spacy download en_core_web_sm\n"
                "python -m spacy download es_core_news_sm"
            ) from e
    
    def _detect_language(self, text: str) -> str:
        if not self.auto_detect_lang or not LANGDETECT_AVAILABLE:
            return self.default_lang
        
        try:
            lang = detect(text)
            return lang if lang in self.models else self.default_lang
        except:
            return self.default_lang
    
    def _process_single(self, text: str) -> str:
        lang = self._detect_language(text)
        cleaned = self.cleaner.clean(text, lang)
        
        # Lemmatize with spaCy
        doc = self.models[lang](cleaned)
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_space])
        
        return lemmatized
    
    def _process_batch(self, texts: List[str]) -> List[str]:
        processed = []
        
        # Group texts by detected language for batch processing
        lang_groups = {}
        for idx, text in enumerate(texts):
            lang = self._detect_language(text)
            cleaned = self.cleaner.clean(text, lang)
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append((idx, cleaned))
        
        # Process each language group in batch
        results = [None] * len(texts)
        for lang, items in lang_groups.items():
            indices, cleaned_texts = zip(*items)
            
            # Batch process with spaCy (much faster!)
            docs = list(self.models[lang].pipe(cleaned_texts, batch_size=50))
            
            for idx, doc in zip(indices, docs):
                lemmatized = " ".join([token.lemma_ for token in doc if not token.is_space])
                results[idx] = lemmatized
        
        return results
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> List[str]:
        if len(X) == 1:
            return [self._process_single(X[0])]
        return self._process_batch(X)


class NLPPipeline:
    def __init__(
        self,
        auto_detect_lang: bool = True,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 5000,
        min_df: int = 2,
        max_iter: int = 1000,
    ):
        self.pipeline = Pipeline([
            ('preprocessor', MultilingualTextProcessor(auto_detect_lang=auto_detect_lang)),
            ('vectorizer', TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                sublinear_tf=True,  # Use log scaling
            )),
            ('classifier', LogisticRegression(max_iter=max_iter, random_state=42)),
        ])
    
    def fit(self, X_train: List[str], y_train: List[int]):
        print("🔄 Training pipeline...")
        self.pipeline.fit(X_train, y_train)
        print("✅ Training complete!")
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test: List[str], y_test: List[int]):
        y_pred = self.predict(X_test)
        
        print("\n" + "="*50)
        print("📊 Model Evaluation")
        print("="*50)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy_score(y_test, y_pred)
    
    def get_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        feature_names = vectorizer.get_feature_names_out()
        top_features = {}
        
        for idx, class_label in enumerate(classifier.classes_):
            coefficients = classifier.coef_[idx]
            top_indices = np.argsort(coefficients)[-n:][::-1]
            top_features[class_label] = [
                (feature_names[i], coefficients[i]) for i in top_indices
            ]
        
        return top_features