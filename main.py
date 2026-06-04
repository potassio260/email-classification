import numpy as np
import ast
from pathlib import Path
from binary_logistic.model import BinaryLogisticRegression
from multiple_logistic.model import MultipleLogisticRegression
from preprocess import MultilingualTextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



def main():
    test_sample = []
    texts = []
    y_labels = []
    
    types = {
        "0": r"data\Competitions",
        "1": r"data\Exam-Notices",
        "2": r"data\Classes-Teams",
        "3": r"data\Clubs-Activities",
        "4": r"data\Direct-Mails",
        "5": r"data\Assignments",
        "6": r"data\Others",
    }

    for text_type in types.keys():
        for file_path in Path(types[text_type]).glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                y_labels.append(int(text_type))
                texts.append(text)
    
    # Process training data
    processor = MultilingualTextProcessor(auto_detect_lang=True)
    processed_texts = processor.transform(texts)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    X = vectorizer.fit_transform(processed_texts)
    
    # Train model
    model = MultipleLogisticRegression(learning_rate=0.01, number_of_iterations=100000)
    model.fit(X, y_labels)
    for test in test_sample:
        processed_test = processor.transform([test])
        X_test = vectorizer.transform(processed_test)
        predictions = model.predict(X_test)
        print("Predicted class for test data:", predictions)
    
if __name__ == "__main__":
    main()
