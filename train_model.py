import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv("language_detection_dataset.csv")

print("Dataset loaded successfully!")
print(df.head())



X = df["text"]
y = df["language"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))



vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5)
)


X_train_tfidf = vectorizer.fit_transform(X_train)


X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF conversion completed!")



model = LogisticRegression(
    max_iter=1000
)



model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")



y_pred = model.predict(X_test_tfidf)



accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)



with open(
    "language_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\nlanguage_model.pkl created successfully!")



with open(
    "vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


print("vectorizer.pkl created successfully!")




test_texts = [
    "I am learning Python.",
    "मुझे पायथन सीखना है।",
    "मला पायथन शिकायचे आहे."
]


test_vectors = vectorizer.transform(
    test_texts
)


predictions = model.predict(
    test_vectors
)


print("\nTest Predictions:")

for text, prediction in zip(
    test_texts,
    predictions
):

    print(
        text,
        "→",
        prediction
    )


print("\nTraining process completed successfully!")