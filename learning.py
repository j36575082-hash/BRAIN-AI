
import json
import os
import re
import math
from collections import Counter, defaultdict

# ==========================================
#       ULTRON - MACHINE LEARNING
#       LEVEL 5
# ==========================================

DATA_FILE = "ultron_training_data.json"
MODEL_FILE = "ultron_nlp_model.json"

print("\n====================================")
print("       ULTRON LEARNING ENGINE")
print("       MACHINE LEARNING V1")
print("====================================")


# ==========================================
# TRAINING DATA
# ==========================================

DEFAULT_DATA = {
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "howdy",
        "hello ultron"
    ],

    "goodbye": [
        "bye",
        "goodbye",
        "see you later",
        "talk to you later"
    ],

    "thanks": [
        "thank you",
        "thanks",
        "thanks a lot",
        "I appreciate it"
    ],

    "how_are_you": [
        "how are you",
        "how are you doing",
        "are you okay",
        "how is everything"
    ],

    "identity": [
        "what is your name",
        "who are you",
        "tell me your name",
        "introduce yourself"
    ],

    "creator": [
        "who created you",
        "who is your creator",
        "who made you",
        "who developed you"
    ],

    "help": [
        "help me",
        "what can you do",
        "show your features",
        "how can you help me"
    ],

    "time": [
        "what time is it",
        "tell me the time",
        "current time",
        "show the time"
    ]
}


# ==========================================
# LOAD TRAINING DATA
# ==========================================

def load_training_data():

    if os.path.exists(DATA_FILE):

        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict) and all(
                isinstance(label, str)
                and isinstance(sentences, list)
                and all(isinstance(s, str) for s in sentences)
                for label, sentences in data.items()
            ) and data:

                return data

        except (OSError, ValueError):
            print("Training data could not be loaded.")

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(DEFAULT_DATA, file, indent=4)

    return DEFAULT_DATA.copy()


training_data = load_training_data()


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"[a-z0-9']+",
        text.lower()
    )


# ==========================================
# NAIVE BAYES TRAINING
# ==========================================

def train_model():

    print("\nULTRON is learning from text...")

    class_counts = Counter()
    word_counts = defaultdict(Counter)
    vocabulary = set()
    total_sentences = 0

    for label, sentences in training_data.items():

        for sentence in sentences:

            words = tokenize(sentence)

            if not words:
                continue

            class_counts[label] += 1
            total_sentences += 1

            for word in words:

                word_counts[label][word] += 1
                vocabulary.add(word)

    if total_sentences == 0 or not vocabulary:
        print("No valid training data found.")
        return None

    model = {
        "class_counts": dict(class_counts),

        "word_counts": {
            label: dict(counts)
            for label, counts in word_counts.items()
        },

        "vocabulary": sorted(vocabulary),

        "total_sentences": total_sentences
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=4)

    print("Training completed!")
    print("Total examples:", total_sentences)
    print("Vocabulary size:", len(vocabulary))
    print("Model saved:", MODEL_FILE)

    return model


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if os.path.exists(MODEL_FILE):

        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as file:
                model = json.load(file)

            if (
                model.get("total_sentences", 0) > 0
                and model.get("vocabulary")
                and model.get("class_counts")
                and model.get("word_counts")
            ):
                print("Existing NLP model loaded!")
                return model

        except (OSError, ValueError):
            print("Old model could not be loaded.")

    return train_model()


model = load_model()


# ==========================================
# PREDICTION ENGINE
# ==========================================

def predict_intent(sentence):

    words = tokenize(sentence)

    if not words:
        return "unknown", 0.0

    classes = model["class_counts"]
    vocabulary = set(model["vocabulary"])

    total_sentences = model["total_sentences"]
    vocab_size = len(vocabulary)

    scores = {}

    for label, count in classes.items():

        score = math.log(count / total_sentences)

        word_total = sum(
            model["word_counts"][label].values()
        )

        denominator = word_total + vocab_size

        for word in words:

            if word not in vocabulary:
                continue

            frequency = model["word_counts"][label].get(
                word, 0
            )

            probability = (
                frequency + 1
            ) / denominator

            score += math.log(probability)

        scores[label] = score

    maximum = max(scores.values())

    probabilities = {
        label: math.exp(score - maximum)
        for label, score in scores.items()
    }

    total = sum(probabilities.values())

    best_label = max(probabilities, key=probabilities.get)

    confidence = probabilities[best_label] / total

    return best_label, confidence


# ==========================================
# RESPONSE ENGINE
# ==========================================

def generate_response(intent):

    responses = {

        "greeting":
            "Hello! I am ULTRON. How can I help you?",

        "goodbye":
            "Goodbye! Conversation ended.",

        "thanks":
            "You are welcome!",

        "how_are_you":
            "I am operational and ready to learn.",

        "identity":
            "I am ULTRON, an AI learning project.",

        "creator":
            "My creator is Harsh.",

        "help":
            "I can recognize greetings, questions and basic intents.",

        "time":
            "Current time: " + __import__("datetime").datetime.now().strftime(
                "%I:%M %p"
            )
    }

    return responses.get(
        intent,
        "I am not confident about this sentence yet."
    )


# ==========================================
# TEACH ULTRON
# ==========================================

def teach_ultron():

    print("\nAvailable intents:")

    labels = list(training_data.keys())

    for index, label in enumerate(labels, 1):
        print(f"{index}. {label}")

    choice = input("\nChoose intent number: ").strip()

    if not choice.isdigit():
        print("Invalid choice.")
        return

    index = int(choice) - 1

    if index < 0 or index >= len(labels):
        print("Invalid intent.")
        return

    label = labels[index]

    sentence = input("Enter a new sentence: ").strip()

    if not sentence or not tokenize(sentence):
        print("Invalid sentence.")
        return

    training_data[label].append(sentence)

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(training_data, file, indent=4)

    print("\nNew example saved!")

    global model
    model = train_model()

    print("ULTRON learned a new example!")


# ==========================================
# MAIN CHAT
# ==========================================

def start_ultron():

    print("\nULTRON: Learning engine activated.")
    print("ULTRON: Ready for conversation.")

    print("\nCommands:")
    print("/teach  - Teach a new sentence")
    print("/exit   - Close ULTRON")

    while True:

        user_text = input("\nYou: ").strip()

        if not user_text:
            continue

        if user_text.lower() in ["/exit", "exit", "quit"]:
            print("\nULTRON: Shutting down.")
            break

        if user_text.lower() == "/teach":
            teach_ultron()
            continue

        intent, confidence = predict_intent(user_text)

        print("\nDetected intent:", intent)
        print("Confidence:", round(confidence * 100, 2), "%")

        if confidence < 0.35:
            print(
                "\nULTRON: I don't understand this sentence yet."
            )
        else:
            print("\nULTRON:", generate_response(intent))


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    if model is not None:
        start_ultron()
    else:
        print("ULTRON could not initialize its learning model.")
