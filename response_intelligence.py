
import json
import os
import random
import re

DATA_FILE = "ultron_response_data.json"
MEMORY_FILE = "ultron_context.json"

DEFAULT_DATA = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning",
            "good evening", "howdy"
        ],
        "responses": [
            "Hello! How can I help you?",
            "Hey! ULTRON is ready.",
            "Greetings! What are we doing today?"
        ]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "good night"],
        "responses": [
            "Goodbye! See you again.",
            "ULTRON signing off."
        ]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thank you so much"],
        "responses": [
            "You're welcome!",
            "Happy to help."
        ]
    },
    "identity": {
        "patterns": [
            "who are you", "what are you",
            "introduce yourself"
        ],
        "responses": [
            "I am ULTRON, an AI learning project.",
            "I am ULTRON. I am continuously learning."
        ]
    },
    "creator": {
        "patterns": [
            "who created you", "who is your creator",
            "who made you"
        ],
        "responses": [
            "I am being developed by my creator.",
            "My creator is building me step by step."
        ]
    },
    "how_are_you": {
        "patterns": ["how are you", "are you fine", "how are you doing"],
        "responses": [
            "I am functioning properly!",
            "All systems are running."
        ]
    },
    "help": {
        "patterns": ["help", "what can you do", "your features"],
        "responses": [
            "I can recognize intents, remember learned examples and respond to messages.",
            "I am learning language processing and conversation."
        ]
    }
}


def load_data():
    if not os.path.exists(DATA_FILE):
        save_data(DEFAULT_DATA)
        return DEFAULT_DATA.copy()

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, OSError):
        pass

    print("Training data could not be loaded. Starting fresh.")
    save_data(DEFAULT_DATA)
    return DEFAULT_DATA.copy()


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def tokenize(sentence):
    return re.findall(r"[a-z0-9']+", sentence.lower())


def calculate_score(message, pattern):
    message_words = set(tokenize(message))
    pattern_words = set(tokenize(pattern))

    if not message_words or not pattern_words:
        return 0

    if message.lower().strip() == pattern.lower().strip():
        return 1.0

    common = message_words.intersection(pattern_words)

    if not common:
        return 0

    precision = len(common) / len(message_words)
    recall = len(common) / len(pattern_words)

    return 2 * precision * recall / (precision + recall)


def recognize_intent(message, data):
    best_intent = None
    best_score = 0

    for intent, details in data.items():
        for pattern in details["patterns"]:
            score = calculate_score(message, pattern)

            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= 0.50:
        return best_intent, best_score

    return None, best_score


def save_conversation(role, message):
    history = []

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                history = json.load(file)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append({
        "role": role,
        "message": message
    })

    history = history[-20:]

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def teach_ultron(data):
    print("\n--- TEACH ULTRON ---")

    intent = input("Intent name: ").strip().lower()
    pattern = input("Example sentence: ").strip()
    response = input("ULTRON's response: ").strip()

    if not intent or not pattern or not response:
        print("All three fields are required.")
        return

    if intent not in data:
        data[intent] = {
            "patterns": [],
            "responses": []
        }

    if pattern not in data[intent]["patterns"]:
        data[intent]["patterns"].append(pattern)

    if response not in data[intent]["responses"]:
        data[intent]["responses"].append(response)

    save_data(data)

    print("\nULTRON learned a new example!")


def show_intents(data):
    print("\n--- LEARNED INTENTS ---")

    for intent, details in data.items():
        print(
            f"{intent}: "
            f"{len(details['patterns'])} patterns, "
            f"{len(details['responses'])} responses"
        )


def main():
    data = load_data()

    print("\n================================")
    print("    ULTRON RESPONSE INTELLIGENCE")
    print("================================")

    print("Commands:")
    print("/teach    - Teach a new example")
    print("/intents  - Show learned intents")
    print("/exit     - Exit ULTRON")

    while True:
        message = input("\nYou: ").strip()

        if not message:
            continue

        if message.lower() == "/exit":
            print("ULTRON: Session saved. Goodbye!")
            break

        if message.lower() == "/teach":
            teach_ultron(data)
            continue

        if message.lower() == "/intents":
            show_intents(data)
            continue

        intent, confidence = recognize_intent(message, data)

        if intent:
            response = random.choice(data[intent]["responses"])

            print(f"ULTRON [{intent}]: {response}")
            print(f"Match score: {confidence:.2f}")

        else:
            response = (
                "I don't understand this message yet. "
                "You can teach me using /teach."
            )
            print("ULTRON:", response)

        save_conversation("user", message)
        save_conversation("ultron", response)


if __name__ == "__main__":
    main()
