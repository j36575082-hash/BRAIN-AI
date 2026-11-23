
import json
import os
import random
import re

DATA_FILE = "ultron_language_data.json"
MODEL_FILE = "ultron_language_model.json"

DEFAULT_TEXTS = [
    "Hello, I am ULTRON.",
    "I am an artificial intelligence learning project.",
    "I learn from examples and training data.",
    "My goal is to understand language and generate responses.",
    "Neural networks help me recognize patterns.",
    "Memory helps me remember previous conversations.",
    "Learning is a continuous process.",
    "I am developing my language generation abilities.",
    "Intelligence requires learning, reasoning and experience.",
    "I am improving my language processing system."
]


class LanguageGenerator:

    def __init__(self):
        self.texts = self.load_texts()
        self.model = {}

    def load_texts(self):
        if not os.path.exists(DATA_FILE):
            self.save_texts(DEFAULT_TEXTS)
            return DEFAULT_TEXTS.copy()

        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

        except (json.JSONDecodeError, OSError):
            pass

        return DEFAULT_TEXTS.copy()

    def save_texts(self, texts):
        with open(DATA_FILE, "w", encoding="utf-8") as file:
            json.dump(texts, file, indent=4)

    def tokenize(self, text):
        return re.findall(r"[a-zA-Z0-9']+|[.!?]", text)

    def train(self):
        self.model = {}

        for sentence in self.texts:
            words = self.tokenize(sentence)

            for i in range(len(words) - 1):
                current = words[i].lower()
                next_word = words[i + 1]

                if current not in self.model:
                    self.model[current] = []

                self.model[current].append(next_word)

        self.save_model()

        print("\nTraining completed!")
        print("Training sentences:", len(self.texts))
        print("Vocabulary states:", len(self.model))

    def save_model(self):
        with open(MODEL_FILE, "w", encoding="utf-8") as file:
            json.dump(self.model, file, indent=4)

        print("Model saved:", MODEL_FILE)

    def load_model(self):
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as file:
                self.model = json.load(file)

            print("Language model loaded!")
            return True

        except (FileNotFoundError, json.JSONDecodeError, OSError):
            print("No saved language model found.")
            return False

    def generate(self, start_word, length=20):
        current = start_word.lower()

        if current not in self.model:
            print("ULTRON: I don't know this starting word yet.")
            return

        output = [current.capitalize()]

        for _ in range(length):
            choices = self.model.get(current, [])

            if not choices:
                break

            next_word = random.choice(choices)

            if next_word in [".", "!", "?"]:
                output[-1] += next_word
                break

            output.append(next_word)
            current = next_word.lower()

        sentence = " ".join(output)

        if sentence[-1] not in ".!?":
            sentence += "."

        print("\nULTRON:", sentence)

    def teach(self):
        sentence = input("\nEnter a training sentence: ").strip()

        if not sentence:
            print("Empty sentence cannot be learned.")
            return

        self.texts.append(sentence)
        self.save_texts(self.texts)

        print("New sentence saved!")
        print("Train the model again to use it.")

    def show_menu(self):
        print("\n================================")
        print("     ULTRON LANGUAGE GENERATOR")
        print("================================")
        print("1. Train language model")
        print("2. Load saved model")
        print("3. Generate sentence")
        print("4. Teach new sentence")
        print("5. Exit")

    def run(self):
        while True:
            self.show_menu()

            choice = input("\nChoose: ").strip()

            if choice == "1":
                self.train()

            elif choice == "2":
                self.load_model()

            elif choice == "3":
                if not self.model:
                    if not self.load_model():
                        continue

                word = input("Starting word: ").strip()

                if word:
                    self.generate(word)

            elif choice == "4":
                self.teach()

            elif choice == "5":
                print("ULTRON Language Generator closed.")
                break

            else:
                print("Invalid choice.")


if __name__ == "__main__":
    generator = LanguageGenerator()
    generator.run()
