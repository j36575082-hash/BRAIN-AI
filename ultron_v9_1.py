
import json
import os
import re
import math
from collections import Counter
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "knowledge": "ultron_v9_knowledge.json",
    "training": "ultron_v9_training.json",
    "vocabulary": "ultron_v9_1_vocabulary.json",
    "unknown": "ultron_v9_1_unknown_words.json"
}


def load(name, default):
    try:
        with open(os.path.join(BASE, FILES[name]), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save(name, data):
    filename = os.path.join(BASE, FILES[name])
    temp = filename + ".tmp"

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(temp, filename)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class VocabularyEngine:

    def __init__(self):
        self.data = load("vocabulary", {
            "words": {},
            "total_sentences": 0
        })

        self.unknown = load("unknown", [])

    def learn_sentence(self, sentence):
        words = tokenize(sentence)

        if not words:
            return

        for word in words:
            self.data["words"][word] = (
                self.data["words"].get(word, 0) + 1
            )

        self.data["total_sentences"] += 1
        save("vocabulary", self.data)

    def vectorize(self, sentence):
        words = tokenize(sentence)
        vocabulary = list(self.data["words"].keys())

        counts = Counter(words)

        return {
            word: counts.get(word, 0)
            for word in vocabulary
        }

    def find_unknown(self, sentence):
        words = tokenize(sentence)
        known = self.data["words"]

        unknown = sorted(set(
            word for word in words
            if word not in known
        ))

        for word in unknown:
            if word not in self.unknown:
                self.unknown.append(word)

        save("unknown", self.unknown)

        return unknown

    def statistics(self):
        print("\n========== VOCABULARY STATISTICS ==========")
        print("Unique words:", len(self.data["words"]))
        print("Total learned sentences:", self.data["total_sentences"])
        print("Unknown words:", len(self.unknown))
        print("===========================================")

    def show_words(self):
        print("\n========== KNOWN WORDS ==========")

        for word, count in sorted(
            self.data["words"].items(),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{word}: {count}")

    def show_unknown(self):
        print("\n========== UNKNOWN WORDS ==========")

        if not self.unknown:
            print("No unknown words.")
            return

        for word in self.unknown:
            print(word)


class SimilarityEngine:

    def similarity(self, first, second):
        a = Counter(tokenize(first))
        b = Counter(tokenize(second))

        all_words = set(a) | set(b)

        if not all_words:
            return 0.0

        dot = sum(a[word] * b[word] for word in all_words)

        norm_a = math.sqrt(sum(value ** 2 for value in a.values()))
        norm_b = math.sqrt(sum(value ** 2 for value in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class LanguageEngine:

    def __init__(self):
        self.vocabulary = VocabularyEngine()
        self.similarity = SimilarityEngine()

    def process(self, sentence):
        self.vocabulary.learn_sentence(sentence)

        vector = self.vocabulary.vectorize(sentence)
        unknown = self.vocabulary.find_unknown(sentence)

        print("\n[LANGUAGE ANALYSIS]")
        print("Original sentence:", sentence)
        print("Tokens:", tokenize(sentence))
        print("Vector:", vector)

        if unknown:
            print("New words detected:", unknown)
        else:
            print("All words are in vocabulary.")

    def compare(self, first, second):
        score = self.similarity.similarity(first, second)

        print("\n[SIMILARITY RESULT]")
        print("Sentence 1:", first)
        print("Sentence 2:", second)
        print(f"Similarity: {score * 100:.2f}%")

        return score


def main():
    print("""
========================================
          ULTRON V9.1
      LANGUAGE UNDERSTANDING ENGINE
========================================
""")

    engine = LanguageEngine()

    print("""
Commands:
/learn sentence
/analyze sentence
/compare sentence 1 | sentence 2
/words
/unknown
/stats
/exit
""")

    while True:
        try:
            command = input("\nYOU > ").strip()

            if command == "/exit":
                print("ULTRON V9.1 offline.")
                break

            elif command.startswith("/learn "):
                sentence = command[7:].strip()
                engine.vocabulary.learn_sentence(sentence)
                print("[OK] Sentence learned.")

            elif command.startswith("/analyze "):
                engine.process(command[9:].strip())

            elif command.startswith("/compare "):
                content = command[9:]

                if "|" not in content:
                    print("Use: /compare sentence 1 | sentence 2")
                    continue

                first, second = content.split("|", 1)
                engine.compare(first, second)

            elif command == "/words":
                engine.vocabulary.show_words()

            elif command == "/unknown":
                engine.vocabulary.show_unknown()

            elif command == "/stats":
                engine.vocabulary.statistics()

            elif command == "/":
                print("Type a valid command.")

            else:
                print("Unknown command. Use /help or a valid command.")

        except (KeyboardInterrupt, EOFError):
            print("\nULTRON V9.1 offline.")
            break


if __name__ == "__main__":
    main()
