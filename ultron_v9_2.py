
import json
import os
import re
import math
from collections import Counter
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

VOCAB_FILE = os.path.join(
    BASE, "ultron_v9_1_vocabulary.json"
)

UNKNOWN_FILE = os.path.join(
    BASE, "ultron_v9_2_unknown_words.json"
)

LOG_FILE = os.path.join(
    BASE, "ultron_v9_2_activity.json"
)


def load_json(filename, default):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(filename, data):
    temp = filename + ".tmp"

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(temp, filename)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class VocabularyEngine:

    def __init__(self):
        self.data = load_json(VOCAB_FILE, {
            "words": {},
            "total_sentences": 0
        })

        if not isinstance(self.data, dict):
            self.data = {"words": {}, "total_sentences": 0}

        if not isinstance(self.data.get("words"), dict):
            self.data["words"] = {}

        self.data.setdefault("total_sentences", 0)

        self.unknown = load_json(UNKNOWN_FILE, {})

        if not isinstance(self.unknown, dict):
            self.unknown = {}

        self.activity = load_json(LOG_FILE, [])

        if not isinstance(self.activity, list):
            self.activity = []

    def log(self, action, sentence):
        self.activity.append({
            "action": action,
            "sentence": sentence,
            "time": datetime.now().isoformat(timespec="seconds")
        })

        self.activity = self.activity[-500:]
        save_json(LOG_FILE, self.activity)

    def learn(self, sentence):
        words = tokenize(sentence)

        if not words:
            return False

        for word in words:
            self.data["words"][word] = (
                self.data["words"].get(word, 0) + 1
            )

            if word in self.unknown:
                del self.unknown[word]

        self.data["total_sentences"] += 1

        save_json(VOCAB_FILE, self.data)
        save_json(UNKNOWN_FILE, self.unknown)

        self.log("learn", sentence)

        return True

    def analyze(self, sentence):
        words = tokenize(sentence)
        known = self.data["words"]

        known_words = [
            word for word in words if word in known
        ]

        unknown_words = [
            word for word in words if word not in known
        ]

        counts = Counter(words)

        vector = {
            word: counts[word]
            for word in known
        }

        self.log("analyze", sentence)

        return {
            "tokens": words,
            "vector": vector,
            "known": known_words,
            "unknown": unknown_words
        }

    def compare(self, first, second):
        a = Counter(tokenize(first))
        b = Counter(tokenize(second))

        all_words = set(a) | set(b)

        if not all_words:
            return 0.0, []

        dot = sum(a[w] * b[w] for w in all_words)

        norm_a = math.sqrt(sum(x * x for x in a.values()))
        norm_b = math.sqrt(sum(x * x for x in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0, []

        score = dot / (norm_a * norm_b)

        common = sorted(
            word for word in set(a) & set(b)
        )

        return score, common

    def record_unknown(self, words):
        for word in words:
            self.unknown[word] = self.unknown.get(word, 0) + 1

        save_json(UNKNOWN_FILE, self.unknown)

    def show_words(self):
        print("\n========== KNOWN VOCABULARY ==========")

        if not self.data["words"]:
            print("Vocabulary is empty.")
            return

        for word, count in sorted(
            self.data["words"].items(),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{word}: {count}")

    def show_unknown(self):
        print("\n========== UNKNOWN WORDS ==========")

        if not self.unknown:
            print("No unknown words detected.")
            return

        for word, count in sorted(
            self.unknown.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            print(f"{word}: detected {count} times")

    def stats(self):
        print("\n========== V9.2 STATISTICS ==========")
        print("Unique words:", len(self.data["words"]))
        print("Learned sentences:", self.data["total_sentences"])
        print("Unknown word types:", len(self.unknown))
        print("Activity records:", len(self.activity))
        print("=====================================")

    def show_history(self):
        print("\n========== ACTIVITY HISTORY ==========")

        if not self.activity:
            print("No activity.")
            return

        for item in self.activity[-20:]:
            print(
                f"{item['action'].upper()}: "
                f"{item['sentence']}"
            )


class UltronLanguageEngine:

    def __init__(self):
        self.vocabulary = VocabularyEngine()

    def learn(self, sentence):
        if self.vocabulary.learn(sentence):
            print("[OK] Sentence learned and saved.")
        else:
            print("[ERROR] Empty sentence.")

    def analyze(self, sentence):
        result = self.vocabulary.analyze(sentence)

        self.vocabulary.record_unknown(
            result["unknown"]
        )

        print("\n========== LANGUAGE ANALYSIS ==========")
        print("Sentence:", sentence)
        print("Tokens:", result["tokens"])
        print("Known words:", result["known"])
        print("Unknown words:", result["unknown"])
        print("Vector:", result["vector"])

        print("\n[OK] Analysis complete.")
        print("[INFO] Vocabulary was not modified.")

    def compare(self, first, second):
        score, common = self.vocabulary.compare(
            first, second
        )

        print("\n========== SENTENCE COMPARISON ==========")
        print("Sentence 1:", first)
        print("Sentence 2:", second)
        print(f"Similarity: {score * 100:.2f}%")
        print("Common words:", common)

        print(
            "[NOTE] Similarity measures word overlap, "
            "not actual semantic understanding."
        )


def main():
    print("""
========================================
             ULTRON V9.2
      SMART LANGUAGE ENGINE
========================================
""")

    engine = UltronLanguageEngine()

    print("""
COMMANDS:

/learn sentence
/analyze sentence
/compare sentence 1 | sentence 2
/words
/unknown
/history
/stats
/help
/exit

========================================
""")

    while True:
        try:
            command = input("\nYOU > ").strip()

            if command == "/exit":
                print("ULTRON V9.2 offline.")
                break

            elif command == "/help":
                print("""
/learn sentence
/analyze sentence
/compare sentence 1 | sentence 2
/words
/unknown
/history
/stats
/exit
""")

            elif command.startswith("/learn "):
                engine.learn(command[7:].strip())

            elif command.startswith("/analyze "):
                engine.analyze(command[9:].strip())

            elif command.startswith("/compare "):
                content = command[9:]

                if "|" not in content:
                    print("Use: /compare sentence 1 | sentence 2")
                    continue

                first, second = content.split("|", 1)

                engine.compare(
                    first.strip(),
                    second.strip()
                )

            elif command == "/words":
                engine.vocabulary.show_words()

            elif command == "/unknown":
                engine.vocabulary.show_unknown()

            elif command == "/history":
                engine.vocabulary.show_history()

            elif command == "/stats":
                engine.vocabulary.stats()

            elif command.startswith("/"):
                print("[ERROR] Unknown command. Type /help.")

            else:
                print("Use /analyze or /learn before entering a sentence.")

        except (KeyboardInterrupt, EOFError):
            print("\nULTRON V9.2 offline.")
            break


if __name__ == "__main__":
    main()
