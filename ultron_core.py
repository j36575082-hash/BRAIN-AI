
import json
import os
import random
import re
from datetime import datetime

DATA_FILE = "ultron_response_data.json"
MEMORY_FILE = "ultron_context.json"
MAX_HISTORY = 20


class UltronCore:

    def __init__(self):
        self.data = self.load_data()
        self.history = self.load_history()

    def load_data(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def load_history(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def save_history(self):
        self.history = self.history[-MAX_HISTORY:]

        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)

    def remember(self, role, message):
        self.history.append({
            "role": role,
            "message": message,
            "time": datetime.now().isoformat(timespec="seconds")
        })
        self.save_history()

    @staticmethod
    def normalize(text):
        return " ".join(
            re.findall(r"[a-z0-9']+", text.lower())
        )

    def similarity(self, message, pattern):
        words1 = set(self.normalize(message).split())
        words2 = set(self.normalize(pattern).split())

        if not words1 or not words2:
            return 0

        if self.normalize(message) == self.normalize(pattern):
            return 1.0

        common = words1.intersection(words2)

        if not common:
            return 0

        precision = len(common) / len(words1)
        recall = len(common) / len(words2)

        return 2 * precision * recall / (precision + recall)

    def recognize_intent(self, message):
        best_intent = None
        best_score = 0

        for intent, details in self.data.items():
            for pattern in details.get("patterns", []):
                score = self.similarity(message, pattern)

                if score > best_score:
                    best_score = score
                    best_intent = intent

        if best_score >= 0.50:
            return best_intent, best_score

        return None, best_score

    def get_name(self):
        for item in reversed(self.history):
            if item["role"] != "user":
                continue

            normalized = self.normalize(item["message"])

            if normalized.startswith("my name is "):
                return item["message"][item["message"].lower().find("my name is") + 11:].strip().title()

        return None

    def get_previous_message(self):
        user_messages = [
            item["message"]
            for item in self.history
            if item["role"] == "user"
        ]

        if len(user_messages) >= 1:
            return user_messages[-1]

        return None

    def generate_response(self, message):
        text = self.normalize(message)

        if text in ["what is my name", "what's my name"]:
            name = self.get_name()

            if name:
                return f"Your name is {name}."

            return "You haven't told me your name yet."

        if text.startswith("my name is "):
            name = message[message.lower().find("my name is") + 11:].strip()
            return f"Nice to meet you, {name.title()}!"

        if text in [
            "what did i say previously",
            "what was my previous message",
            "what did i say before"
        ]:
            previous = self.get_previous_message()

            if previous:
                return f"Your previous message was: {previous}"

            return "I don't have an earlier message."

        intent, score = self.recognize_intent(message)

        if intent:
            responses = self.data[intent].get("responses", [])

            if responses:
                return random.choice(responses)

        return "I don't understand this yet. I need more training data."

    def chat(self):
        print("\n================================")
        print("       ULTRON CORE ENGINE")
        print("================================")
        print("Commands: /history | /clear | /exit")

        while True:
            message = input("\nYou: ").strip()

            if not message:
                continue

            command = message.lower()

            if command == "/exit":
                print("ULTRON: Memory saved. Goodbye!")
                break

            if command == "/history":
                for item in self.history:
                    print(f"{item['role']}: {item['message']}")
                continue

            if command == "/clear":
                self.history = []
                self.save_history()
                print("ULTRON: Memory cleared.")
                continue

            response = self.generate_response(message)

            self.remember("user", message)
            self.remember("ultron", response)

            print("ULTRON:", response)


if __name__ == "__main__":
    ultron = UltronCore()
    ultron.chat()
