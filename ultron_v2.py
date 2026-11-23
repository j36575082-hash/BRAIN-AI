
import json
import os
import re
import math
import random
from datetime import datetime

DATA_FILE = "ultron_response_data.json"
MODEL_FILE = "ultron_neural_core.json"
MEMORY_FILE = "ultron_context.json"

MAX_HISTORY = 20


class UltronV2:

    def __init__(self):
        self.data = self.load_json(DATA_FILE, {})
        self.history = self.load_json(MEMORY_FILE, [])
        self.model = self.load_json(MODEL_FILE, None)

        if self.model:
            self.vocabulary = self.model["vocabulary"]
            self.intents = self.model["intents"]
            self.weights1 = self.model["weights1"]
            self.bias1 = self.model["bias1"]
            self.weights2 = self.model["weights2"]
            self.bias2 = self.model["bias2"]
            print("Neural brain connected!")
        else:
            self.vocabulary = []
            self.intents = []
            print("Neural model not found. Using fallback mode.")

    @staticmethod
    def load_json(filename, default):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

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
        return re.findall(r"[a-z0-9']+", text.lower())

    def vectorize(self, message):
        words = set(self.normalize(message))
        return [
            1.0 if word in words else 0.0
            for word in self.vocabulary
        ]

    @staticmethod
    def sigmoid(value):
        value = max(-50, min(50, value))
        return 1 / (1 + math.exp(-value))

    def neural_predict(self, message):
        if not self.vocabulary or not self.intents:
            return None, 0

        inputs = self.vectorize(message)

        hidden = []

        for h in range(len(self.weights1)):
            total = self.bias1[h]

            for i in range(len(inputs)):
                total += self.weights1[h][i] * inputs[i]

            hidden.append(self.sigmoid(total))

        outputs = []

        for o in range(len(self.weights2)):
            total = self.bias2[o]

            for h in range(len(hidden)):
                total += self.weights2[o][h] * hidden[h]

            outputs.append(self.sigmoid(total))

        best_index = outputs.index(max(outputs))

        return self.intents[best_index], outputs[best_index]

    def get_name(self):
        for item in reversed(self.history):
            if item["role"] == "user":
                message = item["message"]

                match = re.match(
                    r"my name is\s+(.+)",
                    message,
                    re.IGNORECASE
                )

                if match:
                    return match.group(1).strip().title()

        return None

    def get_previous_user_message(self):
        messages = [
            item["message"]
            for item in self.history
            if item["role"] == "user"
        ]

        return messages[-1] if messages else None

    def generate_response(self, message):
        text = " ".join(self.normalize(message))

        if text in ["what is my name", "what's my name"]:
            name = self.get_name()
            return f"Your name is {name}." if name else "You haven't told me your name."

        if text.startswith("my name is "):
            name = message[message.lower().find("my name is") + 11:].strip()
            return f"Nice to meet you, {name.title()}!"

        if text in [
            "what did i say previously",
            "what was my previous message"
        ]:
            previous = self.get_previous_user_message()
            return f"Your previous message was: {previous}" if previous else "No previous message found."

        intent, confidence = self.neural_predict(message)

        if intent and confidence >= 0.50:
            responses = self.data.get(intent, {}).get("responses", [])

            if responses:
                return random.choice(responses)

        return "I don't understand this yet. I need more training data."

    def chat(self):
        print("\n================================")
        print("          ULTRON V2")
        print("================================")
        print("Commands: /history | /clear | /exit")

        while True:
            message = input("\nYou: ").strip()

            if not message:
                continue

            command = message.lower()

            if command == "/exit":
                print("ULTRON: Shutting down. Memory saved.")
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
    ultron = UltronV2()
    ultron.chat()
