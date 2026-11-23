
import json
import os
import random
import re

DATA_FILE = "ultron_response_data.json"


class UltronFeedback:

    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (json.JSONDecodeError, OSError):
                print("Training data error.")
                return {}

        print("No training data found.")
        return {}

    def save_data(self):
        with open(DATA_FILE, "w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=4)

    def tokenize(self, sentence):
        return set(re.findall(r"[a-z0-9']+", sentence.lower()))

    def calculate_score(self, message, pattern):
        words1 = self.tokenize(message)
        words2 = self.tokenize(pattern)

        if not words1 or not words2:
            return 0

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
                score = self.calculate_score(message, pattern)

                if score > best_score:
                    best_score = score
                    best_intent = intent

        if best_score >= 0.50:
            return best_intent, best_score

        return None, best_score

    def generate_response(self, message):
        intent, score = self.recognize_intent(message)

        if intent:
            responses = self.data[intent].get("responses", [])

            if responses:
                return random.choice(responses), intent

        return "I don't understand this yet.", None

    def learn_from_feedback(self, message, old_intent):
        print("\nULTRON: Please teach me the correct response.")

        new_response = input("Correct response: ").strip()

        if not new_response:
            print("ULTRON: Empty response cannot be learned.")
            return

        if old_intent:
            intent = old_intent
        else:
            intent = input("New intent name: ").strip().lower()

        if not intent:
            print("Intent name is required.")
            return

        if intent not in self.data:
            self.data[intent] = {
                "patterns": [],
                "responses": []
            }

        if message not in self.data[intent]["patterns"]:
            self.data[intent]["patterns"].append(message)

        if new_response not in self.data[intent]["responses"]:
            self.data[intent]["responses"].append(new_response)

        self.save_data()

        print("ULTRON: I have learned this example!")
        print("ULTRON: My training data has been updated.")

    def show_learning(self):
        print("\n--- ULTRON LEARNING REPORT ---")

        total_patterns = 0
        total_responses = 0

        for intent, details in self.data.items():
            patterns = len(details.get("patterns", []))
            responses = len(details.get("responses", []))

            total_patterns += patterns
            total_responses += responses

            print(f"{intent}: {patterns} patterns, {responses} responses")

        print("\nTotal patterns:", total_patterns)
        print("Total responses:", total_responses)

    def run(self):
        print("\n================================")
        print("      ULTRON FEEDBACK ENGINE")
        print("================================")

        print("Commands: /report | /exit")

        while True:
            message = input("\nYou: ").strip()

            if not message:
                continue

            if message.lower() == "/exit":
                print("ULTRON: Learning session saved.")
                break

            if message.lower() == "/report":
                self.show_learning()
                continue

            response, intent = self.generate_response(message)

            print("ULTRON:", response)

            if intent:
                print("Recognized intent:", intent)

            feedback = input("Was my response correct? (yes/no): ").strip().lower()

            if feedback in ["yes", "y"]:
                print("ULTRON: Thank you for the feedback!")

            elif feedback in ["no", "n"]:
                self.learn_from_feedback(message, intent)

            else:
                print("ULTRON: Feedback skipped.")


if __name__ == "__main__":
    brain = UltronFeedback()
    brain.run()
