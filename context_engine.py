
import json
import os
from datetime import datetime

MEMORY_FILE = "ultron_context.json"
MAX_HISTORY = 20


class UltronContext:

    def __init__(self):
        self.history = []
        self.load_memory()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                    self.history = json.load(file)
                print("Previous conversation loaded.")
            except (json.JSONDecodeError, OSError):
                self.history = []
                print("Starting fresh memory.")

    def save_memory(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)

    def add_message(self, role, message):
        self.history.append({
            "role": role,
            "message": message,
            "time": datetime.now().isoformat(timespec="seconds")
        })

        self.history = self.history[-MAX_HISTORY:]
        self.save_memory()

    def get_previous_user_message(self):
        messages = [
            item["message"]
            for item in self.history
            if item["role"] == "user"
        ]

        if len(messages) >= 2:
            return messages[-2]

        return None

    def get_saved_name(self):
        for item in reversed(self.history):
            message = item["message"]

            if item["role"] == "user" and message.lower().startswith("my name is "):
                return message[11:].strip().title()

        return None

    def show_history(self):
        if not self.history:
            print("No conversation history yet.")
            return

        print("\n--- ULTRON MEMORY ---")

        for item in self.history:
            print(f"{item['role']}: {item['message']}")

    def clear_history(self):
        self.history = []
        self.save_memory()
        print("ULTRON: Conversation history cleared.")


def generate_reply(user_message, brain):

    text = user_message.lower().strip()

    if text in ["hi", "hello", "hey"]:
        return "Hello! I am ULTRON."

    if text.startswith("my name is "):
        name = user_message[11:].strip()

        if name:
            return f"Nice to meet you, {name.title()}!"

    if text in ["what is my name?", "what's my name?"]:
        name = brain.get_saved_name()

        if name:
            return f"Your name is {name}."

        return "You haven't told me your name yet."

    if "what did i say" in text or "previous message" in text:
        previous = brain.get_previous_user_message()

        if previous:
            return f"Your previous message was: {previous}"

        return "I don't have an earlier message to recall."

    if "how are you" in text:
        return "I am functioning properly. Thanks for asking!"

    if text in ["bye", "goodbye", "exit"]:
        return "Goodbye! Conversation saved."

    return "I am still learning to understand context."


def main():

    brain = UltronContext()

    print("\n================================")
    print("       ULTRON CONTEXT ENGINE")
    print("================================")

    print("Commands:")
    print("/history - Show conversation")
    print("/clear   - Clear conversation")
    print("/exit    - Exit ULTRON")

    while True:

        user_message = input("\nYou: ").strip()

        if not user_message:
            continue

        if user_message == "/exit":
            print("ULTRON: Conversation saved.")
            break

        if user_message == "/history":
            brain.show_history()
            continue

        if user_message == "/clear":
            brain.clear_history()
            continue

        reply = generate_reply(user_message, brain)

        brain.add_message("user", user_message)
        brain.add_message("ultron", reply)

        print("ULTRON:", reply)


if __name__ == "__main__":
    main()
