
import json
import os
import re
from datetime import datetime

# ==========================================
#       ULTRON - LANGUAGE BRAIN
#       LEVEL 4 : NLP v1
# ==========================================

MEMORY_FILE = "ultron_memory.json"

print("\n====================================")
print("       ULTRON LANGUAGE SYSTEM")
print("       NLP ENGINE VERSION 1.0")
print("====================================")


# ==========================================
# TEXT PROCESSING
# ==========================================

def tokenize(text):
    text = text.lower()
    words = re.findall(r"[a-z0-9']+", text)
    return words


def clean_text(text):
    return " ".join(tokenize(text))


# ==========================================
# MEMORY SYSTEM
# ==========================================

def load_memory():

    if os.path.exists(MEMORY_FILE):

        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                return data

        except (OSError, ValueError):
            print("Memory loading failed.")

    return {}


memory = load_memory()


def save_memory():

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


def remember_information(key, value):

    key = key.strip()
    value = value.strip()

    if not key or not value:
        return False

    memory[key.lower()] = {
        "name": key,
        "information": value
    }

    save_memory()
    return True


def recall_information(key):

    key = key.lower().strip()

    if key in memory:
        return memory[key]["information"]

    return None


# ==========================================
# INTENT RECOGNITION
# ==========================================

def detect_intent(words):

    word_set = set(words)

    if word_set.intersection({"hello", "hi", "hey"}):
        return "greeting"

    if "how" in word_set and "you" in word_set:
        return "how_are_you"

    if word_set.intersection({"bye", "goodbye"}):
        return "goodbye"

    if "help" in word_set:
        return "help"

    if "time" in word_set:
        return "time"

    if word_set.intersection({"creator", "created"}):
        return "creator"

    if "name" in word_set and word_set.intersection(
        {"your", "you", "my"}
    ):
        return "name"

    if "remember" in word_set:
        return "remember"

    if word_set.intersection({"who", "what", "tell"}):
        return "question"

    return "unknown"


# ==========================================
# RESPONSE ENGINE
# ==========================================

def generate_response(user_text):

    words = tokenize(user_text)
    intent = detect_intent(words)

    # GREETING
    if intent == "greeting":
        return "Hello! I am ULTRON. How can I help you?"

    # HOW ARE YOU
    elif intent == "how_are_you":
        return "I am operational and ready to learn."

    # GOODBYE
    elif intent == "goodbye":
        return "Goodbye! ULTRON is shutting down conversation."

    # HELP
    elif intent == "help":
        return (
            "You can greet me, ask questions, "
            "save information or recall memories."
        )

    # TIME
    elif intent == "time":
        return "Current system time: " + datetime.now().strftime(
            "%I:%M %p"
        )

    # CREATOR
    elif intent == "creator":

        result = recall_information("Creator")

        if result:
            return "My creator is " + result

        return "I don't know my creator yet."

    # NAME
    elif intent == "name":

        if "your" in words or "you" in words:
            return "My name is ULTRON."

        result = recall_information("Name")

        if result:
            return "Your name is " + result

        return "I don't know your name yet."

    # REMEMBER INFORMATION
    elif intent == "remember":

        match = re.search(
            r"remember\s+(.+?)\s+is\s+(.+)",
            user_text,
            re.IGNORECASE
        )

        if match:

            key = match.group(1).strip()
            value = match.group(2).strip()

            if remember_information(key, value):
                return "I have saved that information in my memory."

        return (
            "Use this format: remember your_name is Harsh"
        )

    # MEMORY QUESTIONS
    elif intent == "question":

        match = re.search(
            r"(?:what|who|tell me)\s+(?:is|are|about)?\s*(.+?)[?.!]*$",
            user_text,
            re.IGNORECASE
        )

        if match:

            key = clean_text(match.group(1))

            result = recall_information(key)

            if result:
                return result

        return "I don't have enough information to answer that."

    return (
        "I don't understand that yet. "
        "I am still learning language patterns."
    )


# ==========================================
# CHAT SYSTEM
# ==========================================

def start_chat():

    print("\nULTRON: Language system activated.")
    print("ULTRON: Memory entries:", len(memory))
    print("\nType 'help' for commands.")
    print("Type 'exit' to close conversation.")

    while True:

        user_text = input("\nYou: ").strip()

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit", "shutdown"]:
            print("\nULTRON: Conversation ended.")
            break

        response = generate_response(user_text)

        print("\nULTRON:", response)


# ==========================================
# START ENGINE
# ==========================================

if __name__ == "__main__":
    start_chat()
