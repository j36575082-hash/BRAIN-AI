
import json
import random
import re
import math

RESPONSE_FILE = "ultron_response_data.json"
NEURAL_FILE = "ultron_neural_core.json"
LANGUAGE_FILE = "ultron_language_model.json"
CONTEXT_FILE = "ultron_context.json"
MEMORY_FILE = "ultron_memory.json"


# ==========================================
# FILE MANAGEMENT
# ==========================================

def load_json(filename, default):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


# ==========================================
# LANGUAGE GENERATOR
# ==========================================

class LanguageGenerator:

    def __init__(self):
        self.model = load_json(LANGUAGE_FILE, {})

    def generate(self, start_word, length=18):
        current = start_word.lower()
        words = [current]

        for _ in range(length - 1):
            options = self.model.get(current)

            if not isinstance(options, list) or not options:
                break

            current = random.choice(options)
            words.append(current)

            if current in [".", "!", "?"]:
                break

        return " ".join(words)


# ==========================================
# NEURAL BRAIN - FIXED MODEL FORMAT
# ==========================================

class NeuralBrain:

    def __init__(self):
        self.model = load_json(NEURAL_FILE, {})

        self.vocabulary = self.model.get("vocabulary", [])
        self.intents = self.model.get("intents", [])
        self.weights1 = self.model.get("weights1", [])
        self.weights2 = self.model.get("weights2", [])
        self.bias1 = self.model.get("bias1", [])
        self.bias2 = self.model.get("bias2", [])

    def sigmoid(self, x):
        x = max(-60, min(60, x))
        return 1 / (1 + math.exp(-x))

    def is_valid(self):
        try:
            if not all([
                self.vocabulary,
                self.intents,
                self.weights1,
                self.weights2,
                self.bias1,
                self.bias2
            ]):
                return False

            # weights1 = hidden neurons x vocabulary
            if len(self.weights1) != len(self.bias1):
                return False

            if any(
                len(row) != len(self.vocabulary)
                for row in self.weights1
            ):
                return False

            # weights2 = intents x hidden neurons
            if len(self.weights2) != len(self.intents):
                return False

            if any(
                len(row) != len(self.bias1)
                for row in self.weights2
            ):
                return False

            if len(self.bias2) != len(self.intents):
                return False

            return True

        except (TypeError, IndexError):
            return False

    def predict(self, sentence):

        if not self.is_valid():
            return None, 0

        words = set(tokenize(sentence))

        inputs = [
            1 if word in words else 0
            for word in self.vocabulary
        ]

        hidden = []

        for j in range(len(self.bias1)):
            total = self.bias1[j]

            for i in range(len(inputs)):
                total += self.weights1[j][i] * inputs[i]

            hidden.append(self.sigmoid(total))

        outputs = []

        for j in range(len(self.bias2)):
            total = self.bias2[j]

            for i in range(len(hidden)):
                total += self.weights2[j][i] * hidden[i]

            outputs.append(self.sigmoid(total))

        best = max(range(len(outputs)), key=lambda i: outputs[i])

        return self.intents[best], outputs[best]


# ==========================================
# ULTRON V3
# ==========================================

class Ultron:

    def __init__(self):

        self.brain = NeuralBrain()
        self.language = LanguageGenerator()

        self.responses = load_json(RESPONSE_FILE, {})
        self.history = load_json(CONTEXT_FILE, [])
        self.memory = load_json(MEMORY_FILE, {})

        if not isinstance(self.history, list):
            self.history = []

        if not isinstance(self.memory, dict):
            self.memory = {}

        if self.brain.is_valid():
            print("Neural brain connected!")
        else:
            print("Neural model incompatible. Using fallback mode.")

        print("Language generator connected!")
        print("Memory connected!")

    # --------------------------------------
    # MEMORY
    # --------------------------------------

    def remember(self, key, value):
        self.memory[key] = value
        save_json(MEMORY_FILE, self.memory)

    def get_name(self):
        return self.memory.get("name")

    # --------------------------------------
    # RESPONSE DATABASE
    # --------------------------------------

    def find_response(self, intent):

        if not intent:
            return None

        data = self.responses.get(intent)

        if isinstance(data, dict):
            options = data.get("responses", [])

            if options:
                return random.choice(options)

        elif isinstance(data, list) and data:
            return random.choice(data)

        return None

    # --------------------------------------
    # RESPONSE ENGINE
    # --------------------------------------

    def respond(self, message):

        text = message.strip()
        lower = text.lower()

        if not text:
            return "Please enter a message."

        # Name detection
        match = re.search(
            r"\bmy name is ([a-zA-Z]+)",
            text,
            re.IGNORECASE
        )

        if match:
            name = match.group(1).capitalize()
            self.remember("name", name)
            return f"Nice to meet you, {name}!"

        # Name recall
        if "what is my name" in lower:
            name = self.get_name()

            if name:
                return f"Your name is {name}."

            return "I don't know your name yet."

        # Previous user message
        if "what did i say previously" in lower:

            for item in reversed(self.history):

                if item.get("role") == "user":
                    return "Your previous message was: " + item["text"]

            return "I don't have previous messages."

        # Previous ULTRON response
        if "what did you say previously" in lower:

            for item in reversed(self.history):

                if item.get("role") == "ultron":
                    return "My previous response was: " + item["text"]

            return "I don't have a previous response."

        # Neural prediction
        intent = None

        try:
            intent, confidence = self.brain.predict(text)

        except (IndexError, TypeError, ValueError, KeyError):
            intent = None

        # Find stored response
        response = self.find_response(intent)

        if response:
            return response

        # Language generation fallback
        for word in tokenize(text):

            if word in self.language.model:

                generated = self.language.generate(word)

                if len(generated.split()) > 2:
                    return generated.capitalize()

        return "I am still learning how to respond to that."

    # --------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------

    def save_history(self, message, response):

        self.history.append({
            "role": "user",
            "text": message
        })

        self.history.append({
            "role": "ultron",
            "text": response
        })

        self.history = self.history[-40:]

        save_json(CONTEXT_FILE, self.history)

    def show_history(self):

        if not self.history:
            print("No conversation history.")
            return

        for item in self.history[-20:]:

            print(
                item.get("role", "unknown")
                + ": "
                + item.get("text", "")
            )

    # --------------------------------------
    # MAIN LOOP
    # --------------------------------------

    def run(self):

        print("\n================================")
        print("          ULTRON V3")
        print("================================")
        print("Commands: /history | /clear | /exit")

        while True:

            message = input("\nYou: ").strip()

            if message.lower() == "/exit":

                save_json(CONTEXT_FILE, self.history)

                print("ULTRON: Shutting down. Memory saved.")
                break

            if message.lower() == "/history":

                self.show_history()
                continue

            if message.lower() == "/clear":

                self.history = []
                save_json(CONTEXT_FILE, [])

                print("ULTRON: Conversation history cleared.")
                continue

            response = self.respond(message)

            print("ULTRON:", response)

            self.save_history(message, response)


# ==========================================
# START ULTRON
# ==========================================

if __name__ == "__main__":

    ultron = Ultron()
    ultron.run()
