
import json
import random
import re
import math
from difflib import SequenceMatcher

RESPONSE_FILE = "ultron_response_data.json"
NEURAL_FILE = "ultron_neural_core.json"
LANGUAGE_FILE = "ultron_language_model.json"
CONTEXT_FILE = "ultron_context.json"
MEMORY_FILE = "ultron_memory.json"
LEARNED_FILE = "ultron_learned_data.json"


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


def normalize(text):
    return " ".join(tokenize(text))


# ==========================================
# LEARNED KNOWLEDGE
# ==========================================

class KnowledgeEngine:

    def __init__(self):
        self.data = load_json(LEARNED_FILE, [])

        if not isinstance(self.data, list):
            self.data = []

    def find_answer(self, question):

        normalized = normalize(question)

        if not normalized:
            return None

        best_score = 0
        best_answer = None

        for item in self.data:

            if not isinstance(item, dict):
                continue

            saved_question = item.get("question", "")
            saved_answer = item.get("answer", "")

            if not saved_question or not saved_answer:
                continue

            score = SequenceMatcher(
                None,
                normalized,
                normalize(saved_question)
            ).ratio()

            if score > best_score:
                best_score = score
                best_answer = saved_answer

        if best_score >= 0.82:
            return best_answer

        return None

    def teach(self, question, answer):

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return False

        normalized = normalize(question)

        for item in self.data:

            if normalize(item.get("question", "")) == normalized:
                item["answer"] = answer
                save_json(LEARNED_FILE, self.data)
                return True

        self.data.append({
            "question": question,
            "answer": answer
        })

        save_json(LEARNED_FILE, self.data)
        return True


# ==========================================
# LANGUAGE GENERATOR
# ==========================================

class LanguageGenerator:

    def __init__(self):
        self.model = load_json(LANGUAGE_FILE, {})

    def generate(self, start_word, length=15):

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
# NEURAL BRAIN
# ==========================================

class NeuralBrain:

    def __init__(self):

        model = load_json(NEURAL_FILE, {})

        self.vocabulary = model.get("vocabulary", [])
        self.intents = model.get("intents", [])
        self.weights1 = model.get("weights1", [])
        self.weights2 = model.get("weights2", [])
        self.bias1 = model.get("bias1", [])
        self.bias2 = model.get("bias2", [])

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

            if len(self.weights1) != len(self.bias1):
                return False

            if any(
                len(row) != len(self.vocabulary)
                for row in self.weights1
            ):
                return False

            if len(self.weights2) != len(self.intents):
                return False

            if any(
                len(row) != len(self.bias1)
                for row in self.weights2
            ):
                return False

            return len(self.bias2) == len(self.intents)

        except (TypeError, IndexError):
            return False

    def sigmoid(self, x):

        x = max(-60, min(60, x))
        return 1 / (1 + math.exp(-x))

    def predict(self, sentence):

        if not self.is_valid():
            return None

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

        return self.intents[best]


# ==========================================
# ULTRON V4
# ==========================================

class Ultron:

    def __init__(self):

        self.brain = NeuralBrain()
        self.language = LanguageGenerator()
        self.knowledge = KnowledgeEngine()

        self.responses = load_json(RESPONSE_FILE, {})
        self.history = load_json(CONTEXT_FILE, [])
        self.memory = load_json(MEMORY_FILE, {})

        if not isinstance(self.history, list):
            self.history = []

        if not isinstance(self.memory, dict):
            self.memory = {}

        print("\nNeural brain:", "Connected" if self.brain.is_valid() else "Fallback mode")
        print("Language generator: Connected")
        print("Knowledge engine: Connected")
        print("Memory: Connected")

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
    # MAIN RESPONSE ENGINE
    # --------------------------------------

    def respond(self, message):

        text = message.strip()
        lower = text.lower()

        if not text:
            return "Please enter a message."

        # Name learning
        match = re.search(
            r"\bmy name is ([a-zA-Z]+)",
            text,
            re.IGNORECASE
        )

        if match:

            name = match.group(1).capitalize()
            self.memory["name"] = name
            save_json(MEMORY_FILE, self.memory)

            return f"Nice to meet you, {name}!"

        # Name recall
        if "what is my name" in lower:

            name = self.memory.get("name")

            return (
                f"Your name is {name}."
                if name else "I don't know your name yet."
            )

        # Previous message
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

        # Check learned knowledge FIRST
        learned_answer = self.knowledge.find_answer(text)

        if learned_answer:
            return learned_answer

        # Neural intent
        intent = self.brain.predict(text)

        response = self.find_response(intent)

        if response:
            return response

        # Language generator fallback
        for word in tokenize(text):

            if word in self.language.model:

                generated = self.language.generate(word)

                if len(generated.split()) > 2:
                    return generated.capitalize()

        return "I don't know that yet. You can teach me using /teach."

    # --------------------------------------
    # TEACH COMMAND
    # --------------------------------------

    def teach(self, message):

        content = message[len("/teach"):].strip()

        if "|" not in content:
            return "Format: /teach question | answer"

        question, answer = content.split("|", 1)

        if self.knowledge.teach(question, answer):
            return "I have learned this new information!"

        return "Please provide both a question and an answer."

    # --------------------------------------
    # HISTORY
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
        print("          ULTRON V4")
        print("================================")

        print("Commands: /history | /clear | /teach | /exit")

        while True:

            message = input("\nYou: ").strip()

            if message.lower() == "/exit":

                save_json(CONTEXT_FILE, self.history)
                print("ULTRON: Memory saved. Shutting down.")
                break

            if message.lower() == "/history":

                self.show_history()
                continue

            if message.lower() == "/clear":

                self.history = []
                save_json(CONTEXT_FILE, [])

                print("ULTRON: Conversation history cleared.")
                continue

            if message.lower().startswith("/teach"):

                response = self.teach(message)
                print("ULTRON:", response)

                self.save_history(message, response)
                continue

            response = self.respond(message)

            print("ULTRON:", response)

            self.save_history(message, response)


if __name__ == "__main__":

    ultron = Ultron()
    ultron.run()
