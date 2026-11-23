
import json
import os
import re
import math
import random
from datetime import datetime

# ==========================================
# ULTRON V9 - NEURAL LEARNING ENGINE
# Built using Python and mathematics
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "training": "ultron_v9_training.json",
    "model": "ultron_v9_model.json",
    "knowledge": "ultron_v9_knowledge.json",
    "memory": "ultron_v9_memory.json",
    "unknown": "ultron_v9_unknown.json",
    "feedback": "ultron_v9_feedback.json",
    "history": "ultron_v9_history.json"
}


def path(name):
    return os.path.join(BASE_DIR, FILES[name])


def load_json(name, default):
    try:
        with open(path(name), "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save_json(name, data):
    target = path(name)
    temporary = target + ".tmp"

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    os.replace(temporary, target)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def normalize(text):
    return " ".join(tokenize(text))


# ==========================================
# DEFAULT TRAINING DATA
# ==========================================

DEFAULT_EXAMPLES = [
    {"text": "hello", "intent": "greeting"},
    {"text": "hi", "intent": "greeting"},
    {"text": "hey", "intent": "greeting"},
    {"text": "good morning", "intent": "greeting"},
    {"text": "how are you", "intent": "greeting"},
    {"text": "how is everything", "intent": "greeting"},

    {"text": "thank you", "intent": "thanks"},
    {"text": "thanks", "intent": "thanks"},
    {"text": "I appreciate it", "intent": "thanks"},

    {"text": "who are you", "intent": "identity"},
    {"text": "what are you", "intent": "identity"},
    {"text": "what is your name", "intent": "identity"},

    {"text": "bye", "intent": "goodbye"},
    {"text": "goodbye", "intent": "goodbye"},
    {"text": "see you later", "intent": "goodbye"}
]


RESPONSES = {
    "greeting": [
        "Hello! I am ULTRON.",
        "Hi! How can I help you?",
        "System online. What would you like to discuss?"
    ],
    "thanks": [
        "You're welcome!",
        "Happy to help.",
        "Acknowledged."
    ],
    "identity": [
        "I am ULTRON, an experimental learning system.",
        "I am ULTRON V9, built using a neural learning engine."
    ],
    "goodbye": [
        "Goodbye!",
        "Shutting down conversation. See you again."
    ]
}


# ==========================================
# NEURAL NETWORK
# ==========================================

class NeuralBrain:

    def __init__(self):
        self.hidden_size = 16
        self.vocabulary = []
        self.labels = []
        self.w1 = []
        self.b1 = []
        self.w2 = []
        self.b2 = []

        self.examples = load_json("training", [])

        if not isinstance(self.examples, list):
            self.examples = []

        self.add_default_examples()

        saved = load_json("model", {})

        if self.load_model(saved):
            print("[OK] Saved neural model loaded")
        else:
            print("[INFO] Creating first neural model")
            self.train()

    def add_default_examples(self):
        existing = {
            (normalize(x.get("text", "")), x.get("intent"))
            for x in self.examples
            if isinstance(x, dict)
        }

        for example in DEFAULT_EXAMPLES:
            key = (
                normalize(example["text"]),
                example["intent"]
            )

            if key not in existing:
                self.examples.append(example.copy())

        save_json("training", self.examples)

    def initialize(self):
        self.vocabulary = sorted({
            word
            for item in self.examples
            for word in tokenize(item["text"])
        })

        self.labels = sorted({
            item["intent"]
            for item in self.examples
        })

        if not self.vocabulary or not self.labels:
            raise ValueError("Training data is empty")

        rng = random.Random(42)

        input_size = len(self.vocabulary)
        hidden_size = self.hidden_size
        output_size = len(self.labels)

        self.w1 = [
            [rng.uniform(-0.15, 0.15) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]

        self.b1 = [0.0] * hidden_size

        self.w2 = [
            [rng.uniform(-0.15, 0.15) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]

        self.b2 = [0.0] * output_size

    def vectorize(self, text):
        words = set(tokenize(text))

        return [
            1.0 if word in words else 0.0
            for word in self.vocabulary
        ]

    @staticmethod
    def softmax(values):
        maximum = max(values)
        exponents = [
            math.exp(max(-50, min(50, value - maximum)))
            for value in values
        ]

        total = sum(exponents)

        return [value / total for value in exponents]

    def forward(self, inputs):
        hidden = []

        for j in range(self.hidden_size):
            value = self.b1[j]

            for i in range(len(inputs)):
                value += self.w1[j][i] * inputs[i]

            hidden.append(math.tanh(value))

        output_values = []

        for k in range(len(self.labels)):
            value = self.b2[k]

            for j in range(self.hidden_size):
                value += self.w2[k][j] * hidden[j]

            output_values.append(value)

        probabilities = self.softmax(output_values)

        return hidden, probabilities

    def train(self, epochs=350, learning_rate=0.05):
        print("\n[TRAINING] Initializing neural network...")

        self.initialize()

        samples = []

        for item in self.examples:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")
            intent = item.get("intent", "")

            if normalize(text) and intent in self.labels:
                samples.append((
                    self.vectorize(text),
                    self.labels.index(intent)
                ))

        if not samples:
            print("[ERROR] No valid training examples")
            return

        rng = random.Random(123)

        for epoch in range(epochs):
            rng.shuffle(samples)

            for inputs, target in samples:

                hidden, probabilities = self.forward(inputs)

                # Output layer error
                output_error = probabilities[:]
                output_error[target] -= 1.0

                # Hidden layer error
                hidden_error = [0.0] * self.hidden_size

                for j in range(self.hidden_size):
                    for k in range(len(self.labels)):
                        hidden_error[j] += (
                            self.w2[k][j] * output_error[k]
                        )

                hidden_error = [
                    hidden_error[j] * (1.0 - hidden[j] ** 2)
                    for j in range(self.hidden_size)
                ]

                # Update output weights and biases
                for k in range(len(self.labels)):
                    for j in range(self.hidden_size):
                        self.w2[k][j] -= (
                            learning_rate *
                            output_error[k] *
                            hidden[j]
                        )

                    self.b2[k] -= learning_rate * output_error[k]

                # Update input weights and biases
                for j in range(self.hidden_size):
                    for i in range(len(inputs)):
                        self.w1[j][i] -= (
                            learning_rate *
                            hidden_error[j] *
                            inputs[i]
                        )

                    self.b1[j] -= learning_rate * hidden_error[j]

            if epoch % 100 == 0:
                print(f"[TRAINING] Epoch {epoch}/{epochs}")

        save_json("model", self.export_model())

        correct = 0

        for inputs, target in samples:
            _, probabilities = self.forward(inputs)

            if probabilities.index(max(probabilities)) == target:
                correct += 1

        accuracy = correct / len(samples) * 100

        print("[OK] Training complete")
        print(f"[RESULT] Training accuracy: {accuracy:.2f}%")
        print("[OK] Neural model saved")

    def predict(self, text):
        if not self.vocabulary:
            return None, 0.0

        inputs = self.vectorize(text)

        if not any(inputs):
            return None, 0.0

        _, probabilities = self.forward(inputs)

        index = probabilities.index(max(probabilities))
        confidence = probabilities[index]

        if confidence < 0.45:
            return None, confidence

        return self.labels[index], confidence

    def teach(self, text, intent):
        if not normalize(text) or not intent:
            return False

        self.examples.append({
            "text": text.strip(),
            "intent": intent.strip().lower()
        })

        save_json("training", self.examples)
        return True

    def export_model(self):
        return {
            "version": 9,
            "vocabulary": self.vocabulary,
            "labels": self.labels,
            "hidden_size": self.hidden_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        }

    def load_model(self, model):
        try:
            if model.get("version") != 9:
                return False

            vocabulary = model["vocabulary"]
            labels = model["labels"]
            hidden_size = model["hidden_size"]

            w1 = model["w1"]
            b1 = model["b1"]
            w2 = model["w2"]
            b2 = model["b2"]

            if not vocabulary or not labels:
                return False

            if len(w1) != hidden_size:
                return False

            if len(b1) != hidden_size:
                return False

            if len(w2) != len(labels):
                return False

            if len(b2) != len(labels):
                return False

            if any(len(row) != len(vocabulary) for row in w1):
                return False

            if any(len(row) != hidden_size for row in w2):
                return False

            self.vocabulary = vocabulary
            self.labels = labels
            self.hidden_size = hidden_size
            self.w1 = w1
            self.b1 = b1
            self.w2 = w2
            self.b2 = b2

            return True

        except (KeyError, TypeError, ValueError):
            return False


# ==========================================
# KNOWLEDGE ENGINE
# ==========================================

class KnowledgeEngine:

    def __init__(self):
        self.data = load_json("knowledge", {})

        if not isinstance(self.data, dict):
            self.data = {}

    def teach(self, question, answer):
        key = normalize(question)

        if not key or not answer.strip():
            return False

        self.data[key] = answer.strip()
        save_json("knowledge", self.data)

        return True

    def answer(self, question):
        return self.data.get(normalize(question))


# ==========================================
# MEMORY ENGINE
# ==========================================

class MemoryEngine:

    def __init__(self):
        self.data = load_json("memory", {})

        if not isinstance(self.data, dict):
            self.data = {}

    def remember(self, key, value):
        self.data[key.strip().lower()] = value.strip()
        save_json("memory", self.data)

    def forget(self, key):
        key = key.strip().lower()

        if key in self.data:
            del self.data[key]
            save_json("memory", self.data)
            return True

        return False

    def show(self):
        return self.data


# ==========================================
# UNKNOWN QUESTION TRACKER
# ==========================================

class UnknownTracker:

    def __init__(self):
        self.data = load_json("unknown", {})

        if not isinstance(self.data, dict):
            self.data = {}

    def add(self, question):
        self.data[question] = self.data.get(question, 0) + 1
        save_json("unknown", self.data)


# ==========================================
# FEEDBACK ENGINE
# ==========================================

class FeedbackEngine:

    def __init__(self):
        self.data = load_json("feedback", [])

        if not isinstance(self.data, list):
            self.data = []

    def add(self, feedback, correction=""):
        self.data.append({
            "feedback": feedback,
            "correction": correction,
            "time": datetime.now().isoformat(timespec="seconds")
        })

        save_json("feedback", self.data)


# ==========================================
# CONTEXT ENGINE
# ==========================================

class ContextEngine:

    def __init__(self):
        self.history = load_json("history", [])

        if not isinstance(self.history, list):
            self.history = []

    def add(self, role, message):
        self.history.append({
            "role": role,
            "message": message,
            "time": datetime.now().isoformat(timespec="seconds")
        })

        self.history = self.history[-200:]
        save_json("history", self.history)

    def clear(self):
        self.history = []
        save_json("history", self.history)


# ==========================================
# ULTRON MAIN SYSTEM
# ==========================================

class Ultron:

    def __init__(self):
        print("\n[INITIALIZING ULTRON V9]")

        self.brain = NeuralBrain()
        print("[OK] Neural Brain")

        self.knowledge = KnowledgeEngine()
        print("[OK] Knowledge Engine")

        self.memory = MemoryEngine()
        print("[OK] Persistent Memory")

        self.unknown = UnknownTracker()
        print("[OK] Unknown Tracker")

        self.feedback = FeedbackEngine()
        print("[OK] Feedback Engine")

        self.context = ContextEngine()
        print("[OK] Context Engine")

        self.responses = RESPONSES

        print("\nULTRON V9 ONLINE.\n")

    def respond(self, question):
        question = question.strip()

        if not question:
            return ""

        self.context.add("user", question)

        # Exact saved knowledge
        answer = self.knowledge.answer(question)

        if answer:
            self.context.add("ultron", answer)
            return answer

        # Memory lookup
        if normalize(question) in (
            "what do you remember about me",
            "show my memory"
        ):
            answer = str(self.memory.show())
            self.context.add("ultron", answer)
            return answer

        # Neural prediction
        intent, confidence = self.brain.predict(question)

        if intent and intent in self.responses:
            answer = random.choice(self.responses[intent])
            answer += f" [Confidence: {confidence:.2f}]"

            self.context.add("ultron", answer)
            return answer

        self.unknown.add(question)

        answer = (
            "I don't understand this question yet. "
            "You can teach me using /teach."
        )

        self.context.add("ultron", answer)

        return answer

    def stats(self):
        print("\n========== ULTRON V9 STATISTICS ==========")
        print("Version: 9.0")
        print("Learned Questions:", len(self.knowledge.data))
        print("Training Examples:", len(self.brain.examples))
        print("Intent Categories:", len(self.brain.labels))
        print("Unknown Questions:", len(self.unknown.data))
        print("Feedback Records:", len(self.feedback.data))
        print("Memory Records:", len(self.memory.data))
        print("Conversation Messages:", len(self.context.history))
        print("Vocabulary Size:", len(self.brain.vocabulary))
        print("Neural Hidden Neurons:", self.brain.hidden_size)
        print("==========================================")

    def show_intents(self):
        print("\n========== NEURAL INTENTS ==========")

        for intent in self.brain.labels:
            count = sum(
                1 for item in self.brain.examples
                if item.get("intent") == intent
            )

            print(f"{intent}: {count} examples")

    def help(self):
        print("""
========== ULTRON V9 COMMANDS ==========

/teach question | answer
/teach_intent sentence | intent
/train
/intents
/unknown
/feedback good
/feedback bad | correction
/remember key | value
/memory
/forget key
/history
/clear
/stats
/help
/exit

=======================================
""")

    def command(self, text):
        if text == "/help":
            self.help()
            return

        if text.startswith("/teach_intent "):
            content = text[len("/teach_intent "):]

            if "|" not in content:
                print("Usage: /teach_intent sentence | intent")
                return

            sentence, intent = content.split("|", 1)

            if self.brain.teach(sentence, intent):
                print("[OK] Training example saved. Run /train to learn it.")
            else:
                print("[ERROR] Invalid example.")

            return

        if text.startswith("/teach "):
            content = text[len("/teach "):]

            if "|" not in content:
                print("Usage: /teach question | answer")
                return

            question, answer = content.split("|", 1)

            if self.knowledge.teach(question, answer):
                print("[OK] Knowledge saved.")
            else:
                print("[ERROR] Invalid question or answer.")

            return

        if text == "/train":
            self.brain.train()
            return

        if text == "/intents":
            self.show_intents()
            return

        if text == "/unknown":
            print("\n========== UNKNOWN QUESTIONS ==========")

            if not self.unknown.data:
                print("No unknown questions.")
            else:
                for question, count in self.unknown.data.items():
                    print(f"{question}: {count}")

            return

        if text.startswith("/feedback"):
            content = text[len("/feedback"):].strip()

            if content.startswith("bad"):
                correction = ""

                if "|" in content:
                    _, correction = content.split("|", 1)

                self.feedback.add("bad", correction.strip())
            elif content == "good":
                self.feedback.add("good")
            else:
                print("Usage: /feedback good OR /feedback bad | correction")
                return

            print("[OK] Feedback saved.")
            return

        if text.startswith("/remember "):
            content = text[len("/remember "):]

            if "|" not in content:
                print("Usage: /remember key | value")
                return

            key, value = content.split("|", 1)

            if not key.strip() or not value.strip():
                print("Both key and value are required.")
                return

            self.memory.remember(key, value)
            print("[OK] Memory saved.")
            return

        if text == "/memory":
            print("\n========== MEMORY ==========")
            print(self.memory.show())
            return

        if text.startswith("/forget "):
            key = text[len("/forget "):]

            if self.memory.forget(key):
                print("[OK] Memory forgotten.")
            else:
                print("[INFO] Memory key not found.")

            return

        if text == "/history":
            print("\n========== CONVERSATION HISTORY ==========")

            for item in self.context.history:
                print(f"{item['role']}: {item['message']}")

            return

        if text == "/clear":
            self.context.clear()
            print("[OK] Conversation history cleared.")
            return

        if text == "/stats":
            self.stats()
            return

        if text == "/exit":
            print("ULTRON V9 shutting down.")
            raise SystemExit

        if text.startswith("/"):
            print("[ERROR] Unknown command. Type /help.")
            return

        answer = self.respond(text)
        print("\nULTRON >", answer)


# ==========================================
# STARTUP
# ==========================================

def main():
    print("""
========================================
             U L T R O N  V9
          NEURAL LEARNING ENGINE
========================================
""")

    try:
        ultron = Ultron()
        ultron.help()

        while True:
            try:
                user_input = input("\nYOU > ").strip()

                if user_input:
                    ultron.command(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\nULTRON V9 shutting down.")
                break

    except SystemExit:
        pass

    except Exception as error:
        print("\n[STARTUP ERROR]", error)
        print("Please share the complete error output.")


if __name__ == "__main__":
    main()
