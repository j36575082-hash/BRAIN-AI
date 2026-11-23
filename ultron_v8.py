
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

# ============================================================
# ULTRON V8 - COMPLETE LEARNING SYSTEM
# ============================================================

VERSION = "8.0"
BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "responses": "ultron_response_data.json",
    "neural": "ultron_neural_core.json",
    "language": "ultron_language_model.json",
    "context": "ultron_context.json",
    "memory": "ultron_memory.json",
    "learned": "ultron_learned_data.json",
    "unknown": "ultron_unknown_questions.json",
    "feedback": "ultron_feedback.json",
    "training": "ultron_intent_training.json",
}


# ============================================================
# CORE UTILITIES
# ============================================================

def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def load_json(filename, default):
    path = BASE_DIR / filename

    if not path.exists():
        save_json(filename, default)
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    except (json.JSONDecodeError, OSError) as error:
        print(f"[WARNING] {filename}: {error}")
        return default


def save_json(filename, data):
    path = BASE_DIR / filename
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        os.replace(temp_path, path)

    except OSError as error:
        print(f"[SAVE ERROR] {filename}: {error}")


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def words(text):
    return normalize(text).split()


# ============================================================
# 1. KNOWLEDGE ENGINE
# ============================================================

class KnowledgeEngine:

    def __init__(self):
        self.data = load_json(
            FILES["responses"],
            {"knowledge": {}}
        )

        self.learned = load_json(
            FILES["learned"],
            {"questions": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"knowledge": {}}

        if not isinstance(self.learned, dict):
            self.learned = {"questions": []}

        if not isinstance(self.data.get("knowledge"), (dict, list)):
            self.data["knowledge"] = {}

        if not isinstance(self.learned.get("questions"), list):
            self.learned["questions"] = []

    def find_answer(self, question):
        target = normalize(question)

        for item in reversed(self.learned["questions"]):

            if not isinstance(item, dict):
                continue

            if normalize(item.get("question", "")) == target:
                return item.get("answer")

        knowledge = self.data["knowledge"]

        if isinstance(knowledge, dict):

            for key, value in knowledge.items():

                if normalize(key) == target:

                    if isinstance(value, dict):
                        return value.get("answer")

                    return value

        elif isinstance(knowledge, list):

            for item in knowledge:

                if not isinstance(item, dict):
                    continue

                if normalize(item.get("question", "")) == target:
                    return item.get("answer")

        return None

    def teach(self, question, answer):
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return False

        target = normalize(question)

        for item in self.learned["questions"]:

            if (
                isinstance(item, dict)
                and normalize(item.get("question", "")) == target
            ):

                item["answer"] = answer
                item["updated_at"] = timestamp()

                save_json(FILES["learned"], self.learned)
                return True

        self.learned["questions"].append({
            "question": question,
            "answer": answer,
            "created_at": timestamp()
        })

        save_json(FILES["learned"], self.learned)
        return True

    def count(self):
        return len(self.learned["questions"])


# ============================================================
# 2. UNKNOWN QUESTION TRACKER
# ============================================================

class UnknownTracker:

    def __init__(self):
        self.data = load_json(
            FILES["unknown"],
            {"questions": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"questions": []}

        if not isinstance(self.data.get("questions"), list):
            self.data["questions"] = []

    def add(self, question):
        target = normalize(question)

        for item in self.data["questions"]:

            if (
                isinstance(item, dict)
                and normalize(item.get("question", "")) == target
            ):

                item["count"] = item.get("count", 1) + 1
                item["last_seen"] = timestamp()

                save_json(FILES["unknown"], self.data)
                return

        self.data["questions"].append({
            "question": question,
            "count": 1,
            "first_seen": timestamp(),
            "last_seen": timestamp()
        })

        save_json(FILES["unknown"], self.data)

    def remove(self, question):
        target = normalize(question)

        self.data["questions"] = [
            item for item in self.data["questions"]
            if not (
                isinstance(item, dict)
                and normalize(item.get("question", "")) == target
            )
        ]

        save_json(FILES["unknown"], self.data)

    def show(self):
        items = self.data["questions"]

        if not items:
            print("No unknown questions.")
            return

        print("\n--- UNKNOWN QUESTIONS ---")

        for index, item in enumerate(items, 1):

            print(
                f"{index}. {item.get('question', '')} "
                f"[Asked {item.get('count', 1)} times]"
            )


# ============================================================
# 3. FEEDBACK ENGINE
# ============================================================

class FeedbackEngine:

    def __init__(self):
        self.data = load_json(
            FILES["feedback"],
            {"records": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"records": []}

        if not isinstance(self.data.get("records"), list):
            self.data["records"] = []

    def add(self, question, answer, rating, correction=None):

        self.data["records"].append({
            "question": question,
            "answer": answer,
            "rating": rating,
            "correction": correction,
            "created_at": timestamp()
        })

        save_json(FILES["feedback"], self.data)

    def count(self):
        return len(self.data["records"])


# ============================================================
# 4. LANGUAGE GENERATOR
# ============================================================

class LanguageGenerator:

    def __init__(self):
        self.data = load_json(
            FILES["language"],
            {"phrases": {}, "patterns": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"phrases": {}, "patterns": []}

        if not isinstance(self.data.get("phrases"), dict):
            self.data["phrases"] = {}

        if not isinstance(self.data.get("patterns"), list):
            self.data["patterns"] = []

        self.default_phrases = {
            "greeting": [
                "Hello! I am ULTRON.",
                "Hello! How can I help you?",
                "System online. Ready to learn."
            ],

            "thanks": [
                "You're welcome.",
                "Happy to help.",
                "Always ready to assist."
            ],

            "identity": [
                "I am ULTRON, an independently developed AI learning system."
            ],

            "unknown": [
                "I don't know the answer yet. You can teach me using /teach."
            ]
        }

    def generate(self, category):
        custom = self.data["phrases"].get(category)

        if isinstance(custom, list) and custom:
            return random.choice(custom)

        if isinstance(custom, str):
            return custom

        return random.choice(
            self.default_phrases.get(
                category,
                self.default_phrases["unknown"]
            )
        )

    def add_phrase(self, category, phrase):
        phrases = self.data["phrases"].setdefault(category, [])

        if isinstance(phrases, str):
            phrases = [phrases]

        phrases.append(phrase)
        self.data["phrases"][category] = phrases

        save_json(FILES["language"], self.data)


# ============================================================
# 5. NEURAL BRAIN
# ============================================================

class NeuralBrain:

    def __init__(self):
        self.data = load_json(
            FILES["neural"],
            {"intents": {}}
        )

        self.training = load_json(
            FILES["training"],
            {"examples": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"intents": {}}

        if not isinstance(self.training, dict):
            self.training = {"examples": []}

        if not isinstance(self.training.get("examples"), list):
            self.training["examples"] = []

        raw_intents = self.data.get("intents", {})

        if isinstance(raw_intents, list):

            converted = {}

            for index, item in enumerate(raw_intents):

                if not isinstance(item, dict):
                    continue

                name = item.get(
                    "intent",
                    item.get("name", f"custom_{index}")
                )

                examples = item.get("examples", [])

                converted[name] = {
                    "examples": examples if isinstance(examples, list) else []
                }

            self.data["intents"] = converted

        elif not isinstance(raw_intents, dict):
            self.data["intents"] = {}

        self.seed_intents()

    def seed_intents(self):

        defaults = {
            "greeting": [
                "hello", "hi", "hey",
                "good morning", "good evening", "namaste"
            ],

            "thanks": [
                "thank you", "thanks", "thank you ultron"
            ],

            "identity": [
                "who are you",
                "what are you",
                "tell me about yourself"
            ],

            "goodbye": [
                "bye", "goodbye", "see you"
            ]
        }

        for intent, examples in defaults.items():

            self.data["intents"].setdefault(
                intent,
                {"examples": examples}
            )

        save_json(FILES["neural"], self.data)

    def teach_intent(self, sentence, intent):

        sentence = sentence.strip()
        intent = normalize(intent)

        if not sentence or not intent:
            return False

        info = self.data["intents"].setdefault(
            intent,
            {"examples": []}
        )

        if isinstance(info, list):
            info = {"examples": info}
            self.data["intents"][intent] = info

        if not isinstance(info, dict):
            return False

        example_list = info.setdefault("examples", [])

        if not isinstance(example_list, list):
            return False

        if normalize(sentence) not in [
            normalize(x) for x in example_list
        ]:
            example_list.append(sentence)

        self.training["examples"].append({
            "sentence": sentence,
            "intent": intent,
            "created_at": timestamp()
        })

        save_json(FILES["neural"], self.data)
        save_json(FILES["training"], self.training)

        return True

    def predict(self, sentence):

        input_words = set(words(sentence))

        if not input_words:
            return None, 0.0

        best_intent = None
        best_score = 0.0

        for intent, info in self.data["intents"].items():

            if isinstance(info, list):
                examples = info

            elif isinstance(info, dict):
                examples = info.get("examples", [])

            else:
                continue

            if not isinstance(examples, list):
                continue

            for example in examples:

                example_words = set(words(example))

                if not example_words:
                    continue

                common = len(input_words & example_words)

                if common == 0:
                    continue

                precision = common / len(input_words)
                recall = common / len(example_words)

                score = (
                    2 * precision * recall /
                    (precision + recall)
                )

                if score > best_score:
                    best_score = score
                    best_intent = intent

        return best_intent, round(best_score, 3)

    def train(self):
        save_json(FILES["neural"], self.data)
        save_json(FILES["training"], self.training)

        return len(self.training["examples"])

    def list_intents(self):

        result = {}

        for intent, info in self.data["intents"].items():

            if isinstance(info, list):
                result[intent] = len(info)

            elif isinstance(info, dict):
                examples = info.get("examples", [])
                result[intent] = len(examples) if isinstance(examples, list) else 0

        return result


# ============================================================
# 6. CONTEXT ENGINE
# ============================================================

class ContextEngine:

    def __init__(self):
        self.data = load_json(
            FILES["context"],
            {"history": []}
        )

        if not isinstance(self.data, dict):
            self.data = {"history": []}

        if not isinstance(self.data.get("history"), list):
            self.data["history"] = []

    def add(self, role, message):

        self.data["history"].append({
            "role": role,
            "message": message,
            "time": timestamp()
        })

        self.data["history"] = self.data["history"][-200:]

        save_json(FILES["context"], self.data)

    def show(self):

        history = self.data["history"]

        if not history:
            print("No conversation history.")
            return

        print("\n--- CONVERSATION HISTORY ---")

        for item in history[-20:]:

            if isinstance(item, dict):
                print(
                    f"[{item.get('role', 'unknown')}] "
                    f"{item.get('message', '')}"
                )

    def clear(self):
        self.data["history"] = []
        save_json(FILES["context"], self.data)


# ============================================================
# 7. PERSISTENT MEMORY
# ============================================================

class MemoryEngine:

    def __init__(self):
        self.data = load_json(
            FILES["memory"],
            {"facts": {}}
        )

        if not isinstance(self.data, dict):
            self.data = {"facts": {}}

        if not isinstance(self.data.get("facts"), dict):
            self.data["facts"] = {}

    def remember(self, key, value):

        key = normalize(key)

        if not key or not value:
            return False

        self.data["facts"][key] = {
            "value": value,
            "updated_at": timestamp()
        }

        save_json(FILES["memory"], self.data)

        return True

    def recall(self, key):

        key = normalize(key)
        item = self.data["facts"].get(key)

        if isinstance(item, dict):
            return item.get("value")

        return item

    def forget(self, key):

        key = normalize(key)

        if key in self.data["facts"]:
            del self.data["facts"][key]

            save_json(FILES["memory"], self.data)
            return True

        return False

    def show(self):

        facts = self.data["facts"]

        if not facts:
            print("Memory is empty.")
            return

        print("\n--- ULTRON MEMORY ---")

        for key, item in facts.items():

            value = (
                item.get("value")
                if isinstance(item, dict)
                else item
            )

            print(f"{key}: {value}")


# ============================================================
# 8. MAIN ULTRON SYSTEM
# ============================================================

class Ultron:

    def __init__(self):

        print("\n[INITIALIZING ULTRON V8]")

        self.knowledge = KnowledgeEngine()
        self.unknown = UnknownTracker()
        self.feedback = FeedbackEngine()
        self.language = LanguageGenerator()
        self.brain = NeuralBrain()
        self.context = ContextEngine()
        self.memory = MemoryEngine()

        self.running = True

        print("[OK] Knowledge Engine")
        print("[OK] Unknown Tracker")
        print("[OK] Feedback Engine")
        print("[OK] Language Generator")
        print("[OK] Neural Brain")
        print("[OK] Context Engine")
        print("[OK] Persistent Memory")

        print("\nULTRON V8 ONLINE.")

    # --------------------------------------------------------
    # RESPONSE PROCESSING
    # --------------------------------------------------------

    def respond(self, question):

        question = question.strip()

        if not question:
            return "Please enter something."

        self.context.add("user", question)

        answer = self.knowledge.find_answer(question)

        if answer:
            self.context.add("ultron", str(answer))
            return str(answer)

        normalized = normalize(question)

        memory_patterns = {
            "what is my name": "name",
            "whats my name": "name",
            "do you remember my name": "name"
        }

        if normalized in memory_patterns:

            value = self.memory.recall(
                memory_patterns[normalized]
            )

            if value:
                answer = f"Your name is {value}."
                self.context.add("ultron", answer)
                return answer

        if normalized == "what do you remember about me":
            self.memory.show()
            return "Here is what I have stored in my memory."

        intent, confidence = self.brain.predict(question)

        answer = None

        if intent == "greeting" and confidence >= 0.30:
            answer = self.language.generate("greeting")

        elif intent == "thanks" and confidence >= 0.30:
            answer = self.language.generate("thanks")

        elif intent == "identity" and confidence >= 0.30:
            answer = self.language.generate("identity")

        elif intent == "goodbye" and confidence >= 0.30:
            answer = "Goodbye! ULTRON going offline."

        if answer is None:

            self.unknown.add(question)
            answer = self.language.generate("unknown")

        self.context.add("ultron", answer)

        return answer

    # --------------------------------------------------------
    # TEACH KNOWLEDGE
    # --------------------------------------------------------

    def command_teach(self, argument):

        if "|" not in argument:
            print("Usage: /teach question | answer")
            return

        question, answer = argument.split("|", 1)

        if self.knowledge.teach(question, answer):

            self.unknown.remove(question)

            print("[SUCCESS] ULTRON learned a new answer.")

        else:
            print("[ERROR] Invalid question or answer.")

    # --------------------------------------------------------
    # TEACH INTENT
    # --------------------------------------------------------

    def command_teach_intent(self, argument):

        if "|" not in argument:
            print("Usage: /teach_intent sentence | intent")
            return

        sentence, intent = argument.split("|", 1)

        if self.brain.teach_intent(sentence, intent):
            print("[SUCCESS] New intent example learned.")

        else:
            print("[ERROR] Could not save intent.")

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    def command_feedback(self, argument):

        parts = argument.split("|", 1)

        rating = parts[0].strip().lower()

        correction = (
            parts[1].strip()
            if len(parts) > 1
            else None
        )

        if rating not in ("good", "bad"):

            print("Usage: /feedback good")
            print("Usage: /feedback bad | correct answer")
            return

        history = self.context.data["history"]

        user_question = ""
        ultron_answer = ""

        for item in reversed(history):

            if not isinstance(item, dict):
                continue

            if item.get("role") == "ultron" and not ultron_answer:
                ultron_answer = item.get("message", "")

            elif item.get("role") == "user" and not user_question:
                user_question = item.get("message", "")

            if user_question and ultron_answer:
                break

        self.feedback.add(
            user_question,
            ultron_answer,
            rating,
            correction
        )

        if rating == "bad" and correction and user_question:

            self.knowledge.teach(
                user_question,
                correction
            )

            self.unknown.remove(user_question)

            print("[SUCCESS] Correction saved.")

        else:
            print("[SUCCESS] Feedback saved.")

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def stats(self):

        print("\n========== ULTRON STATISTICS ==========")

        print("Version:", VERSION)
        print("Learned Questions:", self.knowledge.count())
        print("Unknown Questions:", len(self.unknown.data["questions"]))
        print("Feedback Records:", self.feedback.count())
        print("Intent Categories:", len(self.brain.data["intents"]))
        print("Training Examples:", len(self.brain.training["examples"]))
        print("Memory Records:", len(self.memory.data["facts"]))
        print("Conversation Messages:", len(self.context.data["history"]))

        print("=======================================")

    # --------------------------------------------------------
    # INTENTS
    # --------------------------------------------------------

    def show_intents(self):

        print("\n========== AVAILABLE INTENTS ==========")

        for intent, count in self.brain.list_intents().items():
            print(f"{intent}: {count} examples")

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    def show_help(self):

        print("""
========== ULTRON V8 COMMANDS ==========

/help
/teach question | answer
/teach_intent sentence | intent
/intents
/train
/unknown
/feedback good
/feedback bad | correction
/memory
/forget memory_key
/history
/clear
/stats
/exit

========================================
""")

    # --------------------------------------------------------
    # COMMAND PROCESSOR
    # --------------------------------------------------------

    def process_command(self, command):

        command = command.strip()

        if command == "/exit":

            self.running = False
            print("ULTRON shutting down.")

        elif command == "/history":
            self.context.show()

        elif command == "/clear":

            self.context.clear()
            print("Conversation history cleared.")

        elif command == "/teach":
            print("Usage: /teach question | answer")

        elif command.startswith("/teach "):
            self.command_teach(command[7:])

        elif command == "/teach_intent":
            print("Usage: /teach_intent sentence | intent")

        elif command.startswith("/teach_intent "):
            self.command_teach_intent(command[14:])

        elif command == "/intents":
            self.show_intents()

        elif command == "/train":

            total = self.brain.train()

            print(
                f"[TRAINING COMPLETE] {total} examples saved."
            )

        elif command == "/unknown":
            self.unknown.show()

        elif command == "/stats":
            self.stats()

        elif command == "/feedback":
            print("Usage: /feedback good")
            print("Usage: /feedback bad | correction")

        elif command.startswith("/feedback "):
            self.command_feedback(command[10:])

        elif command == "/memory":
            self.memory.show()

        elif command == "/forget":
            print("Usage: /forget memory_key")

        elif command.startswith("/forget "):

            key = command[8:]

            if self.memory.forget(key):
                print("[SUCCESS] Memory deleted.")

            else:
                print("[INFO] Memory not found.")

        elif command == "/help":
            self.show_help()

        else:
            print("Unknown command. Type /help.")


# ============================================================
# 9. STARTUP SYSTEM
# ============================================================

def main():

    print("""
========================================
          U L T R O N  V8
       INDEPENDENT LEARNING SYSTEM
========================================
""")

    try:

        ultron = Ultron()

        ultron.show_help()

        while ultron.running:

            try:

                user_input = input("\nYOU > ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):

                    ultron.process_command(user_input)

                else:

                    response = ultron.respond(user_input)

                    print("\nULTRON >", response)

            except KeyboardInterrupt:

                print("\nUse /exit to shut down ULTRON.")

            except Exception as error:

                print("[PROCESS ERROR]", error)

    except Exception as error:

        print("\n[STARTUP ERROR]", error)

    finally:

        print("\nULTRON SESSION ENDED.")


if __name__ == "__main__":
    main()
