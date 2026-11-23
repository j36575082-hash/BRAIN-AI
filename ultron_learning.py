
import json
import os
import re
from difflib import SequenceMatcher

DATA_FILE = "ultron_learned_data.json"


# ==========================================
# FILE MANAGEMENT
# ==========================================

def load_data():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ==========================================
# TEXT PROCESSING
# ==========================================

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similarity(text1, text2):
    return SequenceMatcher(
        None,
        normalize(text1),
        normalize(text2)
    ).ratio()


# ==========================================
# ULTRON LEARNING ENGINE
# ==========================================

class UltronLearning:

    def __init__(self):
        self.data = load_data()

        print("\nULTRON Learning System Connected!")
        print("Saved knowledge:", len(self.data))

    # --------------------------------------
    # FIND ANSWER
    # --------------------------------------

    def find_answer(self, question):

        best_answer = None
        best_score = 0

        for item in self.data:

            score = similarity(
                question,
                item["question"]
            )

            if score > best_score:
                best_score = score
                best_answer = item["answer"]

        if best_score >= 0.82:
            return best_answer, best_score

        return None, best_score

    # --------------------------------------
    # TEACH ULTRON
    # --------------------------------------

    def teach(self, question, answer):

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            print("Question and answer cannot be empty.")
            return

        existing = None

        for item in self.data:
            if normalize(item["question"]) == normalize(question):
                existing = item
                break

        if existing:
            existing["answer"] = answer
            print("Existing knowledge updated!")

        else:
            self.data.append({
                "question": question,
                "answer": answer
            })

            print("New knowledge learned!")

        save_data(self.data)

    # --------------------------------------
    # ASK ULTRON
    # --------------------------------------

    def ask(self, question):

        answer, score = self.find_answer(question)

        if answer:
            print("\nULTRON:", answer)
            print("Match score:", round(score * 100, 2), "%")

        else:
            print("\nULTRON: I don't know the answer yet.")
            print("You can teach me using option 2.")

    # --------------------------------------
    # SHOW KNOWLEDGE
    # --------------------------------------

    def show_knowledge(self):

        if not self.data:
            print("No knowledge learned yet.")
            return

        print("\n========== LEARNED KNOWLEDGE ==========")

        for index, item in enumerate(self.data, start=1):
            print(f"\n{index}. Question: {item['question']}")
            print(f"   Answer: {item['answer']}")

    # --------------------------------------
    # DELETE KNOWLEDGE
    # --------------------------------------

    def delete_knowledge(self):

        self.show_knowledge()

        if not self.data:
            return

        try:
            number = int(input("\nEnter knowledge number to delete: "))

            if 1 <= number <= len(self.data):
                removed = self.data.pop(number - 1)
                save_data(self.data)

                print("Deleted:", removed["question"])

            else:
                print("Invalid number.")

        except ValueError:
            print("Please enter a valid number.")

    # --------------------------------------
    # MAIN MENU
    # --------------------------------------

    def run(self):

        while True:

            print("\n================================")
            print("      ULTRON LEARNING ENGINE")
            print("================================")
            print("1. Ask ULTRON")
            print("2. Teach ULTRON")
            print("3. Show learned knowledge")
            print("4. Delete knowledge")
            print("5. Exit")

            choice = input("\nChoose: ").strip()

            if choice == "1":

                question = input("\nYou: ")
                self.ask(question)

            elif choice == "2":

                question = input("\nEnter question: ")
                answer = input("Enter correct answer: ")

                self.teach(question, answer)

            elif choice == "3":

                self.show_knowledge()

            elif choice == "4":

                self.delete_knowledge()

            elif choice == "5":

                print("\nULTRON: Learning system shutting down.")
                break

            else:
                print("Invalid choice.")


if __name__ == "__main__":
    system = UltronLearning()
    system.run()
