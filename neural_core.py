
import json
import os
import random
import re
import math

DATA_FILE = "ultron_response_data.json"
MODEL_FILE = "ultron_neural_core.json"

HIDDEN_SIZE = 16
LEARNING_RATE = 0.05
EPOCHS = 500


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class NeuralCore:

    def __init__(self):
        self.data = self.load_data()
        self.vocabulary = []
        self.intents = []
        self.weights1 = []
        self.bias1 = []
        self.weights2 = []
        self.bias2 = []

    def load_data(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            print("Training data not found.")
            return {}

    def prepare_data(self):
        words = set()

        for details in self.data.values():
            for pattern in details.get("patterns", []):
                words.update(tokenize(pattern))

        self.vocabulary = sorted(words)
        self.intents = sorted(self.data.keys())

        if not self.vocabulary or not self.intents:
            raise ValueError("Training data is empty.")

        self.training_data = []

        for intent_index, intent in enumerate(self.intents):
            for pattern in self.data[intent].get("patterns", []):
                vector = self.vectorize(pattern)

                target = [0.0] * len(self.intents)
                target[intent_index] = 1.0

                self.training_data.append((vector, target))

    def vectorize(self, sentence):
        words = set(tokenize(sentence))

        return [
            1.0 if word in words else 0.0
            for word in self.vocabulary
        ]

    @staticmethod
    def sigmoid(value):
        value = max(-50, min(50, value))
        return 1 / (1 + math.exp(-value))

    def initialize(self):
        input_size = len(self.vocabulary)
        output_size = len(self.intents)

        self.weights1 = [
            [random.uniform(-0.5, 0.5) for _ in range(input_size)]
            for _ in range(HIDDEN_SIZE)
        ]

        self.bias1 = [0.0] * HIDDEN_SIZE

        self.weights2 = [
            [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]
            for _ in range(output_size)
        ]

        self.bias2 = [0.0] * output_size

    def forward(self, inputs):
        hidden = []

        for neuron in range(HIDDEN_SIZE):
            total = self.bias1[neuron]

            for i in range(len(inputs)):
                total += self.weights1[neuron][i] * inputs[i]

            hidden.append(self.sigmoid(total))

        output = []

        for neuron in range(len(self.intents)):
            total = self.bias2[neuron]

            for i in range(HIDDEN_SIZE):
                total += self.weights2[neuron][i] * hidden[i]

            output.append(self.sigmoid(total))

        return hidden, output

    def train(self):
        self.prepare_data()
        self.initialize()

        if not self.training_data:
            raise ValueError("No training examples found.")

        for epoch in range(EPOCHS):
            total_loss = 0.0

            random.shuffle(self.training_data)

            for inputs, targets in self.training_data:
                hidden, outputs = self.forward(inputs)

                output_errors = [
                    outputs[i] - targets[i]
                    for i in range(len(self.intents))
                ]

                total_loss += sum(
                    error ** 2 for error in output_errors
                ) / len(self.intents)

                hidden_errors = [0.0] * HIDDEN_SIZE

                for h in range(HIDDEN_SIZE):
                    for o in range(len(self.intents)):
                        hidden_errors[h] += (
                            output_errors[o]
                            * self.weights2[o][h]
                        )

                for o in range(len(self.intents)):
                    gradient = (
                        output_errors[o]
                        * outputs[o]
                        * (1 - outputs[o])
                    )

                    for h in range(HIDDEN_SIZE):
                        self.weights2[o][h] -= (
                            LEARNING_RATE * gradient * hidden[h]
                        )

                    self.bias2[o] -= LEARNING_RATE * gradient

                for h in range(HIDDEN_SIZE):
                    gradient = (
                        hidden_errors[h]
                        * hidden[h]
                        * (1 - hidden[h])
                    )

                    for i in range(len(inputs)):
                        self.weights1[h][i] -= (
                            LEARNING_RATE * gradient * inputs[i]
                        )

                    self.bias1[h] -= LEARNING_RATE * gradient

            if epoch % 100 == 0:
                average_loss = total_loss / len(self.training_data)
                print(f"Epoch {epoch} | Loss: {average_loss:.6f}")

        self.save_model()
        print("\nNeural training completed!")

    def save_model(self):
        model = {
            "vocabulary": self.vocabulary,
            "intents": self.intents,
            "weights1": self.weights1,
            "bias1": self.bias1,
            "weights2": self.weights2,
            "bias2": self.bias2
        }

        with open(MODEL_FILE, "w", encoding="utf-8") as file:
            json.dump(model, file)

        print("Model saved:", MODEL_FILE)

    def load_model(self):
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as file:
                model = json.load(file)

            self.vocabulary = model["vocabulary"]
            self.intents = model["intents"]
            self.weights1 = model["weights1"]
            self.bias1 = model["bias1"]
            self.weights2 = model["weights2"]
            self.bias2 = model["bias2"]

            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            return False

    def predict(self, sentence):
        if not self.vocabulary:
            print("Train the model first.")
            return

        inputs = self.vectorize(sentence)
        _, outputs = self.forward(inputs)

        best_index = outputs.index(max(outputs))

        print("\nPredicted intent:", self.intents[best_index])
        print("Confidence:", round(outputs[best_index] * 100, 2), "%")

        return self.intents[best_index]


def main():
    brain = NeuralCore()

    print("\n================================")
    print("       ULTRON NEURAL CORE")
    print("================================")

    print("1. Train neural brain")
    print("2. Load saved brain")
    print("3. Test a message")
    print("4. Exit")

    while True:
        choice = input("\nChoose: ").strip()

        if choice == "1":
            brain.train()

        elif choice == "2":
            if brain.load_model():
                print("Saved neural brain loaded!")
            else:
                print("No compatible saved model found.")

        elif choice == "3":
            if not brain.vocabulary:
                if not brain.load_model():
                    print("Train or load the model first.")
                    continue

            message = input("Enter message: ")
            brain.predict(message)

        elif choice == "4":
            print("ULTRON Neural Core closed.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
