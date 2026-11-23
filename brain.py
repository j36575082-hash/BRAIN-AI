
import random
import json
import math
import os

# ==========================================
#          ULTRON AI - LEVEL 3
#          MEMORY + LEARNING SYSTEM
# ==========================================

BRAIN_FILE = "ultron_brain.json"
MEMORY_FILE = "ultron_memory.json"

INPUTS = 2
HIDDEN = 4
OUTPUTS = 1

LEARNING_RATE = 0.5

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]


# ==========================================
# INITIALIZE BRAIN
# ==========================================

def create_brain():

    return {
        "weights_input_hidden": [
            [random.uniform(-1, 1) for _ in range(HIDDEN)]
            for _ in range(INPUTS)
        ],

        "weights_hidden_output": [
            [random.uniform(-1, 1) for _ in range(OUTPUTS)]
            for _ in range(HIDDEN)
        ],

        "bias_hidden": [
            random.uniform(-1, 1) for _ in range(HIDDEN)
        ],

        "bias_output": [
            random.uniform(-1, 1) for _ in range(OUTPUTS)
        ]
    }


def save_brain():

    with open(BRAIN_FILE, "w") as file:
        json.dump(brain, file, indent=4)


def load_brain():

    if os.path.exists(BRAIN_FILE):

        try:
            with open(BRAIN_FILE, "r") as file:
                data = json.load(file)

            if (
                len(data["weights_input_hidden"]) == INPUTS
                and all(len(row) == HIDDEN for row in data["weights_input_hidden"])
                and len(data["weights_hidden_output"]) == HIDDEN
                and all(len(row) == OUTPUTS for row in data["weights_hidden_output"])
                and len(data["bias_hidden"]) == HIDDEN
                and len(data["bias_output"]) == OUTPUTS
            ):
                print("Existing trained brain loaded!")
                return data

        except (ValueError, KeyError, TypeError, OSError):
            print("Old brain could not be loaded.")

    print("Creating a new brain...")
    return create_brain()


brain = load_brain()

weights_input_hidden = brain["weights_input_hidden"]
weights_hidden_output = brain["weights_hidden_output"]
bias_hidden = brain["bias_hidden"]
bias_output = brain["bias_output"]


# ==========================================
# MEMORY SYSTEM
# ==========================================

def load_memory():

    if os.path.exists(MEMORY_FILE):

        try:
            with open(MEMORY_FILE, "r") as file:
                data = json.load(file)

            if isinstance(data, dict):
                return data

        except (ValueError, OSError):
            print("Memory file could not be loaded.")

    return {}


memory = load_memory()


def save_memory():

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)


def remember():

    key = input("\nWhat should I remember? ").strip()

    if not key:
        print("Memory name cannot be empty.")
        return

    value = input("Tell me the information: ").strip()

    if not value:
        print("Information cannot be empty.")
        return

    memory[key.lower()] = {
        "name": key,
        "information": value
    }

    save_memory()

    print("\nULTRON: Information saved successfully!")


def recall():

    key = input("\nWhat should I recall? ").strip().lower()

    if key in memory:

        print("\nULTRON REMEMBERS:")
        print(memory[key]["information"])

    else:

        print("\nULTRON: I don't have this information yet.")


def show_memory():

    if not memory:

        print("\nULTRON memory is empty.")
        return

    print("\n========== ULTRON MEMORY ==========")

    for item in memory.values():

        print(f"\n{item['name']}: {item['information']}")

    print("\n===================================")


# ==========================================
# NEURAL NETWORK
# ==========================================

def sigmoid(x):

    x = max(-60, min(60, x))

    return 1 / (1 + math.exp(-x))


def forward(inputs):

    hidden_values = []

    for j in range(HIDDEN):

        total = bias_hidden[j]

        for i in range(INPUTS):

            total += inputs[i] * weights_input_hidden[i][j]

        hidden_values.append(sigmoid(total))

    output_values = []

    for k in range(OUTPUTS):

        total = bias_output[k]

        for j in range(HIDDEN):

            total += hidden_values[j] * weights_hidden_output[j][k]

        output_values.append(sigmoid(total))

    return hidden_values, output_values


# ==========================================
# TRAINING SYSTEM
# ==========================================

def train_brain():

    print("\nULTRON is learning...")

    epochs = 5000

    for epoch in range(epochs):

        total_loss = 0

        random.shuffle(training_data)

        for inputs, targets in training_data:

            hidden, outputs = forward(inputs)

            output_errors = [
                targets[k] - outputs[k]
                for k in range(OUTPUTS)
            ]

            output_deltas = [
                output_errors[k] * outputs[k] * (1 - outputs[k])
                for k in range(OUTPUTS)
            ]

            hidden_deltas = []

            for j in range(HIDDEN):

                error = 0

                for k in range(OUTPUTS):

                    error += (
                        output_deltas[k] *
                        weights_hidden_output[j][k]
                    )

                hidden_deltas.append(
                    error * hidden[j] * (1 - hidden[j])
                )

            # Update output weights

            for j in range(HIDDEN):

                for k in range(OUTPUTS):

                    weights_hidden_output[j][k] += (
                        LEARNING_RATE *
                        output_deltas[k] *
                        hidden[j]
                    )

            for k in range(OUTPUTS):

                bias_output[k] += (
                    LEARNING_RATE * output_deltas[k]
                )

            # Update input weights

            for i in range(INPUTS):

                for j in range(HIDDEN):

                    weights_input_hidden[i][j] += (
                        LEARNING_RATE *
                        hidden_deltas[j] *
                        inputs[i]
                    )

            for j in range(HIDDEN):

                bias_hidden[j] += (
                    LEARNING_RATE * hidden_deltas[j]
                )

            total_loss += sum(
                error ** 2 for error in output_errors
            )

        if epoch % 1000 == 0:

            print(
                f"Epoch: {epoch} | "
                f"Loss: {total_loss / len(training_data):.6f}"
            )

    save_brain()

    print("\nULTRON: Learning completed!")
    print("Trained brain saved successfully.")


# ==========================================
# PREDICTION
# ==========================================

def predict():

    try:

        first = int(input("\nEnter first input (0 or 1): "))
        second = int(input("Enter second input (0 or 1): "))

        if first not in (0, 1) or second not in (0, 1):
            print("Please enter only 0 or 1.")
            return

        _, output = forward([first, second])

        result = output[0]

        print("\nULTRON RESULT:")
        print("Raw prediction:", round(result, 4))
        print("Predicted class:", round(result))

    except ValueError:

        print("Invalid input. Enter 0 or 1.")


# ==========================================
# MAIN SYSTEM
# ==========================================

def main():

    print("\n====================================")
    print("       ULTRON ARTIFICIAL BRAIN")
    print("       LEVEL 3 - MEMORY SYSTEM")
    print("====================================")

    print("\nBrain: Loaded")
    print("Memory entries:", len(memory))

    while True:

        print("\n========== MAIN MENU ==========")

        print("1. Train Neural Network")
        print("2. Test Brain")
        print("3. Remember Information")
        print("4. Recall Information")
        print("5. Show All Memory")
        print("0. Shutdown ULTRON")

        print("===============================")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":

            train_brain()

        elif choice == "2":

            predict()

        elif choice == "3":

            remember()

        elif choice == "4":

            recall()

        elif choice == "5":

            show_memory()

        elif choice == "0":

            save_brain()
            save_memory()

            print("\nULTRON shutting down...")
            print("Brain and memory saved.")
            break

        else:

            print("\nInvalid choice. Try again.")


if __name__ == "__main__":

    main()
