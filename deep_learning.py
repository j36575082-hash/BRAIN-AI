
import json
import os
import random
import math
import re

# ==========================================
#       ULTRON DEEP LEARNING ENGINE
#       LEVEL 6 - NEURAL NLP
# ==========================================

DATA_FILE = "ultron_training_data.json"
MODEL_FILE = "ultron_deep_model.json"

HIDDEN_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 0.05

random.seed(42)

print("\n======================================")
print("       ULTRON DEEP LEARNING")
print("       NEURAL NLP ENGINE V1")
print("======================================")


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "ultron_training_data.json not found!"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid training data format.")

    examples = []

    for label, sentences in data.items():

        if not isinstance(sentences, list):
            continue

        for sentence in sentences:

            if isinstance(sentence, str) and tokenize(sentence):
                examples.append((sentence, label))

    if len(examples) < 2:
        raise ValueError("Not enough training examples.")

    return examples


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"[a-z0-9']+",
        text.lower()
    )


# ==========================================
# BUILD VOCABULARY
# ==========================================

examples = load_data()

vocabulary = sorted({
    word
    for sentence, label in examples
    for word in tokenize(sentence)
})

labels = sorted({
    label for sentence, label in examples
})

word_index = {
    word: index
    for index, word in enumerate(vocabulary)
}

label_index = {
    label: index
    for index, label in enumerate(labels)
}

INPUT_SIZE = len(vocabulary)
OUTPUT_SIZE = len(labels)


# ==========================================
# TEXT VECTOR REPRESENTATION
# ==========================================

def vectorize(sentence):

    vector = [0.0] * INPUT_SIZE

    for word in tokenize(sentence):

        if word in word_index:
            vector[word_index[word]] += 1.0

    # Normalize word frequency
    total = sum(vector)

    if total > 0:
        vector = [value / total for value in vector]

    return vector


# ==========================================
# NEURAL NETWORK INITIALIZATION
# ==========================================

weights_input_hidden = [
    [
        random.uniform(
            -math.sqrt(6 / (INPUT_SIZE + HIDDEN_SIZE)),
            math.sqrt(6 / (INPUT_SIZE + HIDDEN_SIZE))
        )
        for _ in range(HIDDEN_SIZE)
    ]
    for _ in range(INPUT_SIZE)
]

weights_hidden_output = [
    [
        random.uniform(
            -math.sqrt(6 / (HIDDEN_SIZE + OUTPUT_SIZE)),
            math.sqrt(6 / (HIDDEN_SIZE + OUTPUT_SIZE))
        )
        for _ in range(OUTPUT_SIZE)
    ]
    for _ in range(HIDDEN_SIZE)
]

bias_hidden = [0.0] * HIDDEN_SIZE
bias_output = [0.0] * OUTPUT_SIZE


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def tanh(x):
    return math.tanh(x)


def softmax(values):

    maximum = max(values)

    exponentials = [
        math.exp(value - maximum)
        for value in values
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward(inputs):

    hidden = []

    for j in range(HIDDEN_SIZE):

        total = bias_hidden[j]

        for i in range(INPUT_SIZE):

            total += (
                inputs[i] *
                weights_input_hidden[i][j]
            )

        hidden.append(tanh(total))

    output_raw = []

    for k in range(OUTPUT_SIZE):

        total = bias_output[k]

        for j in range(HIDDEN_SIZE):

            total += (
                hidden[j] *
                weights_hidden_output[j][k]
            )

        output_raw.append(total)

    outputs = softmax(output_raw)

    return hidden, outputs


# ==========================================
# PREPARE TRAINING EXAMPLES
# ==========================================

training_examples = []

for sentence, label in examples:

    inputs = vectorize(sentence)
    target = label_index[label]

    training_examples.append((inputs, target))


# ==========================================
# TRAINING
# ==========================================

def train():

    print("\nULTRON is training its neural network...")

    for epoch in range(EPOCHS):

        random.shuffle(training_examples)

        total_loss = 0.0

        for inputs, target in training_examples:

            hidden, outputs = forward(inputs)

            # Cross entropy loss
            probability = max(outputs[target], 1e-12)

            total_loss -= math.log(probability)

            # Output layer gradients
            output_delta = outputs[:]

            output_delta[target] -= 1.0

            # Hidden layer gradients
            hidden_delta = []

            for j in range(HIDDEN_SIZE):

                error = sum(
                    output_delta[k] *
                    weights_hidden_output[j][k]
                    for k in range(OUTPUT_SIZE)
                )

                hidden_delta.append(
                    error * (1 - hidden[j] ** 2)
                )

            # Update output weights
            for j in range(HIDDEN_SIZE):

                for k in range(OUTPUT_SIZE):

                    weights_hidden_output[j][k] -= (
                        LEARNING_RATE *
                        output_delta[k] *
                        hidden[j]
                    )

            for k in range(OUTPUT_SIZE):

                bias_output[k] -= (
                    LEARNING_RATE *
                    output_delta[k]
                )

            # Update input weights
            for i in range(INPUT_SIZE):

                for j in range(HIDDEN_SIZE):

                    weights_input_hidden[i][j] -= (
                        LEARNING_RATE *
                        hidden_delta[j] *
                        inputs[i]
                    )

            for j in range(HIDDEN_SIZE):

                bias_hidden[j] -= (
                    LEARNING_RATE *
                    hidden_delta[j]
                )

        if epoch % 200 == 0:

            average_loss = total_loss / len(training_examples)

            print(
                f"Epoch: {epoch} | "
                f"Loss: {average_loss:.6f}"
            )

    print("\nTraining completed!")


# ==========================================
# SAVE MODEL
# ==========================================

def save_model():

    model_data = {
        "vocabulary": vocabulary,
        "labels": labels,
        "hidden_size": HIDDEN_SIZE,
        "weights_input_hidden": weights_input_hidden,
        "weights_hidden_output": weights_hidden_output,
        "bias_hidden": bias_hidden,
        "bias_output": bias_output
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as file:

        json.dump(model_data, file, indent=4)

    print("Neural model saved:", MODEL_FILE)


# ==========================================
# PREDICTION
# ==========================================

def predict(sentence):

    words = tokenize(sentence)

    known_words = [
        word for word in words
        if word in word_index
    ]

    if not known_words:

        return "unknown", 0.0

    inputs = vectorize(sentence)

    hidden, outputs = forward(inputs)

    best_index = max(
        range(OUTPUT_SIZE),
        key=lambda i: outputs[i]
    )

    confidence = outputs[best_index]

    return labels[best_index], confidence


# ==========================================
# CHAT SYSTEM
# ==========================================

def respond(intent):

    responses = {

        "greeting":
            "Hello! I am ULTRON. How can I help you?",

        "goodbye":
            "Goodbye! See you again.",

        "thanks":
            "You are welcome!",

        "how_are_you":
            "I am operational and ready to learn.",

        "identity":
            "I am ULTRON, an AI learning project.",

        "creator":
            "My creator is Harsh.",

        "help":
            "I can recognize different types of messages.",

        "time":
            "I can recognize time-related questions."
    }

    return responses.get(
        intent,
        "I don't have a response for that yet."
    )


def chat():

    print("\nULTRON: Deep learning engine activated.")
    print("ULTRON: Ready to communicate.")

    print("\nType /exit to close.")

    while True:

        user_text = input("\nYou: ").strip()

        if user_text.lower() in ["/exit", "exit", "quit"]:
            print("\nULTRON: Shutting down.")
            break

        if not user_text:
            continue

        intent, confidence = predict(user_text)

        print("\nDetected intent:", intent)

        print(
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )

        if intent == "unknown" or confidence < 0.35:

            print(
                "\nULTRON: I could not understand that."
            )

        else:

            print("\nULTRON:", respond(intent))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train()

    save_model()

    print("\n======================================")
    print("       ULTRON DEEP LEARNING READY")
    print("======================================")

    print("Training examples:", len(examples))
    print("Vocabulary size:", INPUT_SIZE)
    print("Hidden neurons:", HIDDEN_SIZE)
    print("Output classes:", OUTPUT_SIZE)

    chat()
