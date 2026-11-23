
import json
import os
import random
import math
import re

# ==========================================
# ULTRON - LEVEL 7
# TRAINABLE WORD EMBEDDINGS
# ==========================================

DATA_FILE = "ultron_training_data.json"
MODEL_FILE = "ultron_embedding_model.json"

EMBED_SIZE = 12
HIDDEN_SIZE = 16
EPOCHS = 3000
LEARNING_RATE = 0.04

random.seed(42)

print("\n======================================")
print("       ULTRON SEMANTIC ENGINE")
print("       WORD EMBEDDINGS V1")
print("======================================")


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# ==========================================
# LOAD TRAINING DATA
# ==========================================

def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(DATA_FILE + " not found!")

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid training data.")

    examples = []

    for label, sentences in data.items():

        if not isinstance(sentences, list):
            continue

        for sentence in sentences:

            if isinstance(sentence, str) and tokenize(sentence):
                examples.append((sentence, label))

    if not examples:
        raise ValueError("No training examples found.")

    return examples


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

VOCAB_SIZE = len(vocabulary)
OUTPUT_SIZE = len(labels)


# ==========================================
# INITIALIZE WORD EMBEDDINGS
# ==========================================

def random_matrix(rows, cols, scale):

    return [
        [
            random.uniform(-scale, scale)
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]


embeddings = random_matrix(
    VOCAB_SIZE,
    EMBED_SIZE,
    0.2
)

weights_hidden = random_matrix(
    EMBED_SIZE,
    HIDDEN_SIZE,
    math.sqrt(2 / (EMBED_SIZE + HIDDEN_SIZE))
)

weights_output = random_matrix(
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    math.sqrt(2 / (HIDDEN_SIZE + OUTPUT_SIZE))
)

bias_hidden = [0.0] * HIDDEN_SIZE
bias_output = [0.0] * OUTPUT_SIZE


# ==========================================
# TEXT TO EMBEDDING
# ==========================================

def get_word_indices(sentence):

    return [
        word_index[word]
        for word in tokenize(sentence)
        if word in word_index
    ]


def sentence_vector(indices):

    vector = [0.0] * EMBED_SIZE

    if not indices:
        return vector

    for index in indices:

        for j in range(EMBED_SIZE):
            vector[j] += embeddings[index][j]

    for j in range(EMBED_SIZE):
        vector[j] /= len(indices)

    return vector


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def softmax(values):

    maximum = max(values)

    exp_values = [
        math.exp(value - maximum)
        for value in values
    ]

    total = sum(exp_values)

    return [value / total for value in exp_values]


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward(indices):

    inputs = sentence_vector(indices)

    hidden = []

    for j in range(HIDDEN_SIZE):

        total = bias_hidden[j]

        for i in range(EMBED_SIZE):
            total += inputs[i] * weights_hidden[i][j]

        hidden.append(math.tanh(total))

    output_raw = []

    for k in range(OUTPUT_SIZE):

        total = bias_output[k]

        for j in range(HIDDEN_SIZE):
            total += hidden[j] * weights_output[j][k]

        output_raw.append(total)

    outputs = softmax(output_raw)

    return inputs, hidden, outputs


# ==========================================
# TRAINING DATA PREPARATION
# ==========================================

training_examples = []

for sentence, label in examples:

    indices = get_word_indices(sentence)

    if indices:
        training_examples.append(
            (indices, label_index[label])
        )


# ==========================================
# TRAINING WITH BACKPROPAGATION
# ==========================================

def train():

    print("\nULTRON is learning word representations...")

    for epoch in range(EPOCHS):

        random.shuffle(training_examples)

        total_loss = 0.0

        for indices, target in training_examples:

            inputs, hidden, outputs = forward(indices)

            # Cross entropy loss
            total_loss -= math.log(
                max(outputs[target], 1e-12)
            )

            # Output gradients
            output_delta = outputs[:]
            output_delta[target] -= 1.0

            # Hidden gradients
            hidden_delta = []

            for j in range(HIDDEN_SIZE):

                error = sum(
                    output_delta[k] * weights_output[j][k]
                    for k in range(OUTPUT_SIZE)
                )

                hidden_delta.append(
                    error * (1 - hidden[j] ** 2)
                )

            # Calculate embedding gradients
            embedding_gradient = [0.0] * EMBED_SIZE

            for i in range(EMBED_SIZE):

                for j in range(HIDDEN_SIZE):

                    embedding_gradient[i] += (
                        hidden_delta[j] * weights_hidden[i][j]
                    )

            # Update output layer
            for j in range(HIDDEN_SIZE):

                for k in range(OUTPUT_SIZE):

                    weights_output[j][k] -= (
                        LEARNING_RATE *
                        output_delta[k] *
                        hidden[j]
                    )

            for k in range(OUTPUT_SIZE):

                bias_output[k] -= (
                    LEARNING_RATE * output_delta[k]
                )

            # Update hidden layer
            for i in range(EMBED_SIZE):

                for j in range(HIDDEN_SIZE):

                    weights_hidden[i][j] -= (
                        LEARNING_RATE *
                        hidden_delta[j] *
                        inputs[i]
                    )

            for j in range(HIDDEN_SIZE):

                bias_hidden[j] -= (
                    LEARNING_RATE * hidden_delta[j]
                )

            # Update word embeddings
            for index in indices:

                for i in range(EMBED_SIZE):

                    embeddings[index][i] -= (
                        LEARNING_RATE *
                        embedding_gradient[i] /
                        len(indices)
                    )

        if epoch % 300 == 0:

            average_loss = (
                total_loss / len(training_examples)
            )

            print(
                f"Epoch: {epoch} | "
                f"Loss: {average_loss:.6f}"
            )

    print("\nWord embedding training completed!")


# ==========================================
# SAVE MODEL
# ==========================================

def save_model():

    model = {
        "vocabulary": vocabulary,
        "labels": labels,
        "embedding_size": EMBED_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "embeddings": embeddings,
        "weights_hidden": weights_hidden,
        "weights_output": weights_output,
        "bias_hidden": bias_hidden,
        "bias_output": bias_output
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    print("Model saved:", MODEL_FILE)


# ==========================================
# PREDICTION
# ==========================================

def predict(sentence):

    indices = get_word_indices(sentence)

    if not indices:
        return "unknown", 0.0

    _, _, outputs = forward(indices)

    best_index = max(
        range(OUTPUT_SIZE),
        key=lambda i: outputs[i]
    )

    return labels[best_index], outputs[best_index]


# ==========================================
# CHAT RESPONSES
# ==========================================

responses = {

    "greeting": "Hello! I am ULTRON. How can I help?",

    "goodbye": "Goodbye! See you again.",

    "thanks": "You are welcome!",

    "how_are_you": "I am operational and learning.",

    "identity": "I am ULTRON, an AI learning project.",

    "creator": "My creator is Harsh.",

    "help": "I can recognize different types of messages.",

    "time": "I can recognize time-related questions."
}


def chat():

    print("\nULTRON: Semantic engine activated.")
    print("ULTRON: Ready for conversation.")
    print("Type /exit to close.")

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

            print("\nULTRON: I don't understand that yet.")

        else:

            print("\nULTRON:", responses.get(
                intent,
                "I don't have a response yet."
            ))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train()

    save_model()

    print("\n======================================")
    print("       ULTRON LEVEL 7 COMPLETE")
    print("======================================")

    print("Vocabulary:", VOCAB_SIZE)
    print("Embedding dimensions:", EMBED_SIZE)
    print("Hidden neurons:", HIDDEN_SIZE)
    print("Intent classes:", OUTPUT_SIZE)

    chat()
