from model import LocalModel, ModelLoadError
from prompts import SYSTEM_PROMPT

from config import MAX_HISTORY_MESSAGES, PROMPT_TOKEN_BUDGET


def estimate_tokens(text):
    """Crude ~4 chars/token estimate. Good enough to avoid gross
    context overflows without pulling in a real tokenizer."""
    return max(1, len(text) // 4)


class Assistant:

    def __init__(self):
        self.model = LocalModel()
        self.history = []

    def clear_history(self):
        self.history = []
        print("\nConversation cleared.\n")

    def build_messages(self, user_message):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Hard cap on message count, then trim further by estimated
        # token budget so we don't silently overflow CONTEXT_SIZE.
        candidate_history = self.history[-MAX_HISTORY_MESSAGES:]

        budget = PROMPT_TOKEN_BUDGET
        budget -= estimate_tokens(SYSTEM_PROMPT)
        budget -= estimate_tokens(user_message)

        trimmed = []
        for msg in reversed(candidate_history):
            cost = estimate_tokens(msg["content"])
            if cost > budget:
                break
            trimmed.insert(0, msg)
            budget -= cost

        messages.extend(trimmed)

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    def respond(self, user_message):
        messages = self.build_messages(user_message)

        print("\nAI: ", end="", flush=True)

        full_response = ""

        for token in self.model.generate(messages):
            print(token, end="", flush=True)
            full_response += token

        print("\n")

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_response})

        return full_response


def print_help():
    print("""
Commands:

  /help       Show this help
  /clear      Clear conversation history
  /exit       Exit the assistant

Anything else is sent to the local AI model.
Start a message with // if you need it to begin with a literal '/'.
""")


def main():
    print("=" * 50)
    print("        Local AI Assistant")
    print("=" * 50)
    print("Type /help for commands.")
    print("Type /exit to quit.")
    print()

    try:
        assistant = Assistant()
    except ModelLoadError as exc:
        print(f"\nCould not start assistant:\n{exc}\n")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print()
            break

        if not user_input:
            continue

        # Escape hatch: "//something" sends a literal message starting
        # with "/", instead of being treated as a command.
        if user_input.startswith("//"):
            assistant.respond(user_input[1:])
            continue

        command = user_input.lower()

        if command == "/exit":
            print("Goodbye.")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/clear":
            assistant.clear_history()
            continue

        assistant.respond(user_input)


if __name__ == "__main__":
    main()