"""CLI demo for the Buddy-Tech chatbot."""
from __future__ import annotations

import json
import os
from typing import List, Dict

import openai
from dotenv import load_dotenv


class Memory:
    """Store chat history and handle periodic summarization."""

    def __init__(self, initial: List[Dict[str, str]]) -> None:
        self.messages: List[Dict[str, str]] = list(initial)
        self.turns = 0
        self.summary = ""

    def add(self, role: str, content: str) -> None:
        """Add a message and summarize every three user turns."""
        self.messages.append({"role": role, "content": content})
        if role == "user":
            self.turns += 1
            if self.turns % 3 == 0:
                self._append_summary()

    def _append_summary(self) -> None:
        """Use OpenAI to summarize the conversation in one sentence."""
        prompt = self.messages + [
            {
                "role": "system",
                "content": (
                    "Summarize the conversation above in one concise sentence "
                    "while keeping ACME-Shop's friendly expert tone."
                ),
            }
        ]
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", temperature=0, messages=prompt
        )
        self.summary = resp.choices[0].message["content"].strip()
        self.messages.append({"role": "system", "content": f"Summary: {self.summary}"})


def load_prompts(path: str) -> List[Dict[str, str]]:
    """Load system and few-shot prompts from a JSON markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    msgs = [{"role": "system", "content": data["system"]}]
    for pair in data.get("examples", []):
        msgs.append({"role": "user", "content": pair["user"]})
        msgs.append({"role": "assistant", "content": pair["assistant"]})
    return msgs


def ask_openai(messages: List[Dict[str, str]]) -> str:
    """Send messages to OpenAI and return the assistant reply."""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", temperature=0, messages=messages
    )
    return resp.choices[0].message["content"].strip()


def main() -> None:
    """Run the conversation loop for five user turns."""
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    messages = load_prompts("prompts.md")
    memory = Memory(messages)

    for _ in range(5):
        user_input = input("You: ")
        memory.add("user", user_input)
        reply = ask_openai(memory.messages)
        print(f"Buddy-Tech: {reply}")
        memory.add("assistant", reply)

    print("\nFinal summary:\n" + memory.summary)


if __name__ == "__main__":
    main()
