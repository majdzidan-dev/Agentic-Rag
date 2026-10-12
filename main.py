"""Entry point - run this file to chat with the agent."""

from agent import agent
from llm_utils import extract_text

# A "thread_id" is just a label for one conversation. Every invoke() call
# using the same thread_id shares memory - the checkpointer in agent.py
# looks up everything said so far under this ID automatically.
CONFIG = {"configurable": {"thread_id": "main-conversation"}}


def main():
    print("My AI agent is ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=CONFIG,
        )
        response = extract_text(result["messages"][-1].content)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()