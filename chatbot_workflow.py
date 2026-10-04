from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


load_dotenv()


class ChatState(TypedDict):
    """The messages that are passed through the chatbot."""

    messages: Annotated[list[BaseMessage], add_messages]


model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)


def chat(state: ChatState) -> ChatState:
    """Send the conversation to Gemini and return its reply."""

    reply = model.invoke(state["messages"])
    return {"messages": [reply]}


def create_chatbot():
    """Create a one-step LangGraph chatbot."""

    workflow = StateGraph(ChatState)
    workflow.add_node("chat", chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    return workflow.compile()


chatbot = create_chatbot()


def content_to_text(content: object) -> str:
    """Convert Gemini's response content into normal text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)

    return str(content)


def run_terminal_chat() -> None:
    """Run the chatbot in the terminal until the user types 'exit'."""

    print("Chatbot is ready. Type 'exit' to stop.")

    while True:
        user_text = input("You: ")
        if user_text.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        result = chatbot.invoke({"messages": [HumanMessage(content=user_text)]})
        answer = content_to_text(result["messages"][-1].content)
        print(f"Bot: {answer}")


if __name__ == "__main__":
    run_terminal_chat()