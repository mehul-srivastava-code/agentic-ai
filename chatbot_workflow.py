from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages


load_dotenv()  # Load environment variables from .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot=graph.compile()


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    return str(content)


if __name__ == "__main__":
    print(chatbot.get_graph().draw_ascii())

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        response = chatbot.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        bot_content = content_to_text(response["messages"][-1].content)
        print(f"Bot: {bot_content}")