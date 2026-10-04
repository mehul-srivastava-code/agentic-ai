import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_workflow import chatbot, content_to_text


st.set_page_config(page_title="first chat bot", page_icon=":speech_balloon:")
st.title("first chat bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.session_state.graph_messages.append(HumanMessage(content=prompt))
            message_chunks = chatbot.stream(
                {"messages": st.session_state.graph_messages},
                stream_mode="messages",
            )

            def text_chunks():
                for message_chunk, _metadata in message_chunks:
                    text = content_to_text(message_chunk.content)
                    if text:
                        yield text

            answer = st.write_stream(text_chunks())

        st.session_state.graph_messages.append(AIMessage(content=answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})
