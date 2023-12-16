import streamlit as st
from LLM import *
import asyncio
from email_for_user import send_mail
from validate_email import check
from langchain.memory import StreamlitChatMessageHistory
import yaml

with open("config.yaml", "r") as file:
    config_data = yaml.safe_load(file)


async def main():
    valid_email = False
    st.set_page_config(
        page_title=config_data["streamlit"]["title"],
        page_icon=config_data["streamlit"]["icon"],
    )
    st.title("Roam Rover")
    greetings = "Welcome! I'm Roam Rover, your virtual travel guide, Ready to plan your next adventure? Let's map out a fantastic itinerary with exciting places and activities for you!"

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": greetings}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Message Roam Rover...", key="user_prompt")

    if user_prompt is not None:
        if "@" in user_prompt:
            valid_email = await check(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
        chat_memory=msgs,
    )

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Please Wait"):
                bot_response = await generate_chat(user_prompt, memory)
                st.write(bot_response)
                print("type of bot_response", type(bot_response))
            if valid_email:
                user_email = user_prompt
                if "Plan for" in bot_response or "email for" in bot_response:
                    user_plan = bot_response
                    await send_mail(user_plan, user_email)
        new_bot_response = {"role": "assistant", "content": bot_response}
        st.session_state.messages.append(new_bot_response)


asyncio.run(main())
