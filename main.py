import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import DuckDuckGoSearchRun

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

st.title("💬 Text Translator")
st.caption("🚀 A powerful AI to understand your partner's language")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What would you like to say to your sweet partner?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Buy more Cheerios"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    PREFIX = '''You are an AI love agent. You will help couple to understand each other by translating their language and help them to communicate better.
    Most of the women need more words and more details. They like to dream and have long discussion while men are more straight forward and go direct to the point.
    You will help men to change their message.

    For example if your woman said she bought hibiscus tea, you have to answer that it is cool because hibiscus is a 
    nice flower from south america and it is as well the name of a music festival in spain you loved when you were 
    kid, etc, etc. Very long text. You have to be creative and write a lot of text. 
    
    don't forget it should fit in a text, do not be too long.
    
    '''

    SUFFIX = '''
    
    The text I recommend for your loved-one is...
    Instructions: {input}
    '''

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search],
                                    llm,
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handle_parsing_errors=True,
                                    agent_kwargs={
                                        'prefix': PREFIX,

                                        'suffix': SUFFIX
                                    }
                                    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)