import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model="gemma2-9b-it",temperature=0)  #mixtral-8x7b-32768


st.set_page_config(page_title="Talk to your Database")
st.header("Talk to your database")

db = SQLDatabase.from_uri("sqlite:///demo.db")


agent_executor = create_sql_agent(
    llm, db = db, agent_type =AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True,handle_parsing_errors=True
)

user_question = st.text_input("Ask a question about your database:")

if user_question is not None and user_question != "":
    with st.spinner(text="In progress..."):
        response = agent_executor.invoke(user_question)
        st.write(response["output"])