import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.tools import Tool


def main(
    task: str = "Print heello world",
    question_for_csv_oppration: str = "give the basic statistics of the csv file",
):
    print("sterted ....")
    instructions = """
    
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    python_agent = create_react_agent(prompt=prompt, tools=tools, llm=llm)

    agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    python_result = agent_executor.invoke({"input": task})

    result = []
    result.append(python_result)

    # ###################### for csv file ##########################

    csv_agent = create_csv_agent(
        llm=llm, path="episode_info.csv", verbose=True, allow_dangerous_code=True
    )

    question_for_csv_oppration += " the used print function whatever print it give back as answer in the format of natural language like fact or sentence"

    csv_response = csv_agent.invoke({"input": question_for_csv_oppration})

    result.append(csv_response)

    return result


def raout(query: str):

    print("sterted ....")
    instructions = """
    
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    python_agent = create_react_agent(prompt=prompt, tools=tools, llm=llm)

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # python_result = agent_executor.invoke({"input": task})

    # result =[]
    # result.append(python_result)

    # ###################### for csv file ##########################

    csv_agent = create_csv_agent(
        llm=llm, path="episode_info.csv", verbose=True, allow_dangerous_code=True
    )

    # question_for_csv_oppration+=" the used print function whatever print it give back as answer in the format of natural language like fact or sentence"

    # csv_response = csv_agent.invoke({"input": question_for_csv_oppration})

    # result.append(csv_response)

    # return result

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""
                           useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT
                          give input as natural language and it will return the result of the code execution
                
            """,
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""
                           useful when you need to answer question from csv file (this take csv file automatically not require to provide the path of the csv file nor the name of the csv file),
                         takes an input the entire question and returns the answer after running pandas calculations
                
            """,
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(prompt=prompt, tools=tools, llm=llm)
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    result = grand_agent_executor.invoke({"input": query})

    return result


if __name__ == "__main__":

    query = input("Enter your query: ")

    result = raout(query)

    print(f"{result=}")

    continues = input("Do you want to continue? (y/n): ")

    if continues.lower() == "y":

        task = input("Enter your task: ")
        question_for_csv_oppration = input("Enter your question for csv operation: ")
        if not task:
            task = "Print hello world"
        if not question_for_csv_oppration:
            question_for_csv_oppration = "give the basic statistics of the csv file"

        result = main(task=task, question_for_csv_oppration=question_for_csv_oppration)

        print(f"{result=}")
