
from tools.prompts import instruction_str, prompt_template, context
from dotenv import load_dotenv
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from data_operations import load_articles_df
import os

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

def query_agent(articles_df):
    """
    Creates a PandasQueryEngine agent that allows user interaction with a chatbot over Medium article data.
    """
    if articles_df is not None:
        try:
            # Set up the Pandas query engine
            articles_query_engine = PandasQueryEngine(
                df=articles_df,
                verbose=True,
                instruction_str=instruction_str
            )
            articles_query_engine.update_prompts({"pandas_prompt": prompt_template})

            # Define metadata for the query tool
            articles_metadata = ToolMetadata(
                name="articles_data",
                description=(
                    "This tool provides facts from Medium articles. "
                    "Use detailed plain-text questions for best results."
                ),
            )

            # Create the tool wrapper
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=articles_query_engine,
                    metadata=articles_metadata,
                ),
            ]

            # Initialize OpenAI LLM using environment variable for API key
            llm = OpenAI()  # Automatically uses OPENAI_API_KEY from .env

            # Set up the ReAct agent with the tool
            agent = ReActAgent.from_tools(
                tools=query_engine_tools,
                llm=llm,
                verbose=True,
                context=context
            )

            # Run a query loop
            while (prompt := input("Enter a prompt (q to quit): ")) != "q":
                result = agent.query(prompt)
                print(result)

        except Exception as e:
            print(f"An error occurred while running the query agent: {str(e)}")
    else:
        print("No data to query.")

if __name__ == "__main__":
    articles_df = load_articles_df()
    query_agent(articles_df)
