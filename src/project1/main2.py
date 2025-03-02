from crewai import Agent, Task, Crew, LLM, Process
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
import os

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# Instantiate the LLM
llm = LLM(
    model='gemini/gemini-1.5-flash',
    api_key=GEMINI_API_KEY,
)

text_source = TextFileKnowledgeSource(
    file_paths=["document.txt"]
)

# Create an agent that uses the knowledge source
agent = Agent(
    role="About papers",
    goal="You know everything about the papers {question}.",
    backstory="You are a master at understanding papers and their content.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    knowledge_sources=[text_source],
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
    }
)

# Define a task for the agent
task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent
)

# Create a crew with the agent and task, and add the knowledge source at the crew level as well
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[text_source],
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
    }
)
def doc_knowledge():
    crew.kickoff(
            inputs={
                "question": "What are Real-World Applications of Agentic Ai?"
            }
        )

if __name__ == "__main__":
    doc_knowledge()
