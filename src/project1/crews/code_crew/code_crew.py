import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM

api_key= os.getenv("GEMINI_API_KEY")

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=api_key
)


@CrewBase
class CodeCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def junior_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["junior_developer"],
            llm=llm
             
        )
    @agent
    def senior_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["senior_developer"],
            llm=llm
        )

    @task
    def write_code(self) -> Task:
        return Task(
            config=self.tasks_config["write_code"],
        )
    @task
    def review_code(self) -> Task:
        return Task(
            config=self.tasks_config["review_code"],
        )

    @crew
    def crew(self) -> Crew:

        return Crew(
            agents=self.agents, 
            tasks=self.tasks,  
            process=Process.sequential,
            verbose=True,
        )
