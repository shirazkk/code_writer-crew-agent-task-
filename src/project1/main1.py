from crewai.flow.flow import Flow, start
from project1.crews.code_crew.code_crew import CodeCrew

class CodeFlow(Flow):

    @start()
    def python_coder(self):
        result = CodeCrew().crew().kickoff(
            inputs={
                "problem": "write a python code which add two numbers."
            }
        )
        return result.raw

def kickoff():
    result = CodeFlow()
    result.kickoff()