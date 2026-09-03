# test_agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.agent import build_quality_agent, run_quality_agent
from langchain_core.messages import HumanMessage, AIMessage

agent = build_quality_agent()

questions = [
    "What is our overall yield rate?",
    "Which defect type is most common?",
    "How many PCBs failed inspection?",
    "What quality improvements do you recommend?",
    "Give me a summary of recent defects.",
]

print("=" * 60)
print("PCB QUALITY AGENT TEST")
print("=" * 60)

history = []

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 40)
    answer = run_quality_agent(agent, question, history)
    print(f"A: {answer}")

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))