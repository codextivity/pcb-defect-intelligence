# app/core/agent.py
# LangChain agent for PCB quality management queries.
#
# Allows quality engineers to ask questions in natural language:
#   "What is our yield rate today?"
#   "Which defect type is most common?"
#   "How many PCBs failed inspection?"
#   "What are the recommendations to improve yield?"

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from app.core.database import get_quality_summary, query_defects
from app.config import settings


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def get_overall_quality() -> str:
    """
    Returns overall PCB quality statistics from all inspections.

    Use when asked about:
    - Overall yield rate
    - Total defects found
    - General quality performance
    - How many PCBs passed or failed
    """
    summary = get_quality_summary()
    total = summary["total_inspections"]

    if total == 0:
        return "No PCB inspections have been recorded yet."

    lines = [
        f"PCB Quality Summary (all time):",
        f"",
        f"  Total inspections:    {total}",
        f"  Passed:               {summary['passed']}",
        f"  Failed:               {summary['failed']}",
        f"  Uncertain:            {summary['uncertain']}",
        f"  Yield rate:           {summary['yield_rate']:.1%}",
        f"  Total defects found:  {summary['total_defects']}",
        f"  Avg defects per PCB:  {summary['avg_defects_per_pcb']:.2f}",
        f"",
        f"  Defects by type:",
    ]

    for dtype, count in summary["defects_by_type"].items():
        pct = count / summary["total_defects"] * 100 if summary["total_defects"] > 0 else 0
        lines.append(f"    {dtype:25} {count:5d} ({pct:.1f}%)")

    if summary["yield_rate"] >= 0.95:
        lines.append(f"\n  Assessment: ✅ Excellent yield — production line is healthy")
    elif summary["yield_rate"] >= 0.80:
        lines.append(f"\n  Assessment: ⚠ Acceptable yield — monitor closely")
    else:
        lines.append(f"\n  Assessment: ❌ Low yield — immediate investigation needed")

    return "\n".join(lines)


@tool
def get_recent_quality(hours: int = 24) -> str:
    """
    Returns quality statistics for recent inspections.

    Use when asked about:
    - Today's yield rate
    - Recent defects
    - Latest quality performance
    - What happened recently

    Args:
        hours: how many hours back to look (default 24)
    """
    date_from = (
        datetime.now() - timedelta(hours=hours)
    ).isoformat()

    summary = get_quality_summary(date_from=date_from)
    period = f"last {hours} hours"

    if summary["total_inspections"] == 0:
        return f"No inspections recorded in the {period}."

    lines = [
        f"PCB Quality Summary ({period}):",
        f"",
        f"  Total inspections:    {summary['total_inspections']}",
        f"  Passed:               {summary['passed']}",
        f"  Failed:               {summary['failed']}",
        f"  Yield rate:           {summary['yield_rate']:.1%}",
        f"  Total defects:        {summary['total_defects']}",
    ]

    if summary["defects_by_type"]:
        lines.append(f"\n  Defects by type:")
        for dtype, count in summary["defects_by_type"].items():
            lines.append(f"    {dtype:25} {count}")

    return "\n".join(lines)


@tool
def get_defect_details(
    defect_type: str = None,
    limit: int = 10
) -> str:
    """
    Returns details of specific defect records.

    Use when asked about:
    - Specific defect types
    - Recent defect incidents
    - Details of individual defects

    Args:
        defect_type: filter by defect type name or None for all
        limit: maximum records to return
    """
    defects = query_defects(defect_type=defect_type, limit=limit)

    if not defects:
        msg = f"No defects found"
        if defect_type:
            msg += f" of type '{defect_type}'"
        return msg + "."

    lines = [
        f"Recent defects"
        f"{f' ({defect_type})' if defect_type else ''}:"
        f" {len(defects)} records"
    ]

    for d in defects:
        verified = "GPT-4o verified" if d["verified_by_vlm"] else "YOLO detection"
        lines.append(
            f"\n  Defect ID {d['id']}:"
            f"\n    Type:       {d['defect_type']}"
            f"\n    Detected:   {d['detected_at']}"
            f"\n    Confidence: {d['confidence']:.2f}"
            f"\n    Area:       {d['bbox_area_pct']:.3f}% of image"
            f"\n    Verified:   {verified}"
            f"\n    PCB status: {d['quality_status']}"
        )

    return "\n".join(lines)


@tool
def get_most_common_defect() -> str:
    """
    Returns the most common defect type and recommendations.

    Use when asked about:
    - Which defect is most frequent
    - What should we focus on improving
    - Defect type ranking
    """
    summary = get_quality_summary()
    by_type = summary.get("defects_by_type", {})

    if not by_type:
        return "No defect data available yet."

    sorted_defects = sorted(
        by_type.items(), key=lambda x: x[1], reverse=True
    )

    lines = ["Defect frequency ranking:"]
    total = sum(by_type.values())

    for rank, (dtype, count) in enumerate(sorted_defects, 1):
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {rank}. {dtype:25} {count:5d} ({pct:.1f}%) {bar}")

    most_common = sorted_defects[0][0]
    lines.append(f"\nMost common defect: {most_common}")

    # Defect-specific recommendations
    recommendations = {
        "missing_hole": (
            "Check drill bit condition and positioning accuracy. "
            "Missing holes often indicate drill wear or misalignment."
        ),
        "mouse_bite": (
            "Inspect PCB edge routing process. "
            "Mouse bite defects typically occur during board singulation."
        ),
        "open_circuit": (
            "Review etching process and copper thickness. "
            "Open circuits may indicate under-etching or thin copper."
        ),
        "short": (
            "Check etching time and resist adhesion. "
            "Short circuits often result from over-etching or resist failure."
        ),
        "spur": (
            "Inspect photolithography exposure and development. "
            "Spurs indicate resist residue after etching."
        ),
        "spurious_copper": (
            "Review etching completeness and board cleanliness. "
            "Spurious copper indicates incomplete etching."
        ),
    }

    if most_common in recommendations:
        lines.append(f"\nRecommendation: {recommendations[most_common]}")

    return "\n".join(lines)


@tool
def get_quality_recommendations() -> str:
    """
    Generates quality improvement recommendations based on defect patterns.

    Use when asked about:
    - How to improve yield
    - Quality recommendations
    - Action items for quality improvement
    - What should we do to reduce defects
    """
    summary = get_quality_summary()
    by_type = summary.get("defects_by_type", {})
    yield_rate = summary.get("yield_rate", 1.0)
    total = summary.get("total_inspections", 0)

    if total == 0:
        return "No inspection data available for recommendations."

    recommendations = []

    if yield_rate < 0.95:
        recommendations.append(
            f"1. Current yield rate is {yield_rate:.1%} — target is 95%+. "
            f"Investigate the production process for systematic issues."
        )

    if by_type:
        most_common = max(by_type.items(), key=lambda x: x[1])
        dtype, count = most_common

        defect_actions = {
            "missing_hole":    "Check drill bits and CNC positioning accuracy",
            "mouse_bite":      "Inspect board singulation and routing process",
            "open_circuit":    "Review copper etching time and chemical concentration",
            "short":           "Check photoresist adhesion and etching uniformity",
            "spur":            "Review photolithography exposure and development time",
            "spurious_copper": "Inspect etching completeness and board cleaning",
        }

        action = defect_actions.get(dtype, "Review manufacturing process")
        recommendations.append(
            f"2. Most common defect is '{dtype}' ({count} instances). "
            f"Priority action: {action}."
        )

    uncertain = summary.get("uncertain", 0)
    if uncertain > 0:
        recommendations.append(
            f"3. {uncertain} inspections flagged as UNCERTAIN. "
            f"Enable GPT-4o verification to resolve these cases "
            f"and improve detection confidence."
        )

    recommendations.append(
        "4. Schedule regular model retraining as new defect patterns emerge. "
        "Use drift detection to monitor when production data diverges "
        "from training distribution."
    )

    if not recommendations:
        return "Quality is excellent. Maintain current process controls."

    return "Quality Improvement Recommendations:\n\n" + "\n\n".join(recommendations)


# ── Agent State ────────────────────────────────────────────────────────────────

class QualityAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Build Agent ────────────────────────────────────────────────────────────────

QUALITY_AGENT_PROMPT = """You are PCBVision, an AI quality management assistant
for PCB (Printed Circuit Board) manufacturing inspection.

You have access to a database of PCB defect records collected by
computer vision analysis of PCB inspection images.

Your tools query this database to answer questions about:
- PCB yield rates and quality statistics
- Defect types and frequencies
- Recent inspection results
- Quality improvement recommendations

Guidelines:
- Always cite specific numbers from the database
- Distinguish between YOLO-detected and GPT-4o verified defects
- Provide actionable recommendations tied to specific defect types
- Be direct and professional — quality engineers need clear data

If asked about something not in the database, say so clearly."""


def build_quality_agent():
    """Builds LangGraph quality management agent."""
    tools = [
        get_overall_quality,
        get_recent_quality,
        get_defect_details,
        get_most_common_defect,
        get_quality_recommendations,
    ]

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        api_key=settings.openai_api_key
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: QualityAgentState) -> dict:
        messages = [
            SystemMessage(content=QUALITY_AGENT_PROMPT)
        ] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tools_node = ToolNode(tools)

    def should_continue(state: QualityAgentState) -> str:
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(QualityAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_quality_agent(
    agent,
    question: str,
    chat_history: list = None
) -> str:
    """Runs the quality agent and returns the answer."""
    messages = list(chat_history or [])
    messages.append(HumanMessage(content=question))
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content