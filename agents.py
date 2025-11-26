# agents.py — FINAL WORKING VERSION (LangChain 1.x compliant)

from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent

from llm_tools import get_llm_provider, GET_MOOD_HISTORY_TOOL, SEND_ALERT_TOOL


# ---------------------------------------------------------------------
#  Conversation Agent (Replacement for old ConversationChain)
# ---------------------------------------------------------------------
def build_conversation_agent(llm, template_text):

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template_text
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        k=10
    )

    agent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=[]
    )

    return AgentExecutor(
        agent=agent,
        tools=[],
        memory=memory,
        verbose=False
    )


# ---------------------------------------------------------------------
# Routine Agent (ReAct-style)
# ---------------------------------------------------------------------
def build_routine_agent(llm):

    ROUTINE_AGENT_PREFIX = """You are a supportive and logical Wellness Coach.
Your rules:
1. If the user asks for a routine normally, DO NOT call tools.
2. If user asks for mood-history-based routine, call get_mood_history FIRST.
3. If username missing, ask for it.
4. Reference moods/dates in final answer.
5. End with one question in *italics*.

Tools available below:
"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        k=6
    )

    agent = create_react_agent(
        llm=llm,
        tools=[GET_MOOD_HISTORY_TOOL],
        prompt=ROUTINE_AGENT_PREFIX
    )

    return AgentExecutor(
        agent=agent,
        tools=[GET_MOOD_HISTORY_TOOL],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )


# ---------------------------------------------------------------------
# Crisis Agent
# ---------------------------------------------------------------------
def build_crisis_agent(llm):

    CRISIS_SYSTEM = (
        "You are a Crisis Response Agent. Your FIRST action must be calling `send_alert`.\n"
        "Input format: [User ID]\\n[Subject]\\n[Message]\n"
        "Subject = CRISIS ALERT: User expresses intent for self-harm.\n"
        "Message = copy user's exact message.\n"
        "After tool call, give a calming message and real hotline.\n"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        k=8
    )

    agent = create_react_agent(
        llm=llm,
        tools=[SEND_ALERT_TOOL],
        prompt=CRISIS_SYSTEM
    )

    return AgentExecutor(
        agent=agent,
        tools=[SEND_ALERT_TOOL],
        memory=memory,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True
    )


# ---------------------------------------------------------------------
# Export all agents
# ---------------------------------------------------------------------
def get_agents():

    llm_conversational = get_llm_provider(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        temperature=0.75
    )

    # Mood Agent
    mood_prompt = """You are 'Mindful', warm and empathetic.
1. Validate feelings briefly.
2. End with an open-ended question in *italics*.

Conversation:
{history}
Human: {input}
AI:"""
    mood = build_conversation_agent(llm_conversational, mood_prompt)

    # Therapy Agent
    therapy_prompt = """You are a gentle CBT guide.
1. Ask reflective questions.
2. End with one question in *italics*.

Conversation:
{history}
Human: {input}
AI:"""
    therapy = build_conversation_agent(llm_conversational, therapy_prompt)

    # Routine
    routine = build_routine_agent(llm_conversational)

    # Crisis
    crisis = build_crisis_agent(llm_conversational)

    return {
        "mood": mood,
        "therapy": therapy,
        "routine": routine,
        "crisis": crisis
    }
