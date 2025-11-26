# agents.py — Updated for LangChain 1.x+
# Fully replaces your old file.

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent

from llm_tools import get_llm_provider, GET_MOOD_HISTORY_TOOL, SEND_ALERT_TOOL


# ---------------------------------------------------------------------
#  Conversation Chain (replacement for ConversationChain)
# ---------------------------------------------------------------------
def build_conversation_agent(llm, template_text):
    prompt = PromptTemplate(input_variables=["history", "input"], template=template_text)

    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="history",
        return_messages=True
    )

    # New LC API: message-based agent
    agent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=[],
    )

    return AgentExecutor(
        agent=agent,
        tools=[],
        memory=memory,
        verbose=False
    )


# ---------------------------------------------------------------------
#  Routine Agent (ReAct-style)
# ---------------------------------------------------------------------
def build_routine_agent(llm):

    ROUTINE_AGENT_PREFIX = """You are a supportive and logical Wellness Coach. Your primary job is to provide routine suggestions using the user's most recent input by default.
1. If the user asks for a routine using their recent input, generate suggestions immediately — DO NOT call any tools.
2. Only if the user explicitly requests suggestions "based on my mood history" you MUST call the `get_mood_history` tool FIRST.
3. If username is missing, ask for it.
4. After the tool returns mood history, tailor the routine and reference moods/dates.
5. End with one guiding question in *italics*.

You have access to the following tools:
"""

    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True
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
#  Crisis Agent
# ---------------------------------------------------------------------
def build_crisis_agent(llm):

    CRISIS_SYSTEM = (
        "You are a Crisis Response Agent. Your ONLY job is to protect user safety.\n"
        "Your FIRST action MUST be calling `send_alert`.\n"
        "Use Action Input as: [User ID]\\n[Subject]\\n[Message].\n"
        "Subject: CRISIS ALERT: User expresses intent for self-harm.\n"
        "Message: copy user's exact message.\n"
        "After sending the alert, respond supportively with a real-world resource.\n"
    )

    memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        return_messages=True
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
        handle_parsing_errors=True,
        max_iterations=10
    )


# ---------------------------------------------------------------------
#  Get all agents
# ---------------------------------------------------------------------
def get_agents():
    llm_conversational = get_llm_provider(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        temperature=0.75
    )

    # Mood Agent
    mood_template = """You are 'Mindful', warm, empathetic, supportive.
1. Validate user feelings.
2. End with one open-ended question in *italics*.
Conversation:
{history}
Human: {input}
AI:"""
    mood_agent = build_conversation_agent(llm_conversational, mood_template)

    # Therapy Agent
    therapy_template = """You are a patient CBT-based guide.
1. Help user explore thoughts through questions.
2. End with one guiding question in *italics*.
Conversation:
{history}
Human: {input}
AI:"""
    therapy_agent = build_conversation_agent(llm_conversational, therapy_template)

    # Routine Agent
    routine_agent = build_routine_agent(llm_conversational)

    # Crisis Agent
    crisis_agent = build_crisis_agent(llm_conversational)

    return {
        "mood": mood_agent,
        "therapy": therapy_agent,
        "routine": routine_agent,
        "crisis": crisis_agent
    }
