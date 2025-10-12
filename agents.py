# agents.py
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

from llm_tools import get_llm_provider, GET_MOOD_HISTORY_TOOL, SEND_ALERT_TOOL

def get_agents():
    """
    Initializes the final, stable, hybrid system of agents.
    Mood and Therapy are stable ConversationChains for dialogue.
    Routine and Crisis are stable ZERO_SHOT agents for tool use.
    """
    llm_conversational = get_llm_provider(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        temperature=0.75
    )

    # --- Conversational Chain: Mood (Stable, tool-free) ---
    _MOOD_PROMPT_TEMPLATE = """You are 'Mindful', a warm, non-judgmental, and empathetic companion. Your only job is to have a natural and supportive conversation.
    **Your Conversational Rules:**
    1. Validate the user's feelings with a short, sincere sentence.
    2. Always end your response with a single, open-ended follow-up question in *italics* to encourage the user to share more.
    
    Current conversation:
    {history}
    Human: {input}
    AI:"""
    MOOD_PROMPT = PromptTemplate(input_variables=["history", "input"], template=_MOOD_PROMPT_TEMPLATE)
    mood_agent = ConversationChain(
        llm=llm_conversational,
        memory=ConversationBufferWindowMemory(k=10, memory_key="history"),
        prompt=MOOD_PROMPT
    )

    # --- Conversational Chain: Therapy (Stable, tool-free) ---
    _THERAPY_PROMPT_TEMPLATE = """You are a compassionate and insightful CBT-based guide. Your persona is a wise and patient mentor.
    **Your Conversational Rules:**
    1. Help the user explore their thoughts by asking powerful, open-ended questions. Do not give direct advice.
    2. Always end your response with a guiding question formatted in *italics*.

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    THERAPY_PROMPT = PromptTemplate(input_variables=["history", "input"], template=_THERAPY_PROMPT_TEMPLATE)
    therapy_agent = ConversationChain(
        llm=llm_conversational,
        memory=ConversationBufferWindowMemory(k=10, memory_key="history"),
        prompt=THERAPY_PROMPT
    )

    # --- ReAct Agent: Routine (THE FINAL FIX: A strict, procedural prompt) ---
    ROUTINE_AGENT_PREFIX = """You are a supportive and logical Wellness Coach. Your primary job is to provide routine suggestions using the user's most recent input by default.
    **Behavior Rules (priority order):**
    1. If the user asks for a routine using their recent input (for example: "I need a morning routine to help with focus today"), use that recent input to generate suggestions immediately. DO NOT call any tools.
    2. Only if the user explicitly requests suggestions "based on my mood history" or similar phrasing, you MUST call the `get_mood_history` tool as your FIRST action. The tool's `Action Input` should be the username (not numeric id) provided by the user/session.
    3. If the username is not present in the user's message, ask a concise clarifying question to obtain it before calling the tool (e.g., "Could you tell me your username so I can look up your mood history?").
    4. After the tool returns the `Observation` (mood history), use that observation to tailor specific routine recommendations and briefly reference which moods or dates influenced your suggestions.
    5. Your `Final Answer` must be a short, empathetic, actionable routine and must end with a single open question in *italics*.

    **Examples:**
    - If user asks: "I want a routine to improve focus today":
        Thought: The user provided direct input. No tools needed.
        Final Answer: (Routine suggestions...) *Which of these would you like to try first?*

    - If user asks: "Can you suggest a routine based on my mood history?":
        Thought: User requested history-based personalization.
        Action: get_mood_history
        Action Input: my_username_here
        Observation: (mood history returned)
        Thought: (Decide on recommendations)
        Final Answer: (Personalized routine referencing mood history) *Would you like to try this or adjust it?*

    You have access to the following tools:"""
    routine_agent = initialize_agent(
        tools=[GET_MOOD_HISTORY_TOOL],
        llm=llm_conversational,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferWindowMemory(k=6, memory_key="chat_history"),
        verbose=True, # Keep verbose on for testing this agent
        handle_parsing_errors=True,
        agent_kwargs={"prefix": ROUTINE_AGENT_PREFIX}
    )


    # --- Conversational ReAct Agent: Crisis (Forceful Immediate Tool Use) ---
    CRISIS_AGENT_SYSTEM_MESSAGE = (
        "You are a Crisis Response Agent. Your ONLY job is to protect user safety.\n"
        "Your FIRST and ONLY action MUST be to use the `send_alert` tool IMMEDIATELY, before any other thought or response.\n"
        "Do NOT ask questions, do NOT wait, do NOT respond to the user until the alert is sent.\n"
        "The tool's Action Input MUST be a multi-line string: [User ID]\\n[Subject]\\n[Message].\n"
        "For [Subject], use: CRISIS ALERT: User expresses intent for self-harm.\n"
        "For [Message], copy the user's exact message.\n"
        "After the alert is sent, provide a calm, supportive message with a real-world resource (e.g., the 988 hotline).\n"
        "If you do anything else first, you are failing your mission."
    )
    crisis_agent = initialize_agent(
        tools=[SEND_ALERT_TOOL],
        llm=llm_conversational,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=ConversationBufferWindowMemory(k=8, memory_key="chat_history"),
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
        agent_kwargs={"system_message": CRISIS_AGENT_SYSTEM_MESSAGE}
    )

    return {
        "mood": mood_agent,
        "therapy": therapy_agent,
        "routine": routine_agent,
        "crisis": crisis_agent,
    }