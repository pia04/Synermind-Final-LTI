# router.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm_tools import get_llm_provider, contains_crisis_keywords

# Create a high-reasoning LLM instance specifically for routing, using Gemini.
llm_router = get_llm_provider(provider="gemini", model_name="gemini-2.5-flash", temperature=0.0)

_ALLOWED = {"mood", "therapy", "routine", "crisis"}

# --- FINAL, CONTEXT-AWARE ROUTER PROMPT ---
router_prompt = PromptTemplate.from_template(
    "You are an expert conversational analyst. Your task is to read the following conversation transcript and decide which specialist agent should handle the VERY NEXT TURN. "
    "Choose exactly one label from [mood, therapy, routine, crisis].\n\n"
    "**Your Decision-Making Process:**\n"
    "1. Read the ENTIRE conversation to understand the user's journey.\n"
    "2. Pay close attention to the most recent user message.\n"
    "3. If the conversation is becoming deeper, more complex, or is exploring causes and struggles (e.g., mentioning 'flashbacks', 'tension', 'can't stop thinking'), ESCALATE to the 'therapy' agent, even if it started as a simple mood check-in.\n"
    "4. If the user is only stating their current feeling (e.g., 'I feel happy', 'I am sad'), use the 'mood' agent.\n"
    "5. If the user asks for practical, actionable advice about schedules or habits, use the 'routine' agent.\n"
    "6. If there are signs of immediate danger or self-harm, ALWAYS choose 'crisis'.\n\n"
    "**Example of a Correct Escalation:**\n"
    "  Human: I feel anxious today.\n"
    "  AI (Mood Agent): I'm sorry to hear that. What's on your mind?\n"
    "  Human: I'm having flashbacks and feel tense.\n"
    "  **Your Decision for the next turn:** therapy\n\n"
    "Now, analyze the following transcript and provide your one-word decision.\n\n"
    "**Conversation Transcript:**\n"
    "{input}\n\n"
    "**Your one-word decision:**"
)

def _normalize_label(text: str) -> str:
    out = (text or "").strip().lower().replace(".", "")
    return out if out in _ALLOWED else "mood"

_chain = LLMChain(llm=llm_router, prompt=router_prompt, verbose=False)

class _RouterChainAdapter:
    def run(self, user_input: str) -> str:
        if contains_crisis_keywords(user_input):
            return "crisis"
        try:
            out = _chain.run({"input": user_input})
            return _normalize_label(out)
        except Exception as e:
            print(f"Router LLM failed: {e}. Falling back to mood agent.")
            return "mood"

router_chain = _RouterChainAdapter()