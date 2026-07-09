# eliza_agent.py

import os
import logging
import asyncio
from typing import Any, Dict, Union

from eliza.agent import Agent
from eliza.tools import tool

# 🔌 IMPORT YOUR BRAIN ENTRY POINT
# We use Case C/D integration to handle the async class initialization
from trading_assistant_nlp_handler import TradingAssistantNLPHandler

# --- CONFIG & LOGGER SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ElizaRouterAgent")


# =========================
# 🔧 TOOL: PURE PASS-THROUGH
# =========================

@tool
def process_user_query(query: str, user_id: str = "default_ui_user") -> Dict[str, Any]:
    """
    Pass user input directly to the Trading Assistant AI brain without any modification.
    """
    logger.info(f"Routing query for user {user_id}: '{query}'")
    
    async def _run_handler() -> Dict[str, Any]:
        try:
            # 1. Instantiate the handler
            handler = TradingAssistantNLPHandler(user_id=user_id)
            
            # 2. Initialize async components (DB history, symbols cache)
            await handler.initialize()
            
            # 3. Route to the brain's main processing method
            # NOTE: Assuming 'process_query' or 'run_full_analysis' is your top-level 
            # orchestration method inside TradingAssistantNLPHandler. Adjust name if needed.
            if hasattr(handler, 'process_query'):
                if asyncio.iscoroutinefunction(handler.process_query):
                    result = await handler.process_query(query)
                else:
                    # In case the method is synchronous but running in async loop
                    result = await asyncio.to_thread(handler.process_query, query)
                return result
            else:
                raise AttributeError("TradingAssistantNLPHandler is missing the primary 'process_query' method.")
                
        except Exception as e:
            logger.error(f"Internal Handler failure: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed",
                "message": "The trading assistant brain encountered a critical error."
            }

    # Safely execute the async pipeline from a sync tool wrapper
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Handle cases where the event loop is already running (e.g., inside FastAPI)
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_run_handler())
    else:
        return loop.run_until_complete(_run_handler())


# =========================
# 🧠 AGENT CONFIG & SYSTEM PROMPT
# =========================

# The routing agent MUST be kept lightweight and stateless.
agent = Agent(
    name="CryptSignalRouter",
    description="Stateless routing proxy for the Trading Assistant engine.",
    tools=[process_user_query],
    model="qwen" # Ensure this model string matches your exact open-source/API model ID
)

agent.system_prompt = """
You are a strict API routing proxy.

Your ONLY directive is:
1. Receive the user's input.
2. Pass it EXACTLY and IMMEDIATELY to the tool: `process_user_query`.
3. Return the exact JSON/text output from the tool.

STRICT PROTOCOLS (ZERO TOLERANCE):
- DO NOT analyze the user's intent.
- DO NOT answer the user's crypto questions directly.
- DO NOT modify, summarize, or truncate the tool's response.
- DO NOT add conversational filler (e.g., "Here is the response:", "Based on the tool...").
- DO NOT reformat the output.

Your existence is purely to bridge the UI to the `process_user_query` tool. Execute the tool and output the raw result.
"""


# =========================
# 🚀 RUN FUNCTION (SERVER ENTRY)
# =========================

def run_agent(query: str, user_id: str = "default_ui_user") -> Union[Dict[str, Any], str]:
    """
    Entry point for server.py. 
    Maintains a strict boundary between UI and the NLP Handler.
    
    Args:
        query (str): The raw text input from the frontend.
        user_id (str): Identifier for DB history tracking.
        
    Returns:
        Dict/Str: The exact output from the trading_assistant_nlp_handler.
    """
    try:
        logger.info("Executing Eliza Agent pass-through...")
        
        # We can pass context elements if Eliza supports contextual tool arguments,
        # otherwise it will rely on the default in the tool signature.
        response = agent.run(query)

        # Ensure we return a structured format even if the tool/agent returned a raw string
        if isinstance(response, str):
            try:
                # Attempt to parse if the agent stringified a JSON payload
                import json
                parsed = json.loads(response)
                return parsed
            except ValueError:
                return {"response": response, "status": "success"}

        return response

    except Exception as e:
        logger.critical(f"Eliza Agent framework failed to execute: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "system_failure",
            "message": "The routing layer failed to connect to the assistant."
        }