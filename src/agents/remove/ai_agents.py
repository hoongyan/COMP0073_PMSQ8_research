from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from src.agents.remove.llm_providers import get_llm, get_structured_llm, SUPPORTED_MODELS
from src.agents.tools import get_tools
from app.model import PoliceResponse
from src.database.remove.database import SessionLocal
from src.models.data_model import Agent, PoliceConfig,VictimConfig, Conversations, MessageDetails, AgentType, SenderType, ConversationType, ScamReportData
from src.agents.utils import get_next_row_index, read_csv_row
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict
from datetime import datetime
import logging
import json
import re
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
# import csv 


# Configure logging
logging.basicConfig(
    filename="errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Store agent instances and their memory by agent_id
AGENT_REGISTRY = {}
MEMORY_REGISTRY = {}
        


def create_police_chatbot(
    agent_type: str,
    agent_name: str,
    is_rag: bool,
    llm_provider: str = "Ollama",
    model: str = "llama3.2",
    prompt: str = "",
    allow_search: bool = False
) -> dict:
    """Instantiate a police chatbot with structured output and save it to the database."""
    logging.debug(f"Creating police chatbot with agent_type: {agent_type}")
    
    if llm_provider not in SUPPORTED_MODELS:
        return {"error": f"Invalid llm_provider. Must be one of {list(SUPPORTED_MODELS.keys())}"}
    if model not in SUPPORTED_MODELS[llm_provider]:
        return {"error": f"Invalid model. Must be one of {SUPPORTED_MODELS[llm_provider]}"}
    if agent_type != AgentType.police.value:
        logging.error(f"Invalid agent_type: {agent_type}, expected: {AgentType.police.value}")
        return {"error": f"Agent type must be '{AgentType.police.value}'"}
    if not prompt.strip():
        return {"error": "Prompt cannot be empty"}

    # Combine user prompt with a note about structured output
    combined_prompt = f"{prompt.strip()}\n\nNote: Responses are structured as JSON."

    # Create an LLM instance with structured output
    try:
        llm = get_structured_llm(provider=llm_provider, model=model, structured_model=PoliceResponse)
    except ValueError as e:
        return {"error": str(e)}

    db = SessionLocal()
    try:
        agent = Agent(
            agent_type=AgentType.police,
            agent_name=agent_name,
            llm_provider=llm_provider,
            model=model,
            is_rag=is_rag
        )
        db.add(agent)
        db.flush()

        police_config = PoliceConfig(id=agent.id, prompt=combined_prompt)
        db.add(police_config)
        db.commit()

        tools = get_tools(allow_search)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            max_turns=5
        )
        agent_instance = create_react_agent(model=llm, tools=tools)
        AGENT_REGISTRY[agent.id] = agent_instance
        MEMORY_REGISTRY[agent.id] = memory

        return {
            "message": f"Police chatbot '{agent_name}' created successfully",
            "agent_id": agent.id,
            "config": {
                "agent_type": agent_type,
                "agent_name": agent_name,
                "llm_provider": llm_provider,
                "model": model,
                "is_rag": is_rag,
                "prompt": combined_prompt,
                "allow_search": allow_search
            }
        }
    except IntegrityError as e:
        db.rollback()
        return {"error": f"Failed to create chatbot: Agent name '{agent_name}' already exists"}
    except Exception as e:
        db.rollback()
        return {"error": f"Failed to create chatbot: {str(e)}"}
    finally:
        db.close()

def create_victim_chatbot(
    agent_type: str,
    agent_name: str,
    is_rag: bool,
    llm_provider: str = "Ollama",
    model: str = "llama3.2",
    prompt: str = "",
    allow_search: bool = False
) -> dict:
    
    """Instantiate a victim chatbot and save it to the database."""
    
    if llm_provider not in SUPPORTED_MODELS:
        return {"error": f"Invalid llm_provider. Must be one of {list(SUPPORTED_MODELS.keys())}"}
    if model not in SUPPORTED_MODELS[llm_provider]:
        return {"error": f"Invalid model. Must be one of {SUPPORTED_MODELS[llm_provider]}"}
    if agent_type != AgentType.victim.value:
        return {"error": f"Agent type must be '{AgentType.victim.value}'"}
    if not prompt.strip():
        return {"error": "Prompt cannot be empty"}
    
    csv_file_path = "src/data/scam_report_data/dataset/synthetic_scam_dataset.csv"
    row_tracker_file_path = "row_tracker.txt"

    # Get the current row index
    row_index = get_next_row_index(csv_file_path, row_tracker_file_path)
    
    # Read the specified row
    row_result = read_csv_row(csv_file_path, row_index)
    if "error" in row_result:
        return row_result
    
    # Convert row to JSON and append to prompt
    row_json = json.dumps(row_result["data"], ensure_ascii=False)
    combined_prompt = f"{prompt}\n\nThe victim and scam details are appended:\n{row_json}"

    db = SessionLocal()
    
    try:
        agent = Agent(
            agent_type=AgentType.victim,
            agent_name=agent_name,
            llm_provider=llm_provider,
            model=model,
            is_rag=is_rag
        )
        db.add(agent)
        db.flush()

        victim_config = VictimConfig(id=agent.id, prompt=combined_prompt)
        db.add(victim_config)
        db.commit()

        llm = get_llm(llm_provider, model)
        tools = get_tools(allow_search)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            max_turns=5
        )
        agent_instance = create_react_agent(model=llm, tools=tools)
        AGENT_REGISTRY[agent.id] = agent_instance
        MEMORY_REGISTRY[agent.id] = memory

        return {
            "message": f"Victim chatbot '{agent_name}' created successfully",
            "agent_id": agent.id,
            "config": {
                "agent_type": agent_type,
                "agent_name": agent_name,
                "llm_provider": llm_provider,
                "model": model,
                "is_rag": is_rag,
                "prompt": prompt,
                "allow_search": allow_search
            }
        }

    except IntegrityError as e:
        db.rollback()
        return {"error": f"Failed to create chatbot: Agent name '{agent_name}' already exists"}
    except Exception as e:
        db.rollback()
        return {"error": f"Failed to create chatbot: {str(e)}"}
    finally:
        db.close()
        
        
def get_response_from_ai_agent(agent_id: int, query: str, user_id: Optional[int] = None, conversation_history: list = None) -> dict:
    
    """Get a response from a police chatbot for a non-autonomous conversation with a victim user."""
    
    db = SessionLocal()
    try:
        logging.debug(f"Processing request for agent_id: {agent_id}, query: {query}, user_id: {user_id}")
        
        if not query.strip():
            logging.error("Empty query provided")
            return {"error": "Query cannot be empty"}

        agent_config = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent_config:
            logging.error(f"Agent not found for agent_id: {agent_id}")
            return {"error": "Agent configuration not found"}

        if agent_config.agent_type.value != AgentType.police.value:
            logging.error(f"Invalid agent_type: {agent_config.agent_type.value} for agent_id: {agent_id}")
            return {"error": "Non-autonomous chats require a police agent"} 

        config = db.query(PoliceConfig).filter(PoliceConfig.id == agent_id).first()
        if not config:
            logging.error(f"PoliceConfig not found for agent_id: {agent_id}")
            return {"error": "Police chatbot configuration not found"}
        
        if agent_config.llm_provider not in SUPPORTED_MODELS or agent_config.model not in SUPPORTED_MODELS[agent_config.llm_provider]:
            logging.error(f"Invalid model: {agent_config.model} for provider: {agent_config.llm_provider}")
            return {"error": f"Invalid model: {agent_config.model} for provider: {agent_config.llm_provider}"}

        if agent_id not in AGENT_REGISTRY:
            logging.debug(f"Creating new agent instance for agent_id: {agent_id}")
            llm = get_llm(agent_config.llm_provider, agent_config.model)
            tools = get_tools(agent_config.is_rag)
            MEMORY_REGISTRY[agent_id] = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                max_turns=5
            )
            AGENT_REGISTRY[agent_id] = create_react_agent(model=llm, tools=tools)

        agent = AGENT_REGISTRY[agent_id]
        memory = MEMORY_REGISTRY[agent_id]

        # Load conversation history - for?
        if not conversation_history:
            logging.debug(f"Loading conversation history from database for agent_id: {agent_id}")
            conversation = db.query(Conversations).filter_by(title=f"Victim-Police Chat {agent_id}-{user_id or 'Anonymous'}").first()
            if conversation:
                db_messages = db.query(MessageDetails).filter_by(conversation_id=conversation.id).order_by(MessageDetails.id).all()
                conversation_history = [
                    {"role": msg.sender_type.value, "content": msg.content}
                    for msg in db_messages
                ]
            else:
                conversation_history = []

        # Validate conversation history roles
        valid_roles = {SenderType.victim.value, SenderType.police.value}
        for msg in conversation_history:
            if msg["role"] not in valid_roles:
                logging.error(f"Invalid role in conversation history: {msg['role']}")
                return {"error": f"Invalid role in conversation history: {msg['role']}"}

        memory.chat_memory.clear()
        for msg in conversation_history:
            if msg["role"] == SenderType.victim.value:
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == SenderType.police.value:
                memory.chat_memory.add_ai_message(msg["content"])

        state = {
            "messages": [
                SystemMessage(content=config.prompt),
                *(HumanMessage(content=msg["content"]) if msg["role"] == SenderType.victim.value else AIMessage(content=msg["content"])
                  for msg in conversation_history),
                HumanMessage(content=query)
            ]
        }

        logging.debug(f"Invoking agent for query: {query}")
        response = agent.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
        ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response. Please try again."

        conversation = db.query(Conversations).filter_by(title=f"Victim-Police Chat {agent_id}-{user_id or 'anonymous'}").first()
        if not conversation:
            conversation = Conversations(
                title=f"Non-Autonomous Chat {agent_id}-{user_id or 'anonymous'}",
                description=f"Non-autonomous conversation with {agent_config.agent_name} ({agent_config.llm_provider}, {agent_config.model})",
                conversation_type=ConversationType.non_autonomous
            )
            db.add(conversation)
            db.flush()

        db.add(MessageDetails(
            conversation_id=conversation.id,
            content=query,
            sender_type=SenderType.victim,
            user_id=user_id,
            agent_id=None
        ))
        db.add(MessageDetails(
            conversation_id=conversation.id,
            content=ai_response,
            sender_type=SenderType.police,
            user_id=None,
            agent_id=agent_id
        ))
        db.commit()

        updated_history = conversation_history + [
            {"role": SenderType.victim.value, "content": query},
            {"role": SenderType.police.value, "content": ai_response}
        ]

        return {
            "response": ai_response,
            "conversation_history": updated_history,
            "conversation_type": ConversationType.non_autonomous.value
        }

    except Exception as e:
        logging.error(f"Error in get_response_from_ai_agent for agent_id {agent_id}: {str(e)}")
        return {"error": f"Failed to get response: {str(e)}"}
    finally:
        db.close()     
                
def get_police_chatbot_config(agent_id: int) -> dict:
    """Retrieve a police chatbot's configuration from the database."""
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        police_config = db.query(PoliceConfig).filter(PoliceConfig.id == agent_id).first()
        
        if not agent or not police_config:
            logging.error(f"Configuration not found for agent_id: {agent_id}")
            return {"error": "Chatbot configuration not found"}
        
        return {
            "agent_id": agent.id,
            "agent_type": agent.agent_type,
            "agent_name": agent.agent_name,
            "llm_provider": agent.llm_provider,
            "model": agent.model,
            "is_rag": agent.is_rag,
            "prompt": police_config.prompt
        }
    except Exception as e:
        logging.error(f"Error retrieving config for agent_id {agent_id}: {str(e)}")
        return {"error": f"Failed to retrieve configuration: {str(e)}"}
    finally:
        db.close()
        
        


# def simulate_autonomous_conversation(
#     police_agent_id: int,
#     victim_agent_id: int,
#     max_turns: int = 10,
#     initial_query: Optional[str] = None
# ) -> Dict:
#     """Simulate an autonomous conversation between a police and victim chatbot."""
#     db = SessionLocal()
#     try:
#         logging.debug(f"Starting autonomous conversation: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")

#         police_agent = db.query(Agent).filter(Agent.id == police_agent_id).first()
#         victim_agent = db.query(Agent).filter(Agent.id == victim_agent_id).first()
#         police_config = db.query(PoliceConfig).filter(PoliceConfig.id == police_agent_id).first()
#         victim_config = db.query(VictimConfig).filter(VictimConfig.id == victim_agent_id).first()

#         if not police_agent:
#             logging.error(f"Police agent not found for ID: {police_agent_id}")
#             return {"error": f"Police agent not found for ID: {police_agent_id}"}
#         if not victim_agent:
#             logging.error(f"Victim agent not found for ID: {victim_agent_id}")
#             return {"error": f"Victim agent not found for ID: {victim_agent_id}"}
#         if not police_config:
#             logging.error(f"PoliceConfig not found for agent ID: {police_agent_id}")
#             return {"error": f"Police chatbot configuration not found for ID: {police_agent_id}"}
#         if not victim_config:
#             logging.error(f"VictimConfig not found for agent ID: {victim_agent_id}")
#             return {"error": f"Victim chatbot configuration not found for ID: {victim_agent_id}"}

#         if police_agent.agent_type != AgentType.police or victim_agent.agent_type != AgentType.victim:
#             logging.error(f"Invalid agent types: police={police_agent.agent_type}, victim={victim_agent.agent_type}")
#             return {"error": f"Invalid agent types: expected police={AgentType.police}, victim={AgentType.victim}"}
        
#         if police_agent.llm_provider not in SUPPORTED_MODELS or police_agent.model not in SUPPORTED_MODELS[police_agent.llm_provider]:
#             return {"error": f"Invalid police agent model: {police_agent.model} for {police_agent.llm_provider}"}
#         if victim_agent.llm_provider not in SUPPORTED_MODELS or victim_agent.model not in SUPPORTED_MODELS[victim_agent.llm_provider]:
#             return {"error": f"Invalid victim agent model: {victim_agent.model} for {victim_agent.llm_provider}"}

#         shared_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="input",
#             output_key="output",
#             max_turns=10
#         )

#         if police_agent_id not in AGENT_REGISTRY:
#             police_llm = get_llm(police_agent.llm_provider, police_agent.model)
#             police_tools = get_tools(police_agent.is_rag)
#             AGENT_REGISTRY[police_agent_id] = create_react_agent(model=police_llm, tools=police_tools)
#             MEMORY_REGISTRY[police_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         if victim_agent_id not in AGENT_REGISTRY:
#             victim_llm = get_llm(victim_agent.llm_provider, victim_agent.model)
#             victim_tools = get_tools(victim_agent.is_rag)
#             AGENT_REGISTRY[victim_agent_id] = create_react_agent(model=victim_llm, tools=victim_tools)
#             MEMORY_REGISTRY[victim_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         police_agent_instance = AGENT_REGISTRY[police_agent_id]
#         victim_agent_instance = AGENT_REGISTRY[victim_agent_id]

#         conversation = Conversations(
#             title=f"Autonomous Conversation {police_agent_id}-{victim_agent_id}",
#             description=f"Autonomous conversation between {police_agent.agent_name} and {victim_agent.agent_name}",
#             conversation_type=ConversationType.autonomous
#         )
#         db.add(conversation)
#         db.flush()

#         conversation_history = []
#         current_query = initial_query or "Hello, this is the police. How can we help?"
#         last_messages = []

#         def get_simulation_response(agent_instance, prompt: str, agent_type: str, query: str, history: list) -> str:
#             try:
#                 logging.debug(f"Invoking {agent_type} agent with query: {query}, agent_instance type: {type(agent_instance)}")
#                 shared_memory.chat_memory.clear()
#                 for msg in history:
#                     if msg["role"] == SenderType.police.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.victim.value else shared_memory.chat_memory.add_ai_message(msg["content"])
#                     elif msg["role"] == SenderType.victim.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.police.value else shared_memory.chat_memory.add_ai_message(msg["content"])

#                 state = {
#                     "messages": [
#                         SystemMessage(content=prompt),
#                         *(HumanMessage(content=msg["content"]) if msg["role"] != agent_type else AIMessage(content=msg["content"])
#                           for msg in history),
#                         HumanMessage(content=query)
#                     ]
#                 }

#                 response = agent_instance.invoke(state)
#                 logging.debug(f"Raw response from {agent_type}: {response}")
                
#                 if isinstance(response, dict) and "messages" in response:
#                     messages = response["messages"]
#                     ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#                     ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."
#                 else:
#                     logging.error(f"Unexpected response format for {agent_type}: {type(response)}")
#                     return "Sorry, I couldn't generate a response."

#                 if not ai_response.strip():
#                     return "I'm not sure how to respond. Can you provide more details?"
#                 return ai_response
#             except Exception as e:
#                 logging.error(f"Error getting {agent_type} response: {str(e)}")
#                 return f"Error occurred: {str(e)}"

#         for turn in range(max_turns):
#             victim_message = get_simulation_response(
#                 victim_agent_instance,
#                 victim_config.prompt,
#                 SenderType.victim.value,
#                 current_query,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.victim.value, "content": victim_message})

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=victim_message,
#                 sender_type=SenderType.victim,
#                 user_id=None,
#                 agent_id=victim_agent_id
#             ))
#             db.commit()
            
#             if "[END_CONVERSATION]" in victim_message:
#                 logging.info("Victim decided to end conversation via explicit intent")
#                 break

#             last_messages.append(victim_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             police_message = get_simulation_response(
#                 police_agent_instance,
#                 police_config.prompt,
#                 SenderType.police.value,
#                 victim_message,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.police.value, "content": police_message})

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=police_message,
#                 sender_type=SenderType.police,
#                 user_id=None,
#                 agent_id=police_agent_id
#             ))
#             db.commit()

#             last_messages.append(police_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             current_query = police_message

#             if "thank you for your cooperation" in police_message.lower():
#                 break

#         return {
#             "status": "Conversation completed",
#             "conversation_id": conversation.id,
#             "conversation_history": conversation_history,
#             "conversation_type": ConversationType.autonomous.value
#         }

#     except Exception as e:
#         db.rollback()
#         logging.error(f"Error in simulate_autonomous_conversation: {str(e)}")
#         return {"error": f"Failed to simulate conversation: {str(e)}"}
#     finally:
#         db.close()





# def simulate_autonomous_conversation(
#     police_agent_id: int,
#     victim_agent_id: int,
#     max_turns: int = 10,
#     initial_query: Optional[str] = None
# ) -> Dict:
#     """Simulate an autonomous conversation with separate extraction LLM."""
#     db = SessionLocal()
#     try:
#         logging.debug(f"Starting autonomous conversation: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")

#         # Validate agents and configs
#         police_agent = db.query(Agent).filter(Agent.id == police_agent_id).first()
#         victim_agent = db.query(Agent).filter(Agent.id == victim_agent_id).first()
#         police_config = db.query(PoliceConfig).filter(PoliceConfig.id == police_agent_id).first()
#         victim_config = db.query(VictimConfig).filter(VictimConfig.id == victim_agent_id).first()

#         if not police_agent or not victim_agent or not police_config or not victim_config:
#             logging.error("Agent or config not found")
#             return {"error": "Agent or configuration not found"}

#         if police_agent.agent_type != AgentType.police or victim_agent.agent_type != AgentType.victim:
#             logging.error(f"Invalid agent types: police={police_agent.agent_type}, victim={victim_agent.agent_type}")
#             return {"error": f"Invalid agent types: expected police={AgentType.police}, victim={AgentType.victim}"}

#         if police_agent.llm_provider not in SUPPORTED_MODELS or police_agent.model not in SUPPORTED_MODELS[police_agent.llm_provider]:
#             return {"error": f"Invalid police agent model: {police_agent.model} for {police_agent.llm_provider}"}
#         if victim_agent.llm_provider not in SUPPORTED_MODELS or victim_agent.model not in SUPPORTED_MODELS[victim_agent.llm_provider]:
#             return {"error": f"Invalid victim agent model: {victim_agent.model} for {victim_agent.llm_provider}"}

#         # Initialize LLMs with the same model and provider
#         conversational_llm = get_llm(provider=police_agent.llm_provider, model=police_agent.model)
#         extraction_llm = get_llm(provider=police_agent.llm_provider, model=police_agent.model)

#         # Initialize shared memory
#         shared_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="input",
#             output_key="output",
#             max_turns=10
#         )

#         # Initialize agent instances
#         if police_agent_id not in AGENT_REGISTRY:
#             police_tools = get_tools(police_agent.is_rag)
#             AGENT_REGISTRY[police_agent_id] = create_react_agent(model=conversational_llm, tools=police_tools)
#             MEMORY_REGISTRY[police_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         if victim_agent_id not in AGENT_REGISTRY:
#             victim_tools = get_tools(victim_agent.is_rag)
#             AGENT_REGISTRY[victim_agent_id] = create_react_agent(model=conversational_llm, tools=victim_tools)
#             MEMORY_REGISTRY[victim_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         police_agent_instance = AGENT_REGISTRY[police_agent_id]
#         victim_agent_instance = AGENT_REGISTRY[victim_agent_id]

#         # Create conversation record
#         conversation = Conversations(
#             title=f"Autonomous Conversation {police_agent_id}-{victim_agent_id}",
#             description=f"Autonomous conversation between {police_agent.agent_name} and {victim_agent.agent_name}",
#             conversation_type=ConversationType.autonomous
#         )
#         db.add(conversation)
#         db.flush()

#         conversation_history = []
#         current_query = initial_query or "Hello, this is the police. How can we help?"
#         last_messages = []
#         report_intent_detected = False

#         def get_simulation_response(agent_instance, prompt: str, agent_type: str, query: str, history: list) -> str:
#             """Get conversational response from an agent."""
#             try:
#                 logging.debug(f"Invoking {agent_type} agent with query: {query}")
#                 shared_memory.chat_memory.clear()
#                 for msg in history:
#                     if msg["role"] == SenderType.police.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.victim.value else shared_memory.chat_memory.add_ai_message(msg["content"])
#                     elif msg["role"] == SenderType.victim.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.police.value else shared_memory.chat_memory.add_ai_message(msg["content"])

#                 state = {
#                     "messages": [
#                         SystemMessage(content=prompt),
#                         *(HumanMessage(content=msg["content"]) if msg["role"] != agent_type else AIMessage(content=msg["content"])
#                           for msg in history),
#                         HumanMessage(content=query)
#                     ]
#                 }

#                 response = agent_instance.invoke(state)
#                 messages = response.get("messages", [])
#                 ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#                 ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."
#                 return ai_response.strip() or "I'm not sure how to respond. Can you provide more details?"
#             except Exception as e:
#                 logging.error(f"Error getting {agent_type} response: {str(e)}")
#                 return f"Error occurred: {str(e)}"

#         def generate_info_form_data(history: list, query: str) -> dict:
#             """Generate info_form_data JSON using the extraction LLM."""
#             conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in history) + f"\nvictim: {query}"
#             victim_messages = " ".join(msg["content"] for msg in history if msg["role"] == SenderType.victim.value) + " " + query
#             return extract_info_form_data(extraction_llm, conversation_text, victim_messages)

#         for turn in range(max_turns):
#             # Get victim response
#             victim_message = get_simulation_response(
#                 victim_agent_instance,
#                 victim_config.prompt,
#                 SenderType.victim.value,
#                 current_query,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.victim.value, "content": victim_message})

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=victim_message,
#                 sender_type=SenderType.victim,
#                 user_id=None,
#                 agent_id=victim_agent_id
#             ))
#             db.commit()

#             # Generate and store info_form_data
#             info_form_data = generate_info_form_data(conversation_history, victim_message)
#             scam_report = db.query(ScamReportData).filter(ScamReportData.conversation_id == conversation.id).first()
#             if scam_report:
#                 scam_report.report_data = info_form_data
#             else:
#                 db.add(ScamReportData(
#                     conversation_id=conversation.id,
#                     report_data=info_form_data
#                 ))
#             db.commit()

#             if "[END_CONVERSATION]" in victim_message:
#                 logging.info("Victim decided to end conversation via explicit intent")
#                 break

#             if "[I am willing to report]" in victim_message:
#                 report_intent_detected = True
#                 logging.info("Victim expressed intent to file a report")

#             last_messages.append(victim_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             # Get police response
#             police_message = get_simulation_response(
#                 police_agent_instance,
#                 police_config.prompt,
#                 SenderType.police.value,
#                 victim_message,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.police.value, "content": police_message})

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=police_message,
#                 sender_type=SenderType.police,
#                 user_id=None,
#                 agent_id=police_agent_id
#             ))
#             db.commit()

#             last_messages.append(police_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             current_query = police_message

#             if "thank you for your cooperation" in police_message.lower():
#                 break

#         return {
#             "status": "Conversation completed",
#             "conversation_id": conversation.id,
#             "conversation_history": conversation_history,
#             "conversation_type": ConversationType.autonomous.value,
#             "info_form_data": info_form_data if report_intent_detected or "thank you for your cooperation" in current_query.lower() else {}
#         }
#     except Exception as e:
#         db.rollback()
#         logging.error(f"Error in simulate_autonomous_conversation: {str(e)}")
#         return {"error": f"Failed to simulate conversation: {str(e)}"}
#     finally:
#         db.close()

# def simulate_autonomous_conversation(
#     police_agent_id: int,
#     victim_agent_id: int,
#     max_turns: int = 10,
#     initial_query: Optional[str] = None
# ) -> Dict:
#     """Simulate an autonomous conversation between a police and victim chatbot, parsing JSON from police responses."""
#     db = SessionLocal()
#     try:
#         logging.debug(f"Starting autonomous conversation: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")

#         # Validate agents and configs
#         police_agent = db.query(Agent).filter(Agent.id == police_agent_id).first()
#         victim_agent = db.query(Agent).filter(Agent.id == victim_agent_id).first()
#         police_config = db.query(PoliceConfig).filter(PoliceConfig.id == police_agent_id).first()
#         victim_config = db.query(VictimConfig).filter(VictimConfig.id == victim_agent_id).first()

#         if not police_agent or not victim_agent or not police_config or not victim_config:
#             logging.error("Agent or config not found")
#             return {"error": "Agent or configuration not found"}

#         if police_agent.agent_type != AgentType.police or victim_agent.agent_type != AgentType.victim:
#             logging.error(f"Invalid agent types: police={police_agent.agent_type}, victim={victim_agent.agent_type}")
#             return {"error": f"Invalid agent types: expected police={AgentType.police}, victim={AgentType.victim}"}

#         if police_agent.llm_provider not in SUPPORTED_MODELS or police_agent.model not in SUPPORTED_MODELS[police_agent.llm_provider]:
#             return {"error": f"Invalid police agent model: {police_agent.model} for {police_agent.llm_provider}"}
#         if victim_agent.llm_provider not in SUPPORTED_MODELS or victim_agent.model not in SUPPORTED_MODELS[victim_agent.llm_provider]:
#             return {"error": f"Invalid victim agent model: {victim_agent.model} for {victim_agent.llm_provider}"}

#         # Initialize shared memory
#         shared_memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="input",
#             output_key="output",
#             max_turns=10
#         )

#         # Initialize agent instances
#         if police_agent_id not in AGENT_REGISTRY:
#             llm = get_llm(police_agent.llm_provider, police_agent.model)
#             tools = get_tools(police_agent.is_rag)
#             AGENT_REGISTRY[police_agent_id] = create_react_agent(model=llm, tools=tools)
#             MEMORY_REGISTRY[police_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         if victim_agent_id not in AGENT_REGISTRY:
#             llm = get_llm(victim_agent.llm_provider, victim_agent.model)
#             tools = get_tools(victim_agent.is_rag)
#             AGENT_REGISTRY[victim_agent_id] = create_react_agent(model=llm, tools=tools)
#             MEMORY_REGISTRY[victim_agent_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=5
#             )

#         police_agent_instance = AGENT_REGISTRY[police_agent_id]
#         victim_agent_instance = AGENT_REGISTRY[victim_agent_id]

#         # Create conversation record
#         conversation = Conversations(
#             title=f"Autonomous Conversation {police_agent_id}-{victim_agent_id}",
#             description=f"Autonomous conversation between {police_agent.agent_name} and {victim_agent.agent_name}",
#             conversation_type=ConversationType.autonomous
#         )
#         db.add(conversation)
#         db.flush()

#         conversation_history = []
#         info_form_data_history = []
#         current_query = initial_query or "Hello, this is the police. How can we help?"
#         last_messages = []

#         def parse_json_from_response(response: str) -> tuple[str, dict]:
#             """Parse JSON from police response, return cleaned response and JSON data."""
#             try:
#                 # Match [SCAM_DETAILS]{json} at the end, allowing for whitespace
#                 json_match = re.search(r'\[SCAM_DETAILS\]\s*(\{.*?\})\s*$', response, re.DOTALL)
#                 if json_match:
#                     json_str = json_match.group(1)
#                     json_data = json.loads(json_str)
#                     # Validate JSON structure
#                     expected_keys = set(DEFAULT_INFO_FORM_DATA.keys())
#                     if not isinstance(json_data, dict) or not expected_keys.issubset(json_data.keys()):
#                         logging.warning(f"Invalid JSON structure: {json_data}")
#                         return response, DEFAULT_INFO_FORM_DATA.copy()
#                     # Remove [SCAM_DETAILS]{json} from response
#                     cleaned_response = re.sub(r'\[SCAM_DETAILS\]\s*\{.*?\}\s*$', '', response, flags=re.DOTALL).strip()
#                     return cleaned_response, json_data
#                 # Fallback: assume response is the conversation text
#                 return response.strip(), DEFAULT_INFO_FORM_DATA.copy()
#             except json.JSONDecodeError as e:
#                 logging.warning(f"Failed to parse JSON from response: {e}")
#                 return response.strip(), DEFAULT_INFO_FORM_DATA.copy()
#             except Exception as e:
#                 logging.error(f"Error parsing JSON: {str(e)}")
#                 return response.strip(), DEFAULT_INFO_FORM_DATA.copy()

#         def get_simulation_response(agent_instance, prompt: str, agent_type: str, query: str, history: list) -> tuple[str, dict]:
#             """Get conversational response from an agent, parse JSON if police."""
#             try:
#                 logging.debug(f"Invoking {agent_type} agent with query: {query}")
#                 shared_memory.chat_memory.clear()
#                 for msg in history:
#                     if msg["role"] == SenderType.police.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.victim.value else shared_memory.chat_memory.add_ai_message(msg["content"])
#                     elif msg["role"] == SenderType.victim.value:
#                         shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.police.value else shared_memory.chat_memory.add_ai_message(msg["content"])

#                 state = {
#                     "messages": [
#                         SystemMessage(content=prompt),
#                         *(HumanMessage(content=msg["content"]) if msg["role"] != agent_type else AIMessage(content=msg["content"])
#                           for msg in history),
#                         HumanMessage(content=query)
#                     ]
#                 }

#                 response = agent_instance.invoke(state)
#                 messages = response.get("messages", [])
#                 ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#                 ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."

#                 if agent_type == SenderType.police.value:
#                     cleaned_response, json_data = parse_json_from_response(ai_response)
#                     return cleaned_response.strip() or "I'm not sure how to respond. Can you provide more details?", json_data
#                 return ai_response.strip() or "I'm not sure how to respond. Can you provide more details?", {}
#             except Exception as e:
#                 logging.error(f"Error getting {agent_type} response: {str(e)}")
#                 return f"Error occurred: {str(e)}", {}

#         for turn in range(max_turns):
#             # Get victim response
#             victim_message, _ = get_simulation_response(
#                 victim_agent_instance,
#                 victim_config.prompt,
#                 SenderType.victim.value,
#                 current_query,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.victim.value, "content": victim_message})

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=victim_message,
#                 sender_type=SenderType.victim,
#                 user_id=None,
#                 agent_id=victim_agent_id
#             ))
#             db.commit()

#             if "[END_CONVERSATION]" in victim_message:
#                 logging.info("Victim decided to end conversation via explicit intent")
#                 break

#             last_messages.append(victim_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             # Get police response and parse JSON
#             police_message, json_data = get_simulation_response(
#                 police_agent_instance,
#                 police_config.prompt,
#                 SenderType.police.value,
#                 victim_message,
#                 conversation_history
#             )
#             conversation_history.append({"role": SenderType.police.value, "content": police_message})

#             if json_data:
#                 info_form_data_history.append(json_data)

#             db.add(MessageDetails(
#                 conversation_id=conversation.id,
#                 content=police_message,
#                 sender_type=SenderType.police,
#                 user_id=None,
#                 agent_id=police_agent_id
#             ))
#             db.commit()

#             last_messages.append(police_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             current_query = police_message

#             if "thank you for your cooperation" in police_message.lower():
#                 break

#         # Return the last JSON data if available, or default
#         final_info_form_data = info_form_data_history[-1] if info_form_data_history else DEFAULT_INFO_FORM_DATA.copy()

#         return {
#             "status": "Conversation completed",
#             "conversation_id": conversation.id,
#             "conversation_history": conversation_history,
#             "conversation_type": ConversationType.autonomous.value,
#             "info_form_data_history": info_form_data_history,
#             "info_form_data": final_info_form_data
#         }
#     except Exception as e:
#         db.rollback()
#         logging.error(f"Error in simulate_autonomous_conversation: {str(e)}")
#         return {"error": f"Failed to simulate conversation: {str(e)}"}
#     finally:
#         db.close()      
        
        
        




# Default JSON template for info_form_data
DEFAULT_INFO_FORM_DATA = {
    "firstName": "",
    "lastName": "",
    "telNo": "",
    "address": "",
    "occupation": "",
    "age": "",
    "incidentDate": "",
    "reportDate": datetime.now().strftime("%Y-%m-%d"),
    "location": "",
    "crimeType": "unknown",
    "approachPlatform": "",
    "communicationPlatform": "",
    "bank": "",
    "bankNo": "",
    "contactInfo": "",
    "description": "",
    "summary": ""
}

def get_simulation_response(agent_instance, prompt: str, agent_type: str, query: str, history: list) -> tuple[str, dict]:
    """Get conversational response from an agent, with JSON parsing for police."""
    try:
        logging.debug(f"Invoking {agent_type} agent with query: {query}")
        shared_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            max_turns=10
        )
        shared_memory.chat_memory.clear()
        for msg in history:
            if msg["role"] == SenderType.police.value:
                shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.victim.value else shared_memory.chat_memory.add_ai_message(msg["content"])
            elif msg["role"] == SenderType.victim.value:
                shared_memory.chat_memory.add_user_message(msg["content"]) if agent_type == SenderType.police.value else shared_memory.chat_memory.add_ai_message(msg["content"])

        state = {
            "messages": [
                SystemMessage(content=prompt),
                *(HumanMessage(content=msg["content"]) if msg["role"] != agent_type else AIMessage(content=msg["content"])
                  for msg in history),
                HumanMessage(content=query)
            ]
        }

        response = agent_instance.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
        ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."

        if agent_type == SenderType.police.value:
            try:
                parsed_response = json.loads(ai_response)
                police_response = PoliceResponse(**parsed_response)
                conversational_response = police_response.conversational_response.strip() or "I'm not sure how to respond. Can you provide more details?"
                json_data = police_response.dict(exclude={"conversational_response"})
                return conversational_response, json_data
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse police response as JSON: {str(e)}")
                return ai_response.strip() or "I'm not sure how to respond. Can you provide more details?", DEFAULT_INFO_FORM_DATA.copy()
        else:
            return ai_response.strip() or "I'm not sure how to respond. Can you provide more details?", {}
    except Exception as e:
        logging.error(f"Error getting {agent_type} response: {str(e)}")
        return f"Error occurred: {str(e)}", {}

def simulate_autonomous_conversation(
    police_agent_id: int,
    victim_agent_id: int,
    max_turns: int = 10,
    initial_query: Optional[str] = None
) -> Dict:
    """Simulate an autonomous conversation between a police and victim chatbot, parsing JSON from police responses."""
    db = SessionLocal()
    try:
        logging.debug(f"Starting autonomous conversation: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")

        # Validate agents and configs
        police_agent = db.query(Agent).filter(Agent.id == police_agent_id).first()
        victim_agent = db.query(Agent).filter(Agent.id == victim_agent_id).first()
        police_config = db.query(PoliceConfig).filter(PoliceConfig.id == police_agent_id).first()
        victim_config = db.query(VictimConfig).filter(VictimConfig.id == victim_agent_id).first()

        if not police_agent or not victim_agent or not police_config or not victim_config:
            logging.error("Agent or config not found")
            return {"error": "Agent or configuration not found"}

        if police_agent.agent_type != AgentType.police or victim_agent.agent_type != AgentType.victim:
            logging.error(f"Invalid agent types: police={police_agent.agent_type}, victim={victim_agent.agent_type}")
            return {"error": f"Invalid agent types: expected police={AgentType.police}, victim={AgentType.victim}"}

        if police_agent.llm_provider not in SUPPORTED_MODELS or police_agent.model not in SUPPORTED_MODELS[police_agent.llm_provider]:
            return {"error": f"Invalid police agent model: {police_agent.model} for {police_agent.llm_provider}"}
        if victim_agent.llm_provider not in SUPPORTED_MODELS or victim_agent.model not in SUPPORTED_MODELS[victim_agent.llm_provider]:
            return {"error": f"Invalid victim agent model: {victim_agent.model} for {victim_agent.llm_provider}"}

        # Check if agents are initialized
        if police_agent_id not in AGENT_REGISTRY or victim_agent_id not in AGENT_REGISTRY:
            logging.error(f"Agent not initialized: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")
            return {"error": "Agents must be initialized via create_police_chatbot or create_victim_chatbot"}

        police_agent_instance = AGENT_REGISTRY[police_agent_id]
        victim_agent_instance = AGENT_REGISTRY[victim_agent_id]

        # Create conversation record
        conversation = Conversations(
            title=f"Autonomous Conversation {police_agent_id}-{victim_agent_id}",
            description=f"Autonomous conversation between {police_agent.agent_name} and {victim_agent.agent_name}",
            conversation_type=ConversationType.autonomous
        )
        db.add(conversation)
        db.flush()

        conversation_history = []
        info_form_data_history = []
        current_query = initial_query or "Hello, this is the police. How can we help?"
        last_messages = []

        for turn in range(max_turns):
            # Get victim response
            victim_message, _ = get_simulation_response(
                victim_agent_instance,
                victim_config.prompt,
                SenderType.victim.value,
                current_query,
                conversation_history
            )
            conversation_history.append({"role": SenderType.victim.value, "content": victim_message})

            db.add(MessageDetails(
                conversation_id=conversation.id,
                content=victim_message,
                sender_type=SenderType.victim,
                user_id=None,
                agent_id=victim_agent_id
            ))
            db.commit()

            if "[END_CONVERSATION]" in victim_message:
                logging.info("Victim decided to end conversation via explicit intent")
                break

            last_messages.append(victim_message)
            if len(last_messages) > 6:
                last_messages.pop(0)
                if len(set(last_messages)) <= 4:
                    logging.warning(f"Detected potential repetition: {last_messages}")
                    break

            # Get police response and parse JSON
            police_message, json_data = get_simulation_response(
                police_agent_instance,
                police_config.prompt,
                SenderType.police.value,
                victim_message,
                conversation_history
            )
            conversation_history.append({"role": SenderType.police.value, "content": police_message})

            if json_data:
                info_form_data_history.append(json_data)

            db.add(MessageDetails(
                conversation_id=conversation.id,
                content=police_message,
                sender_type=SenderType.police,
                user_id=None,
                agent_id=police_agent_id
            ))
            db.commit()

            last_messages.append(police_message)
            if len(last_messages) > 6:
                last_messages.pop(0)
                if len(set(last_messages)) <= 4:
                    logging.warning(f"Detected potential repetition: {last_messages}")
                    break

            current_query = police_message

            if "thank you for your cooperation" in police_message.lower():
                break

        final_info_form_data = info_form_data_history[-1] if info_form_data_history else DEFAULT_INFO_FORM_DATA.copy()

        return {
            "status": "Conversation completed",
            "conversation_id": conversation.id,
            "conversation_history": conversation_history,
            "conversation_type": ConversationType.autonomous.value,
            "info_form_data": final_info_form_data,
            "info_form_data_history": info_form_data_history
        }
    except Exception as e:
        db.rollback()
        logging.error(f"Error in simulate_autonomous_conversation: {str(e)}")
        return {"error": f"Failed to simulate conversation: {str(e)}"}
    finally:
        db.close()











def get_info_form_data(conversation_id: int) -> dict:
    """Retrieve info_form_data for a specific conversation."""
    db = SessionLocal()
    try:
        logging.debug(f"Fetching info_form_data for conversation_id: {conversation_id}")
        scam_report = db.query(ScamReportData).filter(ScamReportData.conversation_id == conversation_id).first()
        if scam_report:
            return {"info_form_data": scam_report.report_data}

        conversation = db.query(Conversations).filter(Conversations.id == conversation_id).first()
        if not conversation:
            logging.error(f"Conversation not found for ID: {conversation_id}")
            return {"error": f"Conversation not found for ID: {conversation_id}"}

        messages = db.query(MessageDetails).filter_by(conversation_id=conversation.id).order_by(MessageDetails.id).all()
        conversation_history = [{"role": msg.sender_type.value, "content": msg.content} for msg in messages]
        police_agent_id = next((msg.agent_id for msg in messages if msg.sender_type == SenderType.police), None)

        if not police_agent_id:
            logging.error(f"No police agent found for conversation_id: {conversation_id}")
            return {"error": "Police agent not found"}

        police_agent = db.query(Agent).filter(Agent.id == police_agent_id).first()
        if not police_agent:
            logging.error(f"Police agent not found for ID: {police_agent_id}")
            return {"error": f"Police agent not found for ID: {police_agent_id}"}

        # Use extraction LLM with the same model and provider
        extraction_llm = get_llm(provider=police_agent.llm_provider, model=police_agent.model)
        conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation_history)
        victim_messages = " ".join(msg["content"] for msg in conversation_history if msg["role"] == SenderType.victim.value)

        info_form_data = extract_info_form_data(extraction_llm, conversation_text, victim_messages)
        db.add(ScamReportData(
            conversation_id=conversation.id,
            report_data=info_form_data
        ))
        db.commit()
        return {"info_form_data": info_form_data}
    except Exception as e:
        logging.error(f"Error retrieving info_form_data for conversation_id {conversation_id}: {str(e)}")
        return {"error": f"Failed to retrieve info_form_data: {str(e)}"}
    finally:
        db.close()




                  
def get_conversation_history() -> dict:
    """Retrieve all conversations for tracking of chatbot simulations."""
    db = SessionLocal()
    try:
        logging.debug("Fetching all conversations")
        conversations = db.query(Conversations).all()
        result = []
        for conv in conversations:
            messages = db.query(MessageDetails).filter_by(conversation_id=conv.id).order_by(MessageDetails.id).all()
            result.append({
                "conversation_id": conv.id,
                "title": conv.title,
                "description": conv.description,
                "conversation_type": conv.conversation_type.value,
                "messages": [
                    {
                        "id": msg.id,
                        "content": msg.content,
                        "sender_type": msg.sender_type.value,
                        "user_id": msg.user_id,
                        "agent_id": msg.agent_id,
                    } for msg in messages
                ]
            })
        logging.debug(f"Retrieved {len(result)} conversations")
        return {"conversations": result}
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {str(e)}")
        return {"error": f"Failed to retrieve conversation history: {str(e)}"}
    finally:
        db.close()
        
def delete_conversation(conversation_id: int) -> dict:
    """Delete a conversation and its associated messages from the database."""
    db = SessionLocal()
    try:
        logging.debug(f"Deleting conversation with ID: {conversation_id}")
        conversation = db.query(Conversations).filter(Conversations.id == conversation_id).first()
        if not conversation:
            logging.error(f"Conversation not found for ID: {conversation_id}")
            return {"error": f"Conversation not found for ID: {conversation_id}"}

        # Delete associated messages (cascade delete)
        db.delete(conversation)
        db.commit()
        logging.debug(f"Conversation {conversation_id} deleted successfully")
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except Exception as e:
        db.rollback()
        logging.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        return {"error": f"Failed to delete conversation: {str(e)}"}
    finally:
        db.close()