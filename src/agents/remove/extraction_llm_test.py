from langchain_core.messages import SystemMessage, AIMessage
from src.agents.remove.llm_providers import get_llm
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

extraction_llm = get_llm(provider="Ollama", model="llama3.2")
conversation_text = """
police: Hello, this is the police. How can we help?
victim: I was scammed on WhatsApp. Someone sold me a fake product.
police: Can you provide more details?
victim: My name is John Doe, and it happened yesterday.
"""

# Default JSON template with all required fields
default_json_template = {
    "firstName": "",
    "lastName": "",
    "telNo": "",
    "address": "",
    "occupation": "",
    "age": "",
    "incidentDate": "",
    "reportDate": datetime.now().strftime('%Y-%m-%d'),
    "location": "",
    "crimeType": "",
    "approachPlatform": "",
    "communicationPlatform": "",
    "bank": "",
    "bankNo": "",
    "contactInfo": "",
    "description": "",
    "summary": ""
}

# Simplified prompt with clear JSON instructions
prompt = f"""
You are an expert at extracting structured information from conversations. Analyze the following conversation and extract details into a JSON object with these fields: firstName, lastName, telNo, address, occupation, age, incidentDate (YYYY-MM-DD), reportDate ({datetime.now().strftime('%Y-%m-%d')}), location, crimeType, approachPlatform, communicationPlatform, bank, bankNo, contactInfo, description, summary (max 200 chars). Use empty strings for missing fields. Return ONLY the JSON object as a string, with no extra text or code blocks.

Example: {json.dumps(default_json_template)}

Conversation:
{conversation_text}
"""

# Invoke LLM with messages
messages = [SystemMessage(content=prompt)]
response = extraction_llm.invoke(messages)
logging.debug(f"Raw response: {response}")

# Extract response content
if isinstance(response, str):
    response_content = response
elif isinstance(response, AIMessage):
    response_content = response.content
else:
    response_content = str(response)

# Try to parse response as JSON, fallback to template if it fails
try:
    json_data = json.loads(response_content)
    # Merge with template to ensure all fields are present
    output_json = default_json_template.copy()
    output_json.update(json_data)
except json.JSONDecodeError as e:
    logging.error(f"JSON parsing error: {e}")
    logging.error(f"Raw response content: {response_content}")
    # Fallback: Manually extract key information if possible
    output_json = default_json_template.copy()
    if "John Doe" in conversation_text:
        output_json.update({
            "firstName": "John",
            "lastName": "Doe",
            "incidentDate": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            "crimeType": "e-commerce scam",
            "approachPlatform": "WhatsApp",
            "communicationPlatform": "WhatsApp",
            "description": "I was scammed on WhatsApp. Someone sold me a fake product. My name is John Doe, and it happened yesterday.",
            "summary": "John Doe was scammed on WhatsApp with a fake product."
        })

print("Extracted JSON:", json.dumps(output_json, indent=2))




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

def extract_info_form_data(extraction_llm, conversation_text: str, victim_messages: str) -> dict:
    """Helper function to extract info_form_data from conversation text."""
    try:
        prompt = f"""
        You are an expert at extracting structured information from conversations. Analyze the following conversation and extract details into a JSON object with these fields: firstName, lastName, telNo, address, occupation, age, incidentDate (YYYY-MM-DD), reportDate ({datetime.now().strftime('%Y-%m-%d')}), location, crimeType, approachPlatform, communicationPlatform, bank, bankNo, contactInfo, description, summary (max 200 chars). Use empty strings for missing fields. Return ONLY the JSON object as a string, with no extra text or code blocks.

        Example: {json.dumps(DEFAULT_INFO_FORM_DATA)}

        Conversation:
        {conversation_text}
        """
        messages = [SystemMessage(content=prompt)]
        response = extraction_llm.invoke(messages)
        logging.debug(f"Raw extraction_llm response: {response}")

        # Extract response content
        if isinstance(response, str):
            response_content = response
        elif isinstance(response, AIMessage):
            response_content = response.content
        elif isinstance(response, dict) and "messages" in response:
            messages = response.get("messages", [])
            ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
            response_content = ai_messages[-1] if ai_messages else ""
        else:
            response_content = str(response)
            logging.warning(f"Unexpected response format: {type(response)}")

        # Try to parse response as JSON
        try:
            json_data = json.loads(response_content)
            # Merge with default template to ensure all fields
            output_json = DEFAULT_INFO_FORM_DATA.copy()
            output_json.update(json_data)
            # Update description and summary
            output_json["description"] = victim_messages
            output_json["summary"] = victim_messages[:200]
            logging.debug(f"Successfully parsed JSON: {output_json}")
            return output_json
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            logging.error(f"Raw response content: {response_content}")
            # Fallback: Manually extract key information
            output_json = DEFAULT_INFO_FORM_DATA.copy()
            output_json["description"] = victim_messages
            output_json["summary"] = victim_messages[:200]

            # Basic extraction from conversation text
            description = victim_messages.lower()
            if "scammed" in description and "whatsapp" in description:
                output_json["approachPlatform"] = "WhatsApp"
                output_json["communicationPlatform"] = "WhatsApp"
                output_json["crimeType"] = "e-commerce scam"
            if "yesterday" in description:
                output_json["incidentDate"] = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            name_match = re.search(r"(?:my name is|I am)\s+([A-Za-z]+)\s+([A-Za-z]+)", victim_messages, re.IGNORECASE)
            if name_match:
                output_json["firstName"] = name_match.group(1)
                output_json["lastName"] = name_match.group(2)
            phone_match = re.search(r"\b(\d{3}-\d{3}-\d{4})\b", victim_messages)
            if phone_match:
                output_json["telNo"] = phone_match.group(1)
            address_match = re.search(r"address\s+is\s+(.+?)(?:\.|$)", victim_messages, re.IGNORECASE)
            if address_match:
                output_json["address"] = address_match.group(1)
            age_match = re.search(r"(?:I am|age is)\s+(\d+)\s*(?:years old)?", victim_messages, re.IGNORECASE)
            if age_match:
                output_json["age"] = age_match.group(1)
            return output_json
    except Exception as e:
        logging.error(f"Error extracting info_form_data: {str(e)}")
        output_json = DEFAULT_INFO_FORM_DATA.copy()
        output_json["description"] = victim_messages
        output_json["summary"] = victim_messages[:200]
        return output_json