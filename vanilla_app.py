import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import streamlit as st
from src.agents.conversation_manager import ConversationManager, ConversationMode, ConversationType
import json
from config.logging_config import setup_logger  # Import for logging alignment
import csv
from filelock import FileLock

# Set wide layout to expand the app to full browser width
st.set_page_config(layout="wide")

# Custom CSS: Match column heights, confine chat_input to col1
st.markdown("""
    <style>
        /* Make the row containing columns a flex container for height matching */
        div[data-testid="column"].css-1r6slb0 {
            display: flex;
            flex-wrap: nowrap;
            align-items: stretch;  /* Stretch columns to match tallest */
        }
        /* Target individual columns for full height */
        div[data-testid="column"] {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        /* Chat history grows naturally */
        div[data-testid="column"]:first-of-type > div:not(.stChatInput) {
            flex: 1;
        }
        /* Confine chat_input to col1's width and position */
        .stChatInput {
            position: relative;
            width: 50% !important;  /* Matches col1's 3/(3+2)=50% ratio; adjust if col1 ratio changes */
            margin-left: 0 !important;  /* Align to left edge of app */
            padding: 10px;
        }
        /* Optional: Style input for consistency */
        .stChatInput > div > div > input {
            width: 100% !important;
        }
    </style>
""", unsafe_allow_html=True)


# Logger for alignment with codebase
logger = setup_logger("StreamlitApp", "app")  # Adjust subdirectory as needed

# All possible fields (excluding conversational_response)
ALL_FIELDS = [
    "scam_incident_date", "scam_type", "scam_approach_platform", "scam_communication_platform",
    "scam_transaction_type", "scam_beneficiary_platform", "scam_beneficiary_identifier",
    "scam_contact_no", "scam_email", "scam_moniker", "scam_url_link", "scam_amount_lost",
    "scam_incident_description"
]

# Main Streamlit app
st.title("Police Chatbot Interface (Vanilla RAG)")

# Initialize or reinitialize ConversationManager in non-autonomous mode
if "manager" not in st.session_state:
    st.session_state.model_name = "qwen2.5:7b"  # Default
    st.session_state.manager = ConversationManager(
        mode=ConversationMode.NONAUTONOMOUS,
        conversation_type=ConversationType.VANILLA_RAG,
        police_model_name=st.session_state.model_name
    )
    st.session_state.messages = []  # List of {"role": "user" or "assistant", "content": str}
    st.session_state.conversation_started = False
    st.session_state.conversation_id = None
    st.session_state.cumulative_structured = {field: "" for field in ALL_FIELDS}
    st.session_state.cumulative_structured["scam_amount_lost"] = 0.0
    st.session_state.evaluations = {}  # dict: msg_index: {"language_proficiency": int, "tech_literacy": int, "emotional_state": int}
    st.session_state.profile_id = None  # NEW: Initialize profile_id as None (optional)

# Sidebar for model selection and debug info (model fixed per conversation; changeable on new chat)
with st.sidebar:
    st.header("Settings")
    if not st.session_state.conversation_started:
        selected_model = st.selectbox(
            "Select LLM Model",
            options=["qwen2.5:7b", "granite3.2:8b", "mistral:7b", "gpt-4o-mini"],  # From your police_agent.py tests
            index=["qwen2.5:7b", "granite3.2:8b", "mistral:7b", "gpt-4o-mini"].index(st.session_state.model_name)
        )
        if selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            provider = "OpenAI" if "gpt" in selected_model else "Ollama"
            st.session_state.manager = ConversationManager(
                mode=ConversationMode.NONAUTONOMOUS,
                conversation_type=ConversationType.VANILLA_RAG,
                police_model_name=selected_model,
                police_llm_provider=provider
            )
            st.rerun()

        # NEW: Add number_input for profile_id under the model selectbox (editable only before start)
        entered_profile_id = st.number_input("Profile ID (optional)", min_value=0, value=0, step=1)
        st.session_state.profile_id = entered_profile_id if entered_profile_id > 0 else None  # Set to None if 0 (optional)
    else:
        st.text(f"Model: {st.session_state.model_name} (fixed for this chat)")

    # Display current conversation ID for tracking (aligns with IDManager emphasis)
    if st.session_state.conversation_id:
        st.text(f"Conversation ID: {st.session_state.conversation_id}")

    # NEW: Always display the profile_id (even after start, for logging/reference)
    st.text(f"Profile ID: {st.session_state.profile_id if st.session_state.profile_id else 'None'}")

# Create columns: left for chat (narrower), right for details (wider for more space)
col1, col2 = st.columns([2, 2])  # Equal space; adjust to [2,3] if need even more for right

# Use a container in col1 for chat history (allows dynamic appends in the same run)
chat_container = col1.container()

# Display *previous* messages in the container (full history except any new ones added below)
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Evaluate this response", expanded=False):
                default_lang = st.session_state.evaluations.get(i, {}).get("language_proficiency", 1) - 1
                default_tech = st.session_state.evaluations.get(i, {}).get("tech_literacy", 1) - 1
                default_emo = st.session_state.evaluations.get(i, {}).get("emotional_state", 1) - 1

                lang = st.selectbox("Language Proficiency (1-5)", [1,2,3,4,5], index=default_lang, key=f"lang_{i}")
                tech = st.selectbox("Tech Literacy (1-5)", [1,2,3,4,5], index=default_tech, key=f"tech_{i}")
                emo = st.selectbox("Emotional State (1-5)", [1,2,3,4,5], index=default_emo, key=f"emo_{i}")

                if st.button("Submit Evaluation", key=f"submit_{i}"):
                    eval_dict = {
                        "language_proficiency": lang,
                        "tech_literacy": tech,
                        "emotional_state": emo
                    }
                    st.session_state.evaluations[i] = eval_dict
                    # Update the CSV with the evaluations as JSON in the single column
                    csv_path = st.session_state.manager.history_csv_path
                    lock = FileLock(f"{csv_path}.lock")
                    with lock:
                        rows = []
                        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        conv_id_str = str(st.session_state.conversation_id)
                        message_content = st.session_state.messages[i]["content"]
                        matching_row = None
                        for row in rows:
                            if row["conversation_id"] == conv_id_str and row["sender_type"] == "police" and row["content"] == message_content:
                                matching_row = row
                                break
                        if matching_row:
                            matching_row["communication_appropriateness_rating"] = json.dumps(eval_dict)
                            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                                writer.writeheader()
                                writer.writerows(rows)
                            logger.info(f"Updated evaluation in CSV for conversation {conv_id_str}, message index {i}")
                        else:
                            st.warning("Could not find matching row in CSV to update evaluation.")

# Chat input (confined to col1 via CSS, auto-clears)
prompt = st.chat_input("Type your message here...")

# Process input if submitted
if prompt:
    if not prompt.strip():
        st.warning("Please enter a non-empty message.")
    else:
        try:
            # Start conversation if not already started (only on first prompt)
            if not st.session_state.conversation_started:
                # NEW: Pass the profile_id as parameter to start_conversation
                st.session_state.conversation_id = st.session_state.manager.start_conversation(profile_id=st.session_state.profile_id)
                st.session_state.conversation_started = True
                logger.info(f"Started new non-autonomous conversation ID: {st.session_state.conversation_id} with profile_id: {st.session_state.profile_id}")

            # Append and *immediately display* user message in the chat container (before LLM processing)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Calculate index for the upcoming assistant message
            assistant_index = len(st.session_state.messages)

            # Process LLM response with spinner for feedback during blocking call
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.manager.process_turn(prompt)
                        assistant_content = response.get("response", "No response from AI.")
                        structured_data = response.get("structured_data", {})
                        logger.debug(f"Processed response for conversation {st.session_state.conversation_id}: {assistant_content}")

                    # Display assistant response immediately after processing
                    st.markdown(assistant_content)

                    # Add expander for evaluation of this new assistant message
                    with st.expander("Evaluate this response", expanded=False):
                        default_lang = st.session_state.evaluations.get(assistant_index, {}).get("language_proficiency", 1) - 1
                        default_tech = st.session_state.evaluations.get(assistant_index, {}).get("tech_literacy", 1) - 1
                        default_emo = st.session_state.evaluations.get(assistant_index, {}).get("emotional_state", 1) - 1

                        lang = st.selectbox("Language Proficiency (1-5)", [1,2,3,4,5], index=default_lang, key=f"lang_{assistant_index}")
                        tech = st.selectbox("Tech Literacy (1-5)", [1,2,3,4,5], index=default_tech, key=f"tech_{assistant_index}")
                        emo = st.selectbox("Emotional State (1-5)", [1,2,3,4,5], index=default_emo, key=f"emo_{assistant_index}")

                        if st.button("Submit Evaluation", key=f"submit_{assistant_index}"):
                            eval_dict = {
                                "language_proficiency": lang,
                                "tech_literacy": tech,
                                "emotional_state": emo
                            }
                            st.session_state.evaluations[assistant_index] = eval_dict
                            # Update the CSV with the evaluations as JSON in the single column
                            csv_path = st.session_state.manager.history_csv_path
                            lock = FileLock(f"{csv_path}.lock")
                            with lock:
                                rows = []
                                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                                    reader = csv.DictReader(f)
                                    rows = list(reader)
                                conv_id_str = str(st.session_state.conversation_id)
                                message_content = assistant_content  # Since not yet appended, use local var
                                matching_row = None
                                for row in rows:
                                    if row["conversation_id"] == conv_id_str and row["sender_type"] == "police" and row["content"] == message_content:
                                        matching_row = row
                                        break
                                if matching_row:
                                    matching_row["communication_appropriateness_rating"] = json.dumps(eval_dict)
                                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                                        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                                        writer.writeheader()
                                        writer.writerows(rows)
                                    logger.info(f"Updated evaluation in CSV for conversation {conv_id_str}, message index {assistant_index}")
                                else:
                                    st.warning("Could not find matching row in CSV to update evaluation.")

            # Append assistant message to history (after rendering)
            st.session_state.messages.append({"role": "assistant", "content": assistant_content})

            # Update cumulative structured data (exclude conversational_response, update if truthy)
            for key, value in structured_data.items():
                if key != "conversational_response" and value:
                    st.session_state.cumulative_structured[key] = value

        except Exception as e:
            logger.error(f"Error processing response in conversation {st.session_state.conversation_id}: {e}", exc_info=True)
            st.error(f"An error occurred: {str(e)}. Please try again.")

# Display structured data *after* processing (reflects any updates from the latest turn)
with col2:
    st.subheader("Scam Report Details")
    
    # Split shorter fields into two sub-columns
    short_fields_left = [
         "scam_type", "scam_approach_platform",
        "scam_communication_platform", "scam_transaction_type", "scam_beneficiary_platform","scam_beneficiary_identifier"
    ]
    short_fields_right = [
        "scam_incident_date", "scam_contact_no", "scam_email",
        "scam_moniker", "scam_url_link", "scam_amount_lost"
    ]
    
    subcol_left, subcol_right = st.columns(2)  # Equal split; adjust to [1,1] or [2,1] if needed
    
    with subcol_left:
        for key in short_fields_left:
            value = st.session_state.cumulative_structured.get(key, "")
            label = key.replace("_", " ").title()
            if isinstance(value, (str, float, int)):
                st.text_input(label, value=str(value), disabled=True)
            else:
                st.json(value)  # Fallback, though unlikely for these fields
    
    with subcol_right:
        for key in short_fields_right:
            value = st.session_state.cumulative_structured.get(key, "")
            label = key.replace("_", " ").title()
            if isinstance(value, (str, float, int)):
                st.text_input(label, value=str(value), disabled=True)
            else:
                st.json(value)  # Fallback
    
    # Full-width fields below the sub-columns (spanning entire col2)
    for key in ["scam_incident_description"]:
        value = st.session_state.cumulative_structured.get(key, "")
        label = key.replace("_", " ").title()
        if key == "scam_incident_description":
            # Use text_area for description to make it "big" (taller/wider)
            st.text_area(label, value=str(value), height=150, disabled=True)  # Adjust height as needed
        else:
            st.json(value)

    # Display evaluations table
    st.subheader("Response Evaluations")
    eval_data = []
    for idx in sorted([k for k in st.session_state.evaluations.keys() if st.session_state.messages[k]["role"] == "assistant"]):
        scores = st.session_state.evaluations[idx]
        eval_data.append({
            "Message Index": idx,
            "Language Proficiency": scores["language_proficiency"],
            "Tech Literacy": scores["tech_literacy"],
            "Emotional State": scores["emotional_state"]
        })
    if eval_data:
        st.dataframe(eval_data)
    else:
        st.text("No evaluations submitted yet.")

# New Chat button to reset
if st.button("New Chat"):
    if st.session_state.conversation_started:
        end_status = st.session_state.manager.force_end_conversation()
        logger.info(f"Ended conversation {st.session_state.conversation_id}: {end_status}")
    st.session_state.messages = []
    st.session_state.conversation_started = False
    st.session_state.conversation_id = None
    st.session_state.cumulative_structured = {field: "" for field in ALL_FIELDS}
    st.session_state.cumulative_structured["scam_amount_lost"] = 0.0
    st.session_state.evaluations = {}
    st.session_state.profile_id = None  # NEW: Reset profile_id on new chat
    st.rerun()