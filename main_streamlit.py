import streamlit as st
import requests

# Set up page configuration
st.set_page_config(
    page_title='MediBOT - Medical Assistant',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: white;
            margin-left: 20%;
        }
        .chat-message.assistant {
            background-color: #475063;
            color: white;
            margin-right: 20%;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
        }
        .chat-message .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
        }
        .stTextInput > div > div > input {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "medical_history" not in st.session_state:
        st.session_state.medical_history = []

initialize_session_state()

# Sidebar for conversation controls
with st.sidebar:
    st.title("Conversation Controls")
    st.markdown("---")
    
    # Display conversation history
    st.subheader("Conversation History")
    for msg in st.session_state.medical_history:
        st.text(f"{msg}")
    
    st.markdown("---")
    
    # New conversation button
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.medical_history = []
        st.rerun()

# Main chat interface
st.title("🏥 MediBOT - Medical Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Update medical history
    st.session_state.medical_history.append(prompt)
    
    # API request
    url = "http://127.0.0.1:9595/qnaConversation"
    headers = {"Content-Type": "application/json"}
    
    body = {
        "user_query": prompt,
        "medical_history": st.session_state.medical_history
    }
    
    try:
        with st.spinner("Thinking..."):
            response = requests.post(url, headers=headers, json=body)
            
            if response.status_code == 200:
                data = response.json()
                if data["succeeded"]:
                    assistant_response = data["data"]
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.write(assistant_response)
                    
                    # Update medical history with assistant's response
                    st.session_state.medical_history.append(assistant_response)
                else:
                    st.error(f"Error: {data['message']}")
            else:
                st.error(f"Failed to get response from the server. Status code: {response.status_code}")
                st.error(f"Response: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Force a rerun to update the UI
    st.rerun()