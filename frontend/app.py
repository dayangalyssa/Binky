# filepath: /root/Binky/frontend/streamlit_app.py
import streamlit as st
import requests
import json
import psutil
import time

st.set_page_config(
    page_title="Binky - Library Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background-color: #F7F9FA;
    color: #212121;
    font-family: 'Segoe UI', sans-serif;
}

header, .st-emotion-cache-zq5wmm.ezrtsby0 {
    background-color: #E6F4F4;
}

.css-18e3th9 {
    padding: 1rem;
    background-color: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

h1, h2, h3, h4 {
    color: #1A8181;
}

.stButton button {
    background-color: rgb(59,165,165);
    color: white;
    font-weight: bold;
    border-radius: 6px;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: rgb(36,154,154);
    color: white;
}

.stChatMessage.user {
    background-color: #E6F4F4;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stChatMessage.assistant {
    background-color: #FFFFFF;
    border-left: 4px solid rgb(59,165,165);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stSidebar {
    background-color: #E6F4F4;
    border-right: 1px solid #ccc;
}

</style>
""", unsafe_allow_html=True)

# --- Backend API Configuration ---
BACKEND_URL = "http://localhost:8000"

# --- Language Selector ---
language_options = {
    "Bahasa Indonesia": "bahasa_indonesia",
    "English": "english"
}
selected_language = st.sidebar.selectbox(
    "Pilih Bahasa / Select Language",
    options=list(language_options.keys()),
    index=0
)
language_code = language_options[selected_language]

# --- Main Title ---
st.title("🦙 Binky - Library Assistant Chatbot")
if selected_language == "Bahasa Indonesia":
    st.subheader("Ajukan pertanyaan tentang perpustakaan atau koleksi akademik Anda")
    placeholder_text = "Tulis pertanyaan di sini..."
    processing_text = "Memproses pertanyaan..."
    error_text = "Terjadi kesalahan. Pastikan backend sudah berjalan."
    backend_down_text = "Backend tidak dapat diakses. Pastikan FastAPI berjalan di http://localhost:8000"
else:
    st.subheader("Ask questions about the library or your academic documents")
    placeholder_text = "Type your question here..."
    processing_text = "Processing your query..."
    error_text = "An error occurred. Make sure the backend is running."
    backend_down_text = "Backend not accessible. Make sure FastAPI is running at http://localhost:8000"

# --- Backend Health Check ---
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# --- Get CPU Usage ---
def get_cpu_usage():
    return psutil.cpu_percent(interval=0.5)

# --- Display backend status ---
with st.sidebar:
    st.markdown("### System Metrics")
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    # Backend Status
    with col1:
        if check_backend_health():
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Offline")
    
    # CPU Usage
    with col2:
        cpu_usage = get_cpu_usage()
        st.metric(label="CPU Usage", value=f"{cpu_usage}%")
    
    # Memory Usage
    memory_usage = psutil.virtual_memory().percent
    st.metric(label="Memory Usage", value=f"{memory_usage}%")
    
    if not check_backend_health():
        st.warning("backend offline:\n```bash\ncd backend\nuvicorn main:app --reload\n```")

# --- Function to send request to backend ---
def send_query_to_backend(query: str, language: str):
    try:
        payload = {
            "query": query,
            "language": language
        }
        
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            #timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Backend error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Make sure FastAPI is running."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 Detail Response"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Docs Retrieved", message["metadata"]["num_docs_retrieved"])
                with col2:
                    st.metric("Has Context", "Yes" if message["metadata"]["has_relevant_context"] else "No")
                with col3:
                    st.metric("Response Time", f"{message['metadata'].get('response_time', 'N/A')}s")

# --- Chat input and response ---
if prompt := st.chat_input(placeholder_text):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner(processing_text):
            import time
            start_time = time.time()
            
            # Get initial CPU
            cpu_before = get_cpu_usage()
            
            # Send request to backend
            result, error = send_query_to_backend(prompt, language_code)
            
            # Get final CPU
            cpu_after = get_cpu_usage()
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            cpu_usage_during_request = round((cpu_before + cpu_after) / 2, 2)
            
            if error:
                # Handle error
                st.error(f"❌ {error}")
                if "Cannot connect" in error:
                    st.info("💡 Pastikan backend FastAPI sudah berjalan dengan perintah:\n```bash\ncd backend\nuvicorn main:app --reload\n```")
                
                # Add error message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ {error}"
                })
            else:
                # Display successful response
                response_text = result["response"]
                st.markdown(response_text)
                
                # Show response metadata
                with st.expander("📊 Response Details"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Docs Retrieved", result["num_docs_retrieved"])
                    with col2:
                        st.metric("Has Context", "Yes" if result["has_relevant_context"] else "No")
                    with col3:
                        st.metric("Response Time", f"{response_time}s")
                    with col4:
                        st.metric("CPU Usage", f"{cpu_usage_during_request}%")
                
                # Add successful response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "metadata": {
                        "num_docs_retrieved": result["num_docs_retrieved"],
                        "has_relevant_context": result["has_relevant_context"],
                        "response_time": response_time,
                        "cpu_usage": cpu_usage_during_request
                    }
                })
# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Run:")
st.sidebar.markdown("""
1. **Start Backend:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

3. **Access:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000/docs
""")

# --- Debug info ---
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.json({
        "Backend URL": BACKEND_URL,
        "Selected Language": language_code,
        "Backend Health": check_backend_health(),
        "CPU Usage": f"{get_cpu_usage()}%",
        "Memory Usage": f"{psutil.virtual_memory().percent}%",
        "Total Messages": len(st.session_state.messages)
    })

# Add real-time CPU monitor
if st.sidebar.checkbox("Show CPU Monitor"):
    cpu_chart = st.sidebar.line_chart()
    mem_chart = st.sidebar.line_chart()
    
    # Only run if visible on page
    if st.sidebar.button("Start Monitoring"):
        for i in range(100):
            # Add metrics to charts
            cpu_chart.add_rows([get_cpu_usage()])
            mem_chart.add_rows([psutil.virtual_memory().percent])
            # Display for 1 second
            time.sleep(1)