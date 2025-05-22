"""
Main application entry point for the Llama RAG chatbot.
"""
import os
import sys
import argparse
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import login
import os
from src.chatbot import LlamaRagChatbot
from utils.helpers import setup_rag_pipeline, time_function
import config

def setup_cli_app():
    """Set up command-line interface application."""
    parser = argparse.ArgumentParser(description="Llama RAG Chatbot")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the RAG pipeline without starting interactive mode")
    parser.add_argument("--language", choices=["bahasa_indonesia", "english"], default="bahasa_indonesia", 
                       help="Output language (default: bahasa_indonesia)")
    args = parser.parse_args()
    
    # Set up the RAG pipeline
    success, message, vectorstore = setup_rag_pipeline()
    print(message)
    
    if not success:
        sys.exit(1)
    
    if args.setup_only:
        print("RAG pipeline setup complete. Exiting.")
        sys.exit(0)
    
    # Initialize the chatbot
    chatbot = LlamaRagChatbot(vectorstore)
    
    # Set language
    chatbot.set_language(args.language)
    print(f"Chatbot language set to: {args.language}")
    
    # Run CLI interactive mode
    if args.language == "bahasa_indonesia":
        welcome_msg = "\nLlama RAG Chatbot - Mode Interaktif"
        prompt_msg = "Anda: "
        exit_msg = "Selamat tinggal!"
        processing_msg = "\nMemproses pertanyaan Anda..."
        bot_prefix = "\nChatbot: "
        retrieved_msg = "\n[Ditemukan {n} dokumen yang relevan]"
    else:
        welcome_msg = "\nLlama RAG Chatbot - Interactive Mode"
        prompt_msg = "You: "
        exit_msg = "Goodbye!"
        processing_msg = "\nProcessing your query..."
        bot_prefix = "\nChatbot: "
        retrieved_msg = "\n[Retrieved {n} relevant documents]"
    
    print(welcome_msg)
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        user_input = input(f"\n{prompt_msg}").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print(exit_msg)
            break
        
        if not user_input:
            continue
        
        # Process the query
        print(processing_msg)
        result = time_function(chatbot.process_query)(user_input)
        
        # Print the response
        print(f"{bot_prefix}{result['response']}")
        print(retrieved_msg.format(n=result['num_docs_retrieved']))


def setup_streamlit_app():
    """Set up Streamlit web application."""
    st.set_page_config(
        page_title="Binky",
        page_icon="🦙",
        layout="wide"
    )
    
    st.title("🦙 Binky")
    
    # Language selector
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
    
    if selected_language == "Bahasa Indonesia":
        st.subheader("Ajukan pertanyaan tentang dokumen Anda")
        placeholder_text = "Ketik pertanyaan Anda di sini..."
        button_text = "Kirim"
        processing_text = "Memproses pertanyaan..."
        setup_text = "Menyiapkan chatbot..."
        no_docs_text = "Tidak ada dokumen yang ditemukan. Periksa file data Anda."
    else:
        st.subheader("Ask questions about your documents")
        placeholder_text = "Type your question here..."
        button_text = "Send"
        processing_text = "Processing query..."
        setup_text = "Setting up chatbot..."
        no_docs_text = "No documents were found. Please check your data files."
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        with st.spinner(setup_text):
            success, message, vectorstore = setup_rag_pipeline()
            
            if not success:
                st.error(no_docs_text)
                st.stop()
            
            st.session_state.chatbot = LlamaRagChatbot(vectorstore)
            st.session_state.chatbot.set_language(language_code)
    
    # Update language if changed
    if "current_language" not in st.session_state:
        st.session_state.current_language = language_code
    elif st.session_state.current_language != language_code:
        st.session_state.chatbot.set_language(language_code)
        st.session_state.current_language = language_code
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(placeholder_text):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner(processing_text):
                result = st.session_state.chatbot.process_query(prompt)
                response = result["response"]
                
                # Display relevant document info
                if result["num_docs_retrieved"] > 0:
                    if language_code == "bahasa_indonesia":
                        st.caption(f"*Ditemukan {result['num_docs_retrieved']} dokumen yang relevan*")
                    else:
                        st.caption(f"*Found {result['num_docs_retrieved']} relevant documents*")
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    # Check if running in Streamlit
    if 'STREAMLIT_SHARE_PATH' in os.environ or 'STREAMLIT_RUN_PATH' in os.environ:
        setup_streamlit_app()
    else:
        setup_cli_app()