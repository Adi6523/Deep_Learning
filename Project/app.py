import streamlit as st
import pandas as pd
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# --- 1. Load Data and Prepare Documents ---
@st.cache_resource
def load_and_prepare_data(csv_path):
    """Loads CSV, converts to documents, and builds the FAISS vector database."""
    # Load CSV using pandas
    df = pd.read_csv(csv_path)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    documents = []
    for index, row in df.iterrows():
        # Create a document for each product
        # Include all details in page_content so the LLM has full context
        content = f"Product Name: {row['name']}\nPrice: {row['price']}\nDescription: {row['description']}"
        metadata = {"name": row['name'], "price": row['price']}
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    # Split text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Generate embeddings and store in FAISS vector database
    # Using HuggingFace embeddings (free and runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(split_docs, embeddings)
    
    return df, vector_db

# --- 2. Extract Budget from Query ---
def extract_budget(query):
    """Extracts a budget amount from the user's query if present."""
    # Look for patterns like "under 20000", "< 20000", "below 20000", "under 20k"
    match = re.search(r'(?:under|below|<)\s*(\d+(?:\.\d+)?)\s*(k|lakh)?', query.lower())
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'k':
            val *= 1000
        elif unit == 'lakh':
            val *= 100000
        return int(val)
    return None

# --- 3. Main Streamlit Interface ---
def main():
    st.set_page_config(page_title="E-Commerce Product Assistant", page_icon="🛒")
    st.title("🛒 E-Commerce Product Assistant")
    st.write("Ask me to recommend products based on your requirements and budget!")
    
    # Sidebar for API Key
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", value="API KEY", type="password")
    st.sidebar.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    # Load data (cached)
    try:
        df, vector_db = load_and_prepare_data("products.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # User Input
    query = st.text_input("What are you looking for? (e.g., 'Best phone under 20000 with good camera')")
    
    if st.button("Get Recommendations"):
        if not api_key:
            st.warning("Please enter your Google Gemini API Key in the sidebar.")
            return
            
        if not query:
            st.warning("Please enter a query.")
            return
            
        with st.spinner("Finding the best products for you..."):
            os.environ["GOOGLE_API_KEY"] = api_key
            
            # Step 1: Extract budget and filter products based on price
            budget = extract_budget(query)
            
            if budget:
                st.info(f"Detected budget: Under ₹{budget}")
                # Filter pandas dataframe directly based on budget
                filtered_df = df[df['price'] <= budget]
                
                if filtered_df.empty:
                    st.warning("No products found within your budget.")
                    return
                
                # Create a temporary filtered vector database for better precision
                # (Alternative: use vector store metadata filtering, but this is simpler for small datasets)
                filtered_docs = []
                for _, row in filtered_df.iterrows():
                    content = f"Product Name: {row['name']}\nPrice: {row['price']}\nDescription: {row['description']}"
                    filtered_docs.append(Document(page_content=content, metadata={"name": row['name'], "price": row['price']}))
                
                temp_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                search_db = FAISS.from_documents(filtered_docs, temp_embeddings)
            else:
                search_db = vector_db
                
            # Step 2: Use RAG to retrieve relevant product descriptions
            retriever = search_db.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(query)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Step 3: Use an LLM to generate a final recommendation
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            
            prompt_template = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                You are a helpful E-commerce Product Assistant.
                Based on the user's query and the provided product context, recommend the top 2-3 most suitable products.
                
                User Query: {query}
                
                Available Products (Context):
                {context}
                
                For each recommended product, provide exactly the following format:
                **Product Name:** [Name]
                **Price:** ₹[Price]
                **Reason:** [Brief reason why it's recommended based on the user's specific request]
                
                If the context doesn't have good matches, politely explain that you couldn't find exact matches but offer the closest alternatives from the context.
                """
            )
            
            chain = prompt_template | llm
            
            # Generate the response
            try:
                response = chain.invoke({"query": query, "context": context})
                
                # Display Results
                st.subheader("💡 Recommendations:")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error generating recommendation: {e}")

if __name__ == "__main__":
    main()
