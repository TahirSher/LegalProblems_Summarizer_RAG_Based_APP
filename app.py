import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import pipeline
import streamlit as st
from datasets import load_dataset

# Load Dataset from Hugging Face with Error Handling
def load_huggingface_dataset(dataset_name, config=None, split="train"):
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        data = pd.DataFrame(dataset)  # Convert to pandas DataFrame
        return data
    except Exception as e:
        st.error(f"Failed to load dataset '{dataset_name}' with config '{config}'. Please try 'lex_glue' or 'eurlex' with appropriate config.")
        st.error(f"Error details: {e}")
        return None

# Prepare the Retrieval Model (BM25)
def prepare_bm25(corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Search for Similar Documents
def search_documents(bm25, query, corpus, top_n=5):
    tokenized_query = query.split(" ")
    scores = bm25.get_top_n(tokenized_query, corpus, n=top_n)
    return scores

# Summarization Model
def summarize_text(text):
    try:
        # Use a public model for summarization
        summarizer = pipeline("summarization", model="t5-base")  # Change to a public model
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return "Summary could not be generated."

# Streamlit App
def main():
    st.title("Legal Case Summarizer")
    
    # Dataset Selection
    dataset_name = st.selectbox("Choose Hugging Face dataset", ["lex_glue", "eurlex"])
    config = None
    
    # Config Selection for lex_glue
    if dataset_name == "lex_glue":
        config = st.selectbox("Select config for lex_glue", ["case_hold", "ecthr_a", "ecthr_b", "eurlex", "ledgar", "scotus", "unfair_tos"])
    
    split = st.selectbox("Choose dataset split", ["train", "validation", "test"])
    
    if dataset_name:
        st.write("Loading dataset from Hugging Face...")
        data = load_huggingface_dataset(dataset_name, config=config, split=split)
        
        if data is not None:
            corpus = data['text'].tolist() if 'text' in data.columns else data.iloc[:, 0].tolist()
            titles = data['title'].tolist() if 'title' in data.columns else ["Title " + str(i) for i in range(len(corpus))]

            # Prepare BM25 Model
            bm25 = prepare_bm25(corpus)

            # User Input
            query = st.text_input("Enter keywords for case search:")
            num_results = st.slider("Number of results to display", 1, 10, 5)

            if query:
                st.write("Searching for relevant cases...")
                results = search_documents(bm25, query, corpus, top_n=num_results)

                for idx, result in enumerate(results):
                    st.write(f"### Case {idx+1}: {titles[corpus.index(result)]}")
                    st.write(result)
                    
                    # Summarize the case
                    st.write("Summary:")
                    summary = summarize_text(result)
                    st.write(summary)

if __name__ == "__main__":
    main()
