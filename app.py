import streamlit as st
import subprocess

# Specify the version you want to install
desired_version = "0.28.0"

# Construct the pip install command
install_command = f"pip install openai=={desired_version}"

# Execute the command using subprocess
try:
    subprocess.run(install_command, shell=True, check=True)
    print(f"openai version {desired_version} installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error installing openai: {e}")

install_command = "pip install transformers"

# Execute the command using subprocess
try:
    subprocess.run(install_command, shell=True, check=True)
    print("transformers library installed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error installing transformers: {e}")

#!pip install pymupdf
import fitz  # PyMuPDF
import pandas as pd
import torch
#!pip install -U openai pinecone-client datasets
from tqdm.auto import tqdm  # this is our progress bar
#!pip install openai==0.28.0
import openai
#!pip show openai
#!pip install pinecone-client
#!pip install --upgrade pinecone-client
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Vector + Data imports
import pandas as pd
import numpy as np
#import torch
#from transformers import DistilBertTokenizer, DistilBertModel

#Pinecone imports
import pinecone
from pinecone import PodSpec
from pinecone import Pinecone
from pinecone import ServerlessSpec

#OpenAI
from pinecone import Pinecone
#!pip install PyPDF2
import PyPDF2

from PyPDF2 import PdfReader
import base64
import openai
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
# Set your OpenAI API key
#openai.api_key = "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"
nltk.download('punkt')
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sentences = nltk.sent_tokenize(text)
    polarity_scores = []
    subjectivity_scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity_scores.append(blob.sentiment.polarity)
        subjectivity_scores.append(blob.sentiment.subjectivity)
    return polarity_scores, subjectivity_scores

# Function to perform sentiment analysis using NLTK's VaderSentimentAnalyzer for each sentence
def analyze_sentiment_nltk(text):
    sid = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    scores = []
    for sentence in sentences:
        scores.append(sid.polarity_scores(sentence)['compound'])
    return scores

# Function to perform temporal analysis using NLTK
def temporal_analysis(text):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence)]
    fdist = FreqDist(words)
    return fdist

def AVM(q,k,o):
    openai.api_key = k
    pinecone_key = o
    query =  q
    def getData(path):
        def extract_text_from_pdf(pdf_path):
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            return text

        # Replace "your_pdf_file.pdf" with the actual file name you uploaded
        pdf_path = path
        text = extract_text_from_pdf(pdf_path)
        #print(text)

        def split_text_into_parts(text, num_parts=10):
            # Calculate the length of each part
            part_length = len(text) // num_parts
            # Split the text into parts
            parts = [text[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
            return parts

        # Example usage:
        # Assuming you have already extracted the text and stored it in the 'text' variable
        text_parts = split_text_into_parts(text, num_parts=20)
        return text_parts


    my_list = getData("book-no-1.pdf")

    def getIndex():
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("quickstart")
        return index

    index = getIndex()

    def upserts(query, my_list, index):
        index = index
        my_list = my_list
        query = query
        MODEL = "text-embedding-3-small"

        res = openai.Embedding.create(
            input=[query], engine=MODEL
        )

        embeds = [record['embedding'] for record in res['data']]

        # load the first 1K rows of the TREC dataset
        #trec = load_dataset('trec', split='train[:1000]')

        batch_size = 32  # process everything in batches of 32
        for i in tqdm(range(0, len(my_list), batch_size)):
            # set end position of batch
            i_end = min(i+batch_size, len(my_list))
            # get batch of lines and IDs
            lines_batch = my_list[i: i+batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            res = openai.Embedding.create(input=lines_batch, engine=MODEL)
            embeds = [record['embedding'] for record in res['data']]
            # prep metadata and upsert batch
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))

    upserts(query, my_list, index)

    def getRes(query):
        query = query
        MODEL = "text-embedding-3-small"

        xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

        res = index.query(vector = [xq], top_k=5, include_metadata=True)

        return res
    
    def vectorQuotes(query):
        similarity = getRes(query)
        justQuotes = []
        for i in range(len(similarity['matches'])):
            justQuotes.append(similarity['matches'][i]['metadata']['text'])
        return justQuotes
    
    def getFinalSummaryGPT4(my_list, queryContext):
        my_list = my_list
        queryContext = queryContext

        # Function to split a list into equal sublists
        def split_list(lst, num_sublists):
            avg = len(lst) // num_sublists
            remainder = len(lst) % num_sublists
            return [lst[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(num_sublists)]

        # Split 'my_list' into n equal sublists
        n = 5
        sublists = split_list(my_list, n)

        # Generate summaries for each sublist using the OpenAI API
        sublist_summaries = []

        for i, sublist in enumerate(sublists):
            sublist_text = ' '.join(sublist)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.9,
                top_p=0.9,
                messages= [{ "role": "user", "content": queryContext+sublist_text }] )

            # Extract the summary from the API response
            summary = response.choices[0].message.content
            sublist_summaries.append(summary)

        # Combine the 10 summaries into one variable
        combined_summary = ' '.join(sublist_summaries)

        # Add a specific prompt tailored to your data
        specific_prompt = f"Given the following summaries:\n{combined_summary}\n\nGenerate a coherent final summary that captures the essence of the provided information."

        specific_prompt = queryContext + specific_prompt
        # Use OpenAI API to generate the final coherent summary

        response_combined = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.9,
            top_p=0.9,
            messages= [{ "role": "user", "content": specific_prompt}] )

        # Extract the final summary from the API response
        final_summary = response_combined.choices[0].message.content.strip()

        return final_summary
    
    contexts = "Based solely on the following information create a coherent answer to the question"
    justQuotes = vectorQuotes(query)
    queryContext = query + ". " + contexts
    responseQuotes = getFinalSummaryGPT4(justQuotes, queryContext)
    return responseQuotes


#print(AVM("If the CHC is far away where should people go?","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","86f31e00-a5cc-438d-b95b-4089562e9b57"))

# Streamlit App
def main():
    st.markdown("<h1 style='text-align: center;'>ChatCHW</h1>", unsafe_allow_html=True)
    # Query input
    pdf_placeholder = st.empty()

    # Placeholder for Close button
    close_button_placeholder = st.empty()
    if st.button("Medical Guide"):
        # Embedded PDF popup
        pdf_path = "book-no-1.pdf"  # Update with the actual path to your PDF file
        with open(pdf_path, "rb") as file:
            pdf_contents = file.read()
            base64_pdf = base64.b64encode(pdf_contents).decode("utf-8")

            # Display PDF using Markdown with iframe
            pdf_placeholder.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>', unsafe_allow_html=True)

        # Show Close button after View PDF is clicked
        if close_button_placeholder.button("Close"):
            pdf_placeholder.empty()  # Clear the PDF content
            close_button_placeholder.empty()
    query_input = st.text_area("Query", key="<query>")
    query = query_input

    # OpenAI API key input
    api_key_input = st.text_input("OpenAI API Key", key="<ok>")
    api_key = api_key_input
    
    p_input = st.text_input("Pinecone API Key", key="<pk>")
    pk = p_input
    # Generate response on button click
    # View PDF button
    st.write('v0.0.1')
    if st.button("Generate Response"):
        if not query or not api_key or not pk:
            st.warning("Please fill in all fields.")
        else:
            # Call OpenAI API

            avm = AVM(query,api_key,pk)

            # Display the generated text
            st.header('Output')
            st.write(avm)

            polarity_scores, subjectivity_scores = analyze_sentiment(avm)
            sentiment_scores_nltk = analyze_sentiment_nltk(avm)

            # Subjectivity Analysis
            subjectivity_scores = analyze_sentiment(avm)[1]

            # Temporal Analysis
            fdist = temporal_analysis(avm)
            df = pd.DataFrame(list(fdist.items()), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

            if st.button("👍 Thumbs Up"):
                st.write("Nice")
            if st.button("👎 Thumbs Down"):
                st.write("Oh no")

            # Streamlit App
            st.write('DevOps & Analysis')

            # Arrange the graphs as 2 on top and 2 on the bottom
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

            # Sentiment Polarity for each sentence
            axes[0, 0].bar(range(1, len(polarity_scores) + 1), polarity_scores)
            axes[0, 0].set_title('Sentiment Polarity for Each Sentence')
            axes[0, 0].set_xlabel('Sentence Number')
            axes[0, 0].set_ylabel('Sentiment Polarity')

            # Sentiment Strength for each sentence
            axes[0, 1].bar(range(1, len(sentiment_scores_nltk) + 1), sentiment_scores_nltk)
            axes[0, 1].set_title('Sentiment Strength for Each Sentence')
            axes[0, 1].set_xlabel('Sentence Number')
            axes[0, 1].set_ylabel('Sentiment Strength')

            # Subjectivity Analysis for each sentence
            axes[1, 0].bar(range(1, len(subjectivity_scores) + 1), subjectivity_scores)
            axes[1, 0].set_title('Subjectivity Analysis for Each Sentence')
            axes[1, 0].set_xlabel('Sentence Number')
            axes[1, 0].set_ylabel('Subjectivity')

            # Temporal Analysis for each sentence
            axes[1, 1].bar(df['Word'][:13], df['Frequency'][:13])
            axes[1, 1].set_title('Temporal Analysis (Top 10 Words)')

            # Set background transparent and axis/label colors to white
            for ax in axes.flatten():
                ax.set_facecolor('none')  # Set background to transparent
                ax.title.set_color('white')  # Set title color to white
                ax.xaxis.label.set_color('white')  # Set x-axis label color to white
                ax.yaxis.label.set_color('white')  # Set y-axis label color to white
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_color('white')  # Set tick label color to white

            # Set the background of the entire plot to transparent
            fig.patch.set_facecolor('none')
            fig.patch.set_alpha(0.0)

            # Display the plots
            st.pyplot(fig)
            # Embedded PDF container
            

if __name__ == "__main__":
    main()