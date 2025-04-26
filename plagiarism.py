import torch
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence
from nltk.probability import FreqDist
import streamlit as st
from collections import Counter
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import plotly.express as px
from nltk.corpus import stopwords
import torch.nn.functional as F
import PyPDF2
# nltk.download('punkt_tab')
# nltk.download('stopwords')
import nltk
print(nltk.data.path)
#Load tokenizer and model
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')#we are using a pretrained model
model=GPT2LMHeadModel.from_pretrained('gpt2')

def calculate_perplexity(text):
    encoded_input=tokenizer.encode(text,add_special_tokens=False,return_tensors='pt')#preprocessing
    #pt indicates that the output would be a pytorch tensor
    input_ids=encoded_input[0]#this takes the entire sequence of token and not only the first token

    with torch.no_grad():#this disables gradient calculation i.e. model weights are not updated
        outputs=model(input_ids)
        logits=outputs.logits

    perplexity=torch.exp(F.cross_entropy(logits.view(-1,logits.size(-1)),input_ids.view(-1)))#e to the power H(entropy)
    return perplexity.item()
def calculate_burstiness(text):
    tokens=nltk.word_tokenize(text.lower())
    word_freq=FreqDist(tokens)#this gives the dictionary with each token as a key having the count as key
    repeated_count=sum(count>1 for count in word_freq.values())#no of words that appear more than once
    burstiness_score=repeated_count/len(word_freq)#different method
    return burstiness_score

text_area=""
st.title("Plagiarism Detector 🔍")
# Input method selection using radio buttons
option = st.radio("Select Input Method", ["Enter Text", "Upload PDF"])

if option == "Enter Text":
    text_area = st.text_area("Enter your text", height=300)
else:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_area = text  # Set the extracted text to the text_area
        st.success("PDF text extracted successfully!")

if text_area :
    if st.button("Analyze"):
       col1,col2,col3=st.columns([1,1,1])
       with col1:
           st.info("Your input text")
           st.success(text_area)
       with col2:
        
        st.info("Calculated Score")
        perplexity = calculate_perplexity(text_area)
        burstiness_score = calculate_burstiness(text_area)
        st.success(f"Perplexity Score: {perplexity}")
        st.success(f"Burstiness Score: {burstiness_score}")

    # Adjusted threshold values based on testing
        perplexity_threshold = 10000  # Lowered to reduce false positives
        burstiness_threshold = 0.15  # Increased to allow more human-like variability

    # Decision logic
        if perplexity >=perplexity_threshold and burstiness_score >=burstiness_threshold:
         st.error("Text Analysis: Likely AI Generated Content")
        elif perplexity >= perplexity_threshold and burstiness_score <= burstiness_threshold:
         st.success("Text Analysis: Likely Human-Written Content")
        elif perplexity > perplexity_threshold and burstiness_score >= burstiness_threshold:
         st.warning("Text Analysis: Could be Complex Human Text or Poorly Generated AI Text")
        elif perplexity <= perplexity_threshold and burstiness_score <=burstiness_threshold:
         st.info("Text Analysis: Could be Simple Human Text ")
        
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Blue */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    
 