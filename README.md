# RAG Evaluation

This repository contains scripts for evaluating a Retrieval-Augmented Generation (RAG) model using a Jupyter notebook (`untitled6.ipynb`) and deploying it as a Streamlit web application (`app.py`).

## Evaluation Results

The evaluation results of the RAG model can be found in the Jupyter notebook `untitled6.ipynb`. This notebook includes:

- Metrics for evaluating the RAG model performance.
- Analysis of model accuracy, context relevancy, answer relevance, and answer similarity.
- Steps to interpret and utilize the evaluation results.

To view the evaluation results, open `rag_evaluation.ipynb` in a Jupyter notebook environment.

## Streamlit Web Application

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SJ-1407/Rag_evlauation.git
   cd your-repository
   ```

2. Install dependencies in a virtual environment:
   ```bash
   # Create and activate a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
   # Install dependencies

   ```

3. Set up environment variables (if required, e.g., API keys).

### Running the Streamlit App

Run the Streamlit web application using the following command:

```bash
streamlit run app.py
```

### Usage

- The Streamlit application (`app.py`) provides a simple interface to interact with the RAG-powered chatbot.
- Users can input questions related to the PDF document used for training and evaluation.
- The chat history is stored and displayed for context-aware interactions.

### Example Usage

- Ask questions about the document to test the chatbot's retrieval and generation capabilities.
- Observe how the chatbot utilizes retrieved context and generates relevant answers.



