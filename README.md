# ChatDoc 🩺 — AI Medical Assistant for Patient Files

ChatDoc is an intelligent medical assistant built with Streamlit and LangChain. It allows users to upload patient medical records in PDF format and ask health-related questions. The system provides responses by analyzing the document and leveraging powerful LLMs and vector embeddings.

---

## 🔧 Features

* Upload and parse patient medical records (PDF)
* Automatically chunk, embed, and store documents using ObjectBox vector database
* Use **GROQ's LLaMA 3.1** model for fast, intelligent responses
* Integrate **Google Generative AI Embeddings** for document understanding
* Ask contextual questions and receive medically-informed answers
* Shows relevant document chunks used for each answer

---

## 📁 Sample Patient File

A sample file is provided for testing:
`Sample-filled-in-MR.pdf`

---

## 🚀 How to Run This Project

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/chatdoc.git
cd chatdoc
```

---

### 2. **Create a Virtual Environment**

```bash
python -m venv chatdoc_env
```

Activate it:

* **Windows**:

  ```bash
  .\chatdoc_env\Scripts\activate
  ```
* **macOS/Linux**:

  ```bash
  source chatdoc_env/bin/activate
  ```

---

### 3. **Install Dependencies**

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

### 4. **Set Up API Keys**

You will need:

* [GROQ API key](https://console.groq.com/)
* [Google AI Studio API key](https://makersuite.google.com/app/apikey)

Create a `.env` file in the root directory and add your keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

### 5. **Run the App**

Before running the app, run the setup in the notebook (if any). Otherwise, launch the app with:

```bash
streamlit run app.py
```

---

## 🧪 How to Use

1. Upload a PDF file of a patient medical record (e.g., `Sample-filled-in-MR.pdf`).
2. Click **"Creating Vector Store"** to process the document.
3. Ask any medical question in the text box (e.g., "What medication is the patient on?").
4. Get an AI-generated answer, along with the relevant document context.

---

## 🧠 Tech Stack

* **Streamlit** for interactive UI
* **LangChain** for chaining LLM + document retriever
* **GROQ (LLaMA 3.1)** for language generation
* **Google Generative AI Embeddings** for document representation
* **ObjectBox** as a vector store
* **PyPDFLoader** to parse medical PDFs

---

## ⚠️ Disclaimer

> This is an AI-powered demo and not a certified medical diagnostic tool. Please consult licensed healthcare professionals for actual diagnosis or treatment.

