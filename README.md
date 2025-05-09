# Local ChatPDF with DeepSeek R1

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

## Features

- **PDF Upload**: Upload one or multiple PDF documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

---

## Installation

Follow the steps below to set up and run the application:

### 1. Request access to Tenstorrent on Koyeb

Get the details on the private preview of Tenstorrent on Koyeb in [this blog](https://www.koyeb.com/blog/tenstorrent-cloud-instances-unveiling-next-gen-ai-accelerators).
Request access to the preview [here](https://tally.so/r/npRak8). I'm the one who approves access, so be sure to give a good reason for using it or your request will be rejected.

### 2. Deploy a vLLM inference server to a Tenstorrent instance

Use the **Deploy to Koyeb** button in [this README](https://github.com/koyeb/tenstorrent-examples/tree/main/tt-models#readme) to deploy  `deepseek-ai/DeepSeek-R1-Distill-Llama-8b`
on a vLLM inference server.

### 3. Clone the Repository

```bash
git clone https://github.com/paquino11/chatpdf-rag-deepseek-r1.git
cd chatpdf-rag-deepseek-r1
```

### 4. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure to include the following packages in your `requirements.txt`:

```
streamlit
langchain
langchain_ollama
langchain_community
streamlit-chat
pypdf
chromadb
sentence-transformers
openai
```

### 7. Pull Required Models for Ollama

To use the specified embedding model (`mxbai-embed-large`), download it via the `ollama` CLI:

```bash
ollama pull mxbai-embed-large
```

### 8. Create .env file

At the root of your project, create a .env file and add your OpenAI API key and the URL for your deployed web service on Koyeb:

OPENAI_API_KEY=<YOUR_API_KEY>
DEEPSEEK_API_URL=https://<YOUR_SERVICE_URL>.koyeb.app/v1


---

## Usage

### 1. Start the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### 2. Upload Documents

- Navigate to the **Upload a Document** section in the web interface.
- Upload one or multiple PDF files to process their content.
- Each file will be ingested automatically and confirmation messages will show processing time.

### 3. Ask Questions

- Type your question in the chat input box and press Enter.
- Adjust retrieval settings (`k` and similarity threshold) in the **Settings** section for better responses.

---

## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── rag.py                  # Core RAG logic for PDF ingestion and question-answering
├── requirements.txt        # List of required Python dependencies
├── chroma_db/              # Local persistent vector store (auto-generated)
└── README.md               # Project documentation
```

---

## Configuration

You can modify the following parameters in `rag.py` to suit your needs:

1. **Models**:
   - Default LLM: `deepseek-r1:latest` (7B parameters)
   - Default Embedding: `mxbai-embed-large` (1024 dimensions)
   - Change these in the `ChatPDF` class constructor or when initializing the class
   - Any Ollama-compatible model can be used by updating the `llm_model` parameter

2. **Chunking Parameters**:
   - `chunk_size=1024` and `chunk_overlap=100`
   - Adjust for larger or smaller document splits

3. **Retrieval Settings**:
   - Adjust `k` (number of retrieved results) and `score_threshold` in `ask()` to control the quality of retrieval.

---

## Requirements

- **Python**: 3.8+
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **PyPDF**: For PDF document processing.
- **ChromaDB**: Vector store for document embeddings.

---

## Troubleshooting

### Common Issues

1. **Missing Models**:
   - Ensure you've pulled the required models using `ollama pull`.

2. **Vector Store Errors**:
   - Delete the `chroma_db/` directory if you encounter dimensionality errors:
     ```bash
     rm -rf chroma_db/
     ```

3. **Streamlit Not Launching**:
   - Verify dependencies are installed correctly using `pip install -r requirements.txt`.

---

## Future Enhancements

- **Memory Integration**: Add persistent memory to maintain conversational context across sessions.
- **Advanced Analytics**: Include visual insights from retrieved content.
- **Expanded Model Support**: Support additional embedding and LLM providers.

---

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

## Acknowledgments

- [Tenstorrent](https://github.com/tenstorrent)
- [Koyeb](https://github.com/koyeb)
- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)

