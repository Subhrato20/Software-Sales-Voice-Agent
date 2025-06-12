# CS614 Final Project: AI Software Sales Agent

This project implements a sophisticated voice-based AI Software Sales Agent. The agent is designed to interact with prospective customers over the phone, understand their needs, and provide information about software services by leveraging a Retrieval-Augmented Generation (RAG) system.

The core of the system is built with Retell AI for real-time voice conversation, OpenAI's GPT models for language understanding and generation, and a Milvus vector database for efficient knowledge retrieval.

## Features

- **Real-time Voice Interaction**: Low-latency, natural-sounding voice conversations powered by Retell AI.
- **Advanced Conversational AI**: Utilizes OpenAI's GPT-4o for intelligent dialogue, comprehension, and response generation.
- **Function Calling**: Employs LLM function calling to dynamically search the knowledge base or end the call gracefully.
- **Retrieval-Augmented Generation (RAG)**: Answers user questions by retrieving relevant information from a local knowledge base (`text_data/`), ensuring accurate and contextually-aware responses.
- **Vector Search**: Leverages a Milvus vector database for fast and efficient semantic search over the knowledge base documents.
- **Long-Term Memory**: Remembers context from previous conversations with a user, powered by Mem0.

## Directory Structure

```
└── subhrato20-cs614-final-project/
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── server.py
    ├── Agent/
    │   ├── custom_types.py
    │   └── llm_with_func.py
    ├── processed_data/
    ├── RAG/
    │   ├── deletedb.py
    │   ├── embedding.py
    │   ├── milvus_init.py
    │   ├── milvus_search.py
    │   ├── milvus_upsert.py
    │   └── test.py
    └── text_data/
        ├── ... (knowledge base files)
```

## Prerequisites

- Python 3.10 or higher.
- Docker and Docker Compose (for running the Milvus database).
- API keys and credentials for the following services:
  - [Retell AI](https://www.retellai.com/)
  - [OpenAI](https://openai.com/)
  - [Mem0](https://mem0.ai/)

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/subhrato20/cs614-final-project.git
cd cs614-final-project
```

### Step 2: Set Up Milvus Vector Database

This project requires a running Milvus instance. The easiest way to set this up is using Docker.

1. Create a `docker-compose.yml` file in the root of the project with the following content:

   ```yaml
   # docker-compose.yml
   version: '3.5'

   services:
     milvus:
       container_name: milvus_standalone
       image: milvusdb/milvus:v2.4.4
       ports:
         - "19530:19530"
         - "9091:9091"
       volumes:
         - ./milvus_data:/var/lib/milvus
       environment:
         ETCD_USE_EMBED: "true"
         ETCD_DATA_DIR: "/var/lib/milvus/etcd"
   ```

2. Start the Milvus server:
   ```bash
   docker-compose up -d
   ```
   This will start a Milvus instance and expose it on port `19530`.

### Step 3: Configure Environment Variables

Create a `.env` file in the root of the project. You can copy the example below and fill in your actual credentials.

```dotenv
# .env

# Retell AI API Key
RETELL_API_KEY="your_retell_api_key"

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key"

# Optional: Specify the model, defaults to gpt-4o
# GEMINI_MODEL="gpt-4o"

# Mem0 API Key and Configuration for long-term memory
MEM0_API_KEY="your_mem0_api_key"
MEM0_ORG_ID="your_mem0_org_id"
MEM0_PROJECT_ID="your_mem0_project_id"

# Milvus Configuration (should match the Docker setup)
MILVUS_URI="http://localhost:19530"
RAG_COLLECTION_NAME="Software_Sales"
```

### Step 4: Install Dependencies

It is highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the required Python packages
pip install -r requirements.txt
```

## How to Run the System (Reproducing Results)

Follow these steps in order to populate the knowledge base and start the agent server.

### Step 1: Prepare and Embed the Knowledge Base

This script processes the `.txt` files in `text_data/`, generates embeddings for them, and saves the output to `processed_data/`.

```bash
python RAG/embedding.py
```

### Step 2: Initialize the Milvus Collection

This script creates the specified collection (`Software_Sales`) in your running Milvus instance.

```bash
python RAG/milvus_init.py
```

### Step 3: Upsert Data into Milvus

This script loads the processed embeddings and text from `processed_data/` and inserts them into the Milvus collection.

```bash
python RAG/milvus_upsert.py
```

**Optional**: To clear the database and start fresh, you can run:
```bash
python RAG/deletedb.py
```

### Step 4: Start the Agent Server

The server uses Uvicorn to run the FastAPI application, which listens for incoming WebSocket connections from Retell.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
The server will now be running on `http://localhost:8000`.

### Step 5: Connect with Retell AI

The agent is designed to be connected to the Retell AI platform.

1.  **Expose Your Local Server**: Since the server is running locally, you need to expose it to the public internet so Retell's servers can connect to it. Use a tool like [ngrok](https://ngrok.com/).
    ```bash
    ngrok http 8000
    ```
    ngrok will provide you with a public URL (e.g., `https://<random-string>.ngrok-free.app`).

2.  **Configure Your Retell Agent**:
    - Log in to your Retell AI dashboard.
    - Create a new agent.
    - In the agent's configuration, set the **LLM WebSocket URL** to the public URL from ngrok, pointing to your WebSocket endpoint. The format should be:
      `wss://<your-ngrok-url>/llm-websocket/{call_id}`
      (Replace `<your-ngrok-url>` with the one provided by ngrok, removing the `https://` prefix).
    - Configure other agent settings as desired (e.g., voice).

3.  **Make a Call**:
    - Link a phone number to your agent in the Retell dashboard.
    - Call the phone number.
    - You should now be connected to your local AI agent. You can ask it questions about the services detailed in the `text_data/` directory, such as:
      - *"What services do you offer?"*
      - *"Tell me more about Custom Software Development."*
      - *"What are your prices for DevOps implementation?"*

## Additional Libraries and Dependencies

All required Python libraries are listed in `requirements.txt`. The core components of this project rely on the following key libraries:

-   **Backend & WebSockets**:
    -   `fastapi`: A modern, high-performance web framework for building APIs.
    -   `uvicorn`: An ASGI server for running FastAPI.
-   **AI & Voice**:
    -   `retell-sdk`: The official SDK for integrating with Retell AI's voice platform.
    -   `openai`: The official SDK for interacting with OpenAI's language models (GPT).
-   **RAG & Vector Database**:
    -   `pymilvus`: The client for interacting with the Milvus vector database.
    -   `transformers` & `torch`: Used to load models from Hugging Face for generating text embeddings and performing re-ranking.
-   **Memory & Configuration**:
    -   `mem0`: A client for a long-term memory service.
    -   `python-dotenv`: For loading environment variables from a `.env` file.

In addition to Python libraries, a running instance of **Milvus** is a critical dependency, which can be managed via Docker as described in the setup instructions.

## Utility Scripts

-   **`RAG/test.py`**: A standalone script to test the search and re-ranking functionality of the RAG pipeline directly, without needing to run the full server. This is useful for debugging. *Note: you may need to adjust the `COLLECTION_NAME` inside the script to match your setup (`Software_Sales`).*
-   **`RAG/deletedb.py`**: A utility to drop the Milvus collection, allowing you to re-initialize the database from scratch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
