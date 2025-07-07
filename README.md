🚨🚨🚨

**Dev's Note:**  Semantik is currently in a **pre-release state**, and under heavy development. While you are encouraged to use it and file issues and feature requests, please be aware that you will likely encounter rough-edges and bugs, and we do not yet recommend its usage in a production environment. 

While we will do our best to avoid any breaking changes, we do not gurantee backwards compatibility between releases at this time. This is in the interest of providing the strongest long-term foundation for Semantik by the time of release.

🚨🚨🚨

# Semantik 🔎 - Easy, Private, and Powerful Document Search

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Semantik is a self-hosted search engine that turns your personal or team's file server into a powerful, private knowledge base. Find what you need instantly, without your data ever leaving your hardware.**

No cloud APIs, no subscriptions, no vendor lock-in. Just powerful, open-source search you control completely.

## Why Semantik? Find Your Files Without Sacrificing Privacy.

Commercial search solutions are powerful, but they require you to upload your sensitive data to their servers. Open-source libraries are private, but they require you to build an entire application from scratch.

Semantik gives you the best of both worlds: a production-ready application that runs entirely on your own hardware.

| Semantik is for you if... | You might not need Semantik if... |
| :--- | :--- |
| ✅ You have a large collection of documents on a NAS or local server. | ❌ All your data is already in a cloud service you like. |
| ✅ You value privacy and want to keep your data 100% in-house. | ❌ You prefer a fully managed, hands-off solution. |
| ✅ You want the power of modern AI-powered search. | ❌ You only need simple keyword search. |
| ✅ You're a technical user who likes to tune and control your software. | ❌ You don't want to manage your own server/application. |

## 🌟 Key Features

<details>
<summary><strong>🔒 100% Private & Self-Hosted</strong></summary>
<br>
Your data and models stay on your hardware. Semantik makes zero external API calls. It's the perfect solution for searching sensitive work documents, research papers, or personal files.
</details>

<details>
<summary><strong>🧠 State-of-the-Art Search Intelligence</strong></summary>
<br>
Go beyond simple keyword matching. Semantik uses top-performing open-source AI models to understand the <i>meaning</i> of your documents.
<ul>
  <li><b>Semantic Search:</b> Find concepts, not just words.</li>
  <li><b>Cross-Encoder Reranking:</b> A two-stage search process delivers highly accurate and relevant results.</li>
  <li><b>Hybrid Search:</b> Combines the best of vector and keyword search for ultimate precision.</li>
  <li><b>Integrated RAG/Model Contex Protocol Server(Coming Soon):</b> Optionally, give an LLM of your choice the ability to reference and chat about your documents.</li>
</ul>
</details>

<details>
<summary><strong>🔧 Complete Control & Tunability</strong></summary>
<br>
Semantik is built for users who want to be in the driver's seat.
<ul>
  <li><b>Model Selection:</b> Choose from a wide range of open-source embedding models.</li>
  <li><b>Resource Management:</b> Tune performance for your specific hardware with quantization (float32, float16, int8) and adaptive batching.</li>
  <li><b>Intelligent GPU Handling:</b> Features automatic model loading/unloading to efficiently manage VRAM.</li>
  <li><b>Headless API:</b> The search engine can be run independently and integrated into your own scripts and pipelines.</li>
</ul>
</details>

<details>
<summary><strong>✨ Easy-to-Use Interface</strong></summary>
<br>
A powerful backend deserves a clean frontend.
<ul>
  <li><b>Intuitive UI:</b> Easily create and manage your search collections.</li>
  <li><b>Real-time Monitoring:</b> Watch your documents get processed with a detailed job monitor.</li>
  <li><b>Document Viewer:</b> Preview your search results directly in the app.</li>
</ul>
</details>



## ⚙️ System Requirements

Semantik can run on both CPU and GPU, but a CUDA-compatible GPU is **strongly recommended** for reasonable performance.

<br>

<details>
<summary><strong>✅ Recommended (GPU Setup)</strong></summary>
<br>

- **GPU**: NVIDIA GPU with CUDA support & **8GB+ of VRAM** (e.g., RTX 3060 / 4060 or newer).
- **CPU**: Modern quad-core or better.
- **RAM**: 16GB+.
- **STORAGE**:  SSD for model storage; HDD is fine for documents
- **Software**: Python 3.12, Cuda Toolkit, Docker & Docker Compose

> **Why the GPU?** Creating search embeddings is a highly parallel task. A GPU can be 10-100x faster than a CPU, turning a process that takes hours into one that takes minutes.

> **CUDA Setup Issues?** If you encounter CUDA library errors, see our [Troubleshooting Guide](TROUBLESHOOTING.md#cuda-library-issues-on-linux).

</details>

<br>

<details>
<summary><strong>⚠️ Minimum (CPU-Only Setup)</strong></summary>
<br>

- **CPU**: Modern quad-core or better.
- **RAM**: 16GB+.
- **Storage**: SSD for model storage; HDD is fine for documents
- **Software**: Python 3.12, Docker & Docker Compose.

> **Performance Note:** A CPU-only setup will be **very slow** for indexing and search. It is suitable for evaluation and small personal document collections only.

</details>

## ⚡ Quick Start

### Option 1: Docker Compose (Recommended) 🐳

The fastest way to get started with Semantik is using Docker Compose.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jbmiller10/semantik.git
   cd semantik
   ```

2. **Set up your environment:**
   ```bash
   cp .env.docker.example .env
   # Edit .env with your configuration
   # At minimum, change JWT_SECRET_KEY for security
   ```

3. **Start all services:**
   ```bash
   make docker-up
   # Or directly: docker-compose up -d
   ```

4. **Access Semantik:**
   Open your browser to http://localhost:8080

That's it! 🎉 Semantik is now running with:
- **WebUI** on port 8080
- **Search API** on port 8000
- **Qdrant** vector database on port 6333

**Additional Docker Options:**
- **GPU Support**: `docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
- **Production**: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

### Option 2: Development Setup (Shell Scripts)

<details>
<summary>For developers who prefer traditional shell scripts</summary>

1. **Prerequisites:**
   - Python 3.12+
   - Node.js 18+
   - Running Qdrant instance
   - Poetry for Python dependency management

2. **Setup:**
   ```bash
   poetry install
   make frontend-build
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services:**
   ```bash
   ./start_all_services.sh
   ```

4. **Other commands:**
   ```bash
   ./status_services.sh  # Check service status
   ./stop_all_services.sh  # Stop all services
   ./restart_all_services_rebuild.sh  # Rebuild and restart
   ```
</details>


## 🤝 Semantik vs. The Alternatives

| Feature | Semantik | Elasticsearch | Pinecone | ChromaDB |
| :--- | :---: | :---: | :---: | :---: |
| **Self-hosted & Private** | ✅ | ✅ | ❌ | ✅ |
| **Intelligent GPU Management** | ✅ | ❌ | N/A | ❌ |
| **Advanced Search UI Out-of-the-Box** | ✅ | ❌ | ✅ | ❌ |
| **Built-in Reranking & Hybrid Search** | ✅ | Plugin | ❌ | ❌ |
| **Truly Open Source (AGPL)** | ✅ | ❌ | ❌ | ✅ |

## 📄 License

Semantik is licensed under the [GNU Affero General Public License v3.0](LICENSE). The core of this license is that if you use this code (or a modified version of it) to provide a service over a network, you must also make the source code available. This ensures that the project and its derivatives will always remain open source.



## **Disclosure**

AI tooling was used in the development of this repo. All code was reviewed and tested by a human operator.


## **Roadmap**
<ul>
<li><b>Multimodality & support for additional file formats:</b> index images, code, and more </li>
<li><b>Integrated OCR:</b> Seamlessly extract and index text from pdfs or images. </li>
<li><b>Support for additional models:</b>Use any SentenceTransformers compatible model</li>
<li><b>Integrated Model Context Protocol Server</b>: Seamlessly bring the knowledge of your corpus to the LLM of your choice.</li>
<li><b>Contextually aware document chunking:</b></li>
<ul>


<p align="center">
  ⭐ If you find Semantik useful, please consider starring the repository! ⭐
</p>
<p align="center">
  <a href="https://github.com/jbmiller10/semantik/issues">Report a Bug</a> •
  <a href="https://github.com/jbmiller10/semantik/discussions">Request a Feature</a>
</p>
