**Note:** Semantik is currently in a **pre-release state**. While you are encouraged to use it and file issues and feature requests, please be aware that you will likely encounter rough-edges and bugs, and we do not yet recommend its usage in a production environment.

# Semantik 🚀 - Private, Powerful, and Controllable Document Search

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Semantik is a self-hosted search engine that turns your personal or team's file server into a powerful, private knowledge base. Find what you need instantly, without your data ever leaving your hardware.**

No cloud APIs, no subscriptions, no vendor lock-in. Just powerful, open-source search you control completely.

![Semantik Dashboard](docs/images/semantik-dashboard.png)

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

## ⚡ Quick Start (5 minutes)

All you need is Docker and Docker Compose.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/semantik.git
cd semantik

# 2. Create a directory for your documents on your host machine
mkdir -p ./my_documents

# 3. Create your .env file
cp .env.example .env
# (No changes are needed to get started)

# 4. Start the services!
docker-compose up -d

# 5. Access the Web UI
# Open your browser to http://localhost:8080
```

Once running, use the UI to create your first collection pointing to the `/data` directory (which is mapped to the `my_documents` folder you created).

## 🚀 Deployment

The recommended deployment method is using `docker-compose`. See the [Quick Start](#-quick-start-5-minutes) guide above. For manual installation and production deployment guidance, see our [Installation Guide](docs/installation.md).

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

---

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
