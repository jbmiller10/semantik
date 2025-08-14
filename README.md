🚨🚨🚨

** Dev's Note:**  Semantik is currently in a **pre-release state**, and under heavy development. While you are encouraged to use it and file issues and feature requests, please be aware that you will likely encounter rough-edges and bugs, and we do not yet recommend its usage in a production environment. 

While we will do our best to avoid any breaking changes, we do not gurantee backwards compatibility between releases at this time. This is in the interest of providing the strongest long-term foundation for Semantik by the time of release.


🚨🚨🚨

---


# Semantik - Easy, Private, and Powerful Document Search

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Semantik is a self-hosted search engine that turns your personal or team's file server into a powerful, private knowledge base. Find what you need instantly, without your data ever leaving your hardware.**

## Key Features

<details>
<summary><strong> 100% Private & Self-Hosted</strong></summary>
<br>
Your data and models stay on your hardware. Semantik makes zero external API calls. It's the perfect solution for searching sensitive work documents, research papers, or personal files.
</details>

<details>
<summary><strong> State-of-the-Art Search Intelligence</strong></summary>
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
<summary><strong> Complete Control & Tunability</strong></summary>
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
<summary><strong> Easy-to-Use Interface</strong></summary>
<br>
A powerful backend deserves a clean frontend.
<ul>
  <li><b>Intuitive UI:</b> Easily create and manage your search collections.</li>
  <li><b>Real-time Monitoring:</b> Watch your documents get processed with detailed operation tracking.</li>
  <li><b>Document Viewer:</b> Preview your search results directly in the app.</li>
</ul>
</details>



##  System Requirements

Semantik can run on both CPU and GPU, but a CUDA-compatible GPU is **strongly recommended** for reasonable performance.

<br>

<details>
<summary><strong> Recommended (GPU Setup)</strong></summary>
<br>

- **GPU**: NVIDIA GPU with CUDA support & **8GB+ of VRAM** (e.g., RTX 3060 / 4060 or newer).
- **CPU**: Modern quad-core or better.
- **RAM**: 16GB+.
- **Storage**: SSD for model storage with at least 30GB of free space; HDD is fine for documents
- **Software**: Python 3.12, Cuda Toolkit, Docker & Docker Compose

> **Why the GPU?** Creating search embeddings is a highly parallel task. A GPU can be 10-100x faster than a CPU, turning a process that takes hours into one that takes minutes.

> **CUDA Setup Issues?** If you encounter CUDA library errors, see our [Troubleshooting Guide](TROUBLESHOOTING.md#cuda-library-issues-on-linux).

</details>

<br>

<details>
<summary><strong> (CPU-Only Setup)</strong></summary>
<br>

- **CPU**: Modern quad-core or better.
- **RAM**: 16GB+.
- **Storage**: SSD for model storage with at least 30GB of free space; HDD is fine for documents
- **Software**: Python 3.12, Docker & Docker Compose.

> **Performance Note:** A CPU-only setup will be **very slow** for indexing and search. It is suitable for evaluation and small personal document collections only.

</details>

## Quick Start

### Option 1: Interactive Setup Wizard (Recommended)
<details>
<summary>The easiest way to get started is using our interactive setup wizard.</summary>

The setup wizard guides you through the entire configuration process with an intuitive text-based interface.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jbmiller10/semantik.git
   cd semantik
   ```

2. **Run the wizard:**
   
   **Linux/macOS:**
   ```bash
   make wizard
   ```
   
   **Windows:**
   ```cmd
   python wizard_launcher.py
   ```
   
   The wizard will automatically:
   - Work on Windows, macOS, and Linux
   - Install Poetry if needed
   - Install all required dependencies
   - Launch the interactive setup

3. **Follow the interactive prompts:**
   - Choose between **Quick Setup** (recommended) or **Custom Setup**
   - Select your document directories with the built-in file browser
   - The wizard auto-detects GPU availability and configures accordingly
   - All security keys are generated automatically

4. **That's it!** The wizard will:
   - Check system requirements
   - Configure GPU or CPU mode automatically
   - Set up your document directories
   - Generate secure configurations
   - Start all services
   - Provide a service monitor for ongoing management

> ** Tip:** If you've already configured Semantik, running `make wizard` again will take you directly to the service monitor where you can start, stop, and manage your services.

</details>

### Option 2: Manual Docker Setup 
<details>
<summary>For users who prefer manual configuration or need specific customizations.</summary>

> ** GPU Note:** The default Docker configuration uses GPU acceleration for optimal performance. Ensure you have NVIDIA Docker runtime installed. For CPU-only systems, see the CPU-only option below.

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
   # Or directly: docker compose up -d
   # Note: Default configuration works for both GPU and CPU. PyTorch will automatically use CPU if no GPU is available.
   ```

4. **Access Semantik:**
   Open your browser to http://localhost:8080

That's it!  Semantik is now running with:
- **WebUI** on port 8080
- **Search API** on port 8001
- **Qdrant** vector database on port 6333

**Additional Docker Options:**
- **Production**: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
- **CUDA-enabled GPU**: `docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d`
</details>

### Option 3: Development Setup (Shell Scripts)

<details>
<summary>For developers who prefer traditional shell scripts</summary>

1. **Prerequisites:**
   - Python 3.11+
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




## 📖 Documentation

- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Start here! Complete guide to all documentation
- **[API Reference](docs/API_REFERENCE.md)** - Full REST and WebSocket API documentation
- **[Architecture Guide](docs/ARCH.md)** - System design and component relationships
- **[Configuration Guide](docs/CONFIGURATION.md)** - All environment variables and settings

### Quick Links by Role

**For Users:**
- [Collection Management](docs/COLLECTION_MANAGEMENT.md) - Creating and managing document collections
- [Search Guide](docs/SEARCH_SYSTEM.md) - Advanced search features and optimization

**For Developers:**
- [Local Development](docs/local-development.md) - Setting up your development environment
- [Testing Guide](docs/TESTING.md) - Writing and running tests

**For Operations:**
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment best practices
- [Monitoring Guide](docs/HEALTH_MONITORING.md) - System health and performance monitoring

##  Troubleshooting Common Issues

<details>
<summary><strong>GPU Not Detected</strong></summary>
<br>

**Symptoms**: Semantik falls back to CPU mode despite having a GPU

**Solutions**:
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. For Docker: Ensure nvidia-docker2 is installed
4. Check GPU memory: Minimum 4GB VRAM required

```bash
# Test GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```
</details>

<details>
<summary><strong>Out of Memory Errors</strong></summary>
<br>

**Symptoms**: "CUDA out of memory" or system freezes during processing

**Solutions**:
1. Use smaller models: `Qwen/Qwen3-Embedding-0.6B`
2. Enable quantization: Set `DEFAULT_QUANTIZATION=int8`
3. Reduce batch size: Lower `BATCH_SIZE` in configuration
4. Enable model auto-unloading: Set `MODEL_UNLOAD_AFTER_SECONDS=60`

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi
```
</details>

<details>
<summary><strong>Slow Search Performance</strong></summary>
<br>

**Symptoms**: Searches take several seconds to complete

**Solutions**:
1. Ensure Qdrant indexes are built (happens automatically after 20k vectors)
2. Disable reranking for simple searches
3. Check system resources during search
4. Consider using SSD for Qdrant storage

```bash
# Check Qdrant collection info
curl http://localhost:6333/collections/work_docs
```
</details>

<details>
<summary><strong>WebSocket Connection Issues</strong></summary>
<br>

**Symptoms**: Real-time progress updates not working

**Solutions**:
1. Check browser console for WebSocket errors
2. Ensure no proxy is blocking WebSocket connections
3. Verify ports 8080 and 8001 are accessible
4. Try disabling browser extensions

```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8080/ws/test');
ws.onopen = () => console.log('Connected');
ws.onerror = (e) => console.error('Error:', e);
```
</details>

For more troubleshooting tips, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).


##  License

Semantik is licensed under the [GNU Affero General Public License v3.0](LICENSE). The core of this license is that if you use this code (or a modified version of it) to provide a service over a network, you must also make the source code available. This ensures that the project and its derivatives will always remain open source.



## **Disclosure**

AI tooling was used in the development of this repo. All code was reviewed and tested by a human operator.


## **Roadmap**
<ul>
<li><b>SPLADE sparse indexing for hybrid search</b>
<li><b>Multimodality & support for additional file formats:</b> index images, code, and more </li>
<li><b>Integrated OCR</b>
<li><b>Support for additional models:</b>Use any SentenceTransformers compatible model</li>
<li><b>Integrated Model Context Protocol Server</b>: Seamlessly bring the knowledge of your corpus to the LLM of your choice.</li>
<li><b>Contextually aware document chunking:</b></li>
<ul>


