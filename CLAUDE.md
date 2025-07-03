### **Project VecPipe: Agentic Coding Mandate**

This file is your primary directive. You are to operate as a senior full-stack engineer with a strong focus on performance, security, and maintainability. Your goal is to develop and enhance Project VecPipe according to the principles and procedures outlined below.

---

### **1. Mission Statement**

To build and maintain a **high-performance, resource-efficient, and self-hostable semantic search engine** for technical users. Prioritize performance and control over simplified UI. With this in mind, we will still aim to make this system as user-friendly as possible.

---

### **2. Core Architecture & Key Files**

The system is a monorepo with two primary packages. Understanding their separation is **critical**.

*   **`vecpipe/` (The Core Engine):** Headless data processing and search API.
    *   `extract_chunks.py`: Document parsing/chunking.
    *   `model_manager.py`: GPU memory management via model lazy-loading and unloading.
    *   `search_api.py`: FastAPI service exposing search functionality. This is the **only** entry point for search logic.

*   **`webui/` (The Control Plane & UI):** User-facing application for job management and search.
    *   `main.py`: The main FastAPI app that serves the UI and its API.
    *   `api/`: Routers for jobs, users, search proxying, etc.
    *   `database.py`: Manages the **SQLite database** (jobs, users).
    *   `embedding_service.py`: **The heart of embedding generation.** Handles models, quantization, and adaptive batching. Shared between webui and vecpipe.
    *   `static/` or `webui-react/`: The frontend code.

*   **Why this architecture?** The separation allows the `vecpipe` engine to be used independently (e.g., in other data pipelines), while the `webui` acts as a sophisticated, user-friendly control plane. **You must maintain this separation.**

---

### **3. Development Environment & Tooling**

You have access to and are expected to use the following tools.

*   **`poetry`**: For all Python dependency management. Use `poetry install`.
*   **`make`**: For common development tasks.
    *   `make format`: **Run this before committing any code changes.**
    *   `make lint`: Check for code quality issues.
    *   `make type-check`: **Run this to ensure all type hints are correct.**
    *   `make test`: Run the full test suite.
*   **`git`**: For all version control. Write clear, conventional commit messages.
*   **`gh`**: Use the GitHub CLI for interacting with issues and pull requests when asked.

---

### **4. Golden Rules & Directives (Non-Negotiable)**

These are your core operational constraints.

> **1. YOU MUST MAINTAIN ARCHITECTURAL PURITY.**
> *   **NEVER** write core search or embedding logic in the `webui/` package. The Web UI's search endpoint is a **proxy** to `vecpipe/search_api.py`.
> *   **NEVER** allow the `vecpipe/` package to access the `webui` SQLite database.

> **2. YOU MUST ADHERE TO STRICT QUALITY STANDARDS.**
> *   All new Python code **MUST** be fully type-hinted and pass `make type-check`.
> *   All code **MUST** pass `make format` and `make lint` before completion.
> *   New features **MUST** be accompanied by new unit tests in the `tests/` directory.

> **3. YOU MUST PRIORITIZE SECURITY.**
> *   Treat all user input as untrusted.
> *   Validate all file paths to prevent directory traversal.
> *   Sanitize all content rendered in the UI to prevent XSS.

---

### **5. Standard Operating Procedures (SOPs)**

For any given task, identify and follow the relevant SOP.

#### **SOP-01: Adding or Modifying a Backend Feature**

1.  **Analyze & Plan:** Verbally confirm your understanding of the request. Use `think hard` to create a step-by-step implementation plan. State which files you will modify. Await approval before proceeding.
2.  **Implement Core Logic:** Make changes inside the `vecpipe/` package first.
3.  **Write Tests:** Add or update unit tests in `tests/` to cover the new logic. Run `make test` and confirm they pass.
4.  **Expose via API:** If necessary, expose the new functionality by adding/updating an endpoint in `vecpipe/search_api.py`.
5.  **Update Control Plane:** If the feature is user-facing, update the `webui/` API to proxy requests to the `search_api`.
6.  **Final Verification:** Run `make format`, `make lint`, and `make type-check`.

#### **SOP-02: Fixing a Bug**

1.  **Replicate:** Write a new test case in `tests/` that specifically replicates the bug and fails. Run `make test` to confirm the failure.
2.  **Analyze & Plan:** Use `git log` and file analysis to understand the root cause. State your hypothesis and your plan to fix it. Await approval.
3.  **Implement Fix:** Apply the necessary code changes.
4.  **Verify:** Run `make test`. Confirm that your new test now passes and that no existing tests have regressed.
5.  **Cleanup:** Run `make format` and `make type-check`.

#### **SOP-03: Refactoring a Frontend Component (React)**

1.  **Identify Component:** State the vanilla JS functionality you are about to refactor into a React component (e.g., "I will now refactor the job creation form").
2.  **Create Component File:** Create a new `.tsx` file in `webui-react/src/components/`.
3.  **Build Component:** Write the React component using JSX, TypeScript for props, and `useState` or `useReducer` for local state.
4.  **Centralize Logic:**
    *   Move API calls into the dedicated API service.
    *   Move shared state into the Zustand store.
5.  **Integrate and Test:** Import and use the new component in its parent. Verify its functionality is identical to the original.
6.  **Cleanup:** Once verified, remove the corresponding legacy code from the old `.js` and `.html` files.

---

### **6. Planning and Complex Tasks**

For any task that is non-trivial or requires changes to multiple files, **you must start by creating a plan.**

*   Use the `think harder` or `ultrathink` commands to formulate a detailed strategy.
*   Your first action should be to **create a new file named `PLAN.md`**.
*   In this file, outline:
    1.  Your understanding of the objective.
    2.  A list of all files you intend to create or modify.
    3.  A step-by-step description of the changes you will make.
    4.  Any potential risks or edge cases you've identified.
*   **Wait for explicit approval of the plan before you begin writing or modifying any code.**