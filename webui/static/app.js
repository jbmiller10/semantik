// Global state
let scannedFiles = [];
let activeWebSocket = null;

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active state from buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('text-blue-600', 'border-blue-600');
        btn.classList.add('text-gray-700', 'border-transparent');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Highlight active button
    const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
    activeBtn.classList.remove('text-gray-700', 'border-transparent');
    activeBtn.classList.add('text-blue-600', 'border-blue-600');
    
    // Load data for specific tabs
    if (tabName === 'jobs') {
        refreshJobs();
    } else if (tabName === 'search') {
        loadJobCollections();
    }
}

// Model selection handler
document.getElementById('model-name').addEventListener('change', function(e) {
    const customInput = document.getElementById('custom-model');
    if (e.target.value === 'custom') {
        customInput.style.display = 'block';
        customInput.required = true;
    } else {
        customInput.style.display = 'none';
        customInput.required = false;
    }
});

// Scan directory
async function scanDirectory() {
    const path = document.getElementById('directory-path').value;
    if (!path) {
        alert('Please enter a directory path');
        return;
    }
    
    try {
        const response = await fetch('/api/scan-directory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: path, recursive: true })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }
        
        const data = await response.json();
        scannedFiles = data.files;
        displayScanResults(data);
    } catch (error) {
        alert(`Scan failed: ${error.message}`);
    }
}

// Display scan results
function displayScanResults(data) {
    const resultsDiv = document.getElementById('scan-results');
    const summaryDiv = document.getElementById('scan-summary');
    const fileListDiv = document.getElementById('file-list');
    
    resultsDiv.classList.remove('hidden');
    
    // Summary
    const totalSize = data.files.reduce((sum, f) => sum + f.size, 0);
    summaryDiv.innerHTML = `
        <p class="font-semibold">Found ${data.count} supported files</p>
        <p class="text-sm text-gray-600">Total size: ${formatBytes(totalSize)}</p>
        <p class="text-sm text-gray-600">File types: ${getFileTypes(data.files).join(', ')}</p>
    `;
    
    // File list
    fileListDiv.innerHTML = data.files.map(file => `
        <div class="text-sm py-1 hover:bg-gray-50 px-2 rounded">
            <i class="fas fa-file mr-1 text-gray-400"></i>
            ${file.path}
            <span class="text-gray-500">(${formatBytes(file.size)})</span>
        </div>
    `).join('');
}

// Create job form submission
document.getElementById('create-job-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (scannedFiles.length === 0) {
        alert('Please scan a directory first');
        return;
    }
    
    const modelSelect = document.getElementById('model-name');
    const modelName = modelSelect.value === 'custom' 
        ? document.getElementById('custom-model').value 
        : modelSelect.value;
    
    const jobData = {
        name: document.getElementById('job-name').value,
        description: document.getElementById('job-description').value,
        directory_path: document.getElementById('directory-path').value,
        model_name: modelName,
        chunk_size: parseInt(document.getElementById('chunk-size').value),
        chunk_overlap: parseInt(document.getElementById('chunk-overlap').value),
        batch_size: 96
    };
    
    try {
        const response = await fetch('/api/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }
        
        const job = await response.json();
        alert(`Job created successfully! ID: ${job.id}`);
        
        // Switch to jobs tab
        switchTab('jobs');
        
        // Connect to WebSocket for updates
        connectToJobWebSocket(job.id);
        
    } catch (error) {
        alert(`Failed to create job: ${error.message}`);
    }
});

// Refresh jobs list
async function refreshJobs() {
    try {
        const response = await fetch('/api/jobs');
        const jobs = await response.json();
        
        const jobsList = document.getElementById('jobs-list');
        
        if (jobs.length === 0) {
            jobsList.innerHTML = '<p class="text-gray-500 text-center py-8">No jobs found</p>';
            return;
        }
        
        jobsList.innerHTML = jobs.map(job => `
            <div class="border border-gray-200 rounded-lg p-4">
                <div class="flex justify-between items-start mb-2">
                    <div>
                        <h3 class="font-semibold text-lg">${job.name}</h3>
                        <p class="text-sm text-gray-600">${job.directory_path}</p>
                    </div>
                    <span class="px-3 py-1 rounded-full text-sm ${getStatusBadgeClass(job.status)}">
                        ${job.status}
                    </span>
                </div>
                
                <div class="grid grid-cols-3 gap-4 mb-3">
                    <div>
                        <p class="text-sm text-gray-500">Total Files</p>
                        <p class="font-medium">${job.total_files}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Processed</p>
                        <p class="font-medium">${job.processed_files}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Failed</p>
                        <p class="font-medium">${job.failed_files}</p>
                    </div>
                </div>
                
                ${job.status === 'processing' ? `
                    <div class="mb-3">
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-blue-600 h-2 rounded-full progress-bar" 
                                 style="width: ${(job.processed_files / job.total_files * 100).toFixed(1)}%"></div>
                        </div>
                        ${job.current_file ? `
                            <p class="text-sm text-gray-600 mt-1">Processing: ${job.current_file.split('/').pop()}</p>
                        ` : ''}
                    </div>
                ` : ''}
                
                ${job.error ? `
                    <div class="bg-red-50 border border-red-200 rounded p-2 mb-3">
                        <p class="text-sm text-red-700">${job.error}</p>
                    </div>
                ` : ''}
                
                <div class="flex justify-between items-center text-sm text-gray-500">
                    <span>Model: ${job.model_name}</span>
                    <span>Created: ${new Date(job.created_at).toLocaleString()}</span>
                </div>
                
                <div class="mt-3 flex space-x-2">
                    ${job.status === 'processing' ? `
                        <button onclick="connectToJobWebSocket('${job.id}')"
                                class="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
                            <i class="fas fa-chart-line mr-1"></i>Monitor
                        </button>
                    ` : ''}
                    ${job.status === 'completed' ? `
                        <button onclick="useJobForSearch('${job.id}', '${job.name}')"
                                class="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700">
                            <i class="fas fa-search mr-1"></i>Search
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load jobs:', error);
    }
}

// Connect to job WebSocket for real-time updates
function connectToJobWebSocket(jobId) {
    if (activeWebSocket) {
        activeWebSocket.close();
    }
    
    const ws = new WebSocket(`ws://${window.location.host}/ws/${jobId}`);
    activeWebSocket = ws;
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
            // Update job progress in UI
            refreshJobs();
        } else if (data.type === 'completed') {
            alert('Job completed successfully!');
            refreshJobs();
            ws.close();
        } else if (data.type === 'error') {
            alert(`Job error: ${data.message}`);
            refreshJobs();
            ws.close();
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        activeWebSocket = null;
    };
}

// Load job collections for search
async function loadJobCollections() {
    try {
        const response = await fetch('/api/jobs');
        const jobs = await response.json();
        
        const select = document.getElementById('search-collection');
        select.innerHTML = '<option value="">Default Collection</option>';
        
        jobs.filter(job => job.status === 'completed').forEach(job => {
            const option = document.createElement('option');
            option.value = job.id;
            option.textContent = job.name;
            select.appendChild(option);
        });
        
    } catch (error) {
        console.error('Failed to load collections:', error);
    }
}

// Use job for search
function useJobForSearch(jobId, jobName) {
    switchTab('search');
    setTimeout(() => {
        document.getElementById('search-collection').value = jobId;
    }, 100);
}

// Search form submission
document.getElementById('search-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const searchData = {
        query: document.getElementById('search-query').value,
        k: parseInt(document.getElementById('search-k').value),
        job_id: document.getElementById('search-collection').value || null
    };
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }
        
        const results = await response.json();
        displaySearchResults(results);
        
    } catch (error) {
        alert(`Search failed: ${error.message}`);
    }
});

// Display search results
function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const container = document.getElementById('results-container');
    
    resultsDiv.classList.remove('hidden');
    
    if (data.results.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-8">No results found</p>';
        return;
    }
    
    container.innerHTML = data.results.map((result, idx) => `
        <div class="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
            <div class="flex justify-between items-start mb-2">
                <h4 class="font-medium">
                    <i class="fas fa-file-alt mr-1 text-gray-400"></i>
                    ${result.payload.path.split('/').pop()}
                </h4>
                <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                    Score: ${result.score.toFixed(3)}
                </span>
            </div>
            
            <p class="text-sm text-gray-600 mb-2">${result.payload.path}</p>
            
            ${result.payload.text ? `
                <div class="bg-gray-50 rounded p-3">
                    <p class="text-sm text-gray-700">${result.payload.text}...</p>
                </div>
            ` : ''}
            
            <div class="mt-2 text-xs text-gray-500">
                Chunk ID: ${result.payload.chunk_id}
            </div>
        </div>
    `).join('');
}

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileTypes(files) {
    const types = new Set(files.map(f => f.extension));
    return Array.from(types);
}

function getStatusBadgeClass(status) {
    const classes = {
        'created': 'bg-gray-100 text-gray-800',
        'scanning': 'bg-yellow-100 text-yellow-800',
        'processing': 'bg-blue-100 text-blue-800',
        'completed': 'bg-green-100 text-green-800',
        'failed': 'bg-red-100 text-red-800'
    };
    return classes[status] || 'bg-gray-100 text-gray-800';
}

function resetForm() {
    document.getElementById('create-job-form').reset();
    document.getElementById('scan-results').classList.add('hidden');
    scannedFiles = [];
}

function closeProgressModal() {
    document.getElementById('progress-modal').classList.add('hidden');
}

// Load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const select = document.getElementById('model-name');
        const deviceInfo = document.getElementById('device-info');
        
        // Show device info
        deviceInfo.textContent = `(Running on ${data.current_device.toUpperCase()})`;
        
        // Clear existing options
        select.innerHTML = '';
        
        // Add model options
        Object.entries(data.models).forEach(([modelName, info]) => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = `${modelName} (${info.dim}d)`;
            select.appendChild(option);
        });
        
        // Add custom option
        const customOption = document.createElement('option');
        customOption.value = 'custom';
        customOption.textContent = 'Custom Model...';
        select.appendChild(customOption);
        
        // Update description when model changes
        select.addEventListener('change', () => {
            const descriptionEl = document.getElementById('model-description');
            if (select.value === 'custom') {
                descriptionEl.textContent = 'Enter any HuggingFace model name';
            } else if (data.models[select.value]) {
                descriptionEl.textContent = data.models[select.value].description;
            } else {
                descriptionEl.textContent = '';
            }
        });
        
        // Trigger initial description
        select.dispatchEvent(new Event('change'));
        
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Load models
    loadModels();
    
    // Set default tab
    switchTab('create');
});