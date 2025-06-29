// Authentication helper
const auth = {
    getToken: () => localStorage.getItem('access_token'),
    
    getHeaders: () => ({
        'Authorization': `Bearer ${auth.getToken()}`,
        'Content-Type': 'application/json'
    }),
    
    checkAuth: async () => {
        const token = auth.getToken();
        if (!token) {
            window.location.href = '/login.html';
            return false;
        }
        
        try {
            const response = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            
            if (!response.ok) {
                throw new Error('Invalid token');
            }
            
            const user = await response.json();
            // Update UI with user info if needed
            const userElement = document.getElementById('current-user');
            if (userElement) {
                userElement.textContent = user.username;
            }
            
            return true;
        } catch (error) {
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            window.location.href = '/login.html';
            return false;
        }
    },
    
    logout: () => {
        const refreshToken = localStorage.getItem('refresh_token');
        fetch('/api/auth/logout', {
            method: 'POST',
            headers: auth.getHeaders(),
            body: JSON.stringify({ refresh_token: refreshToken })
        }).finally(() => {
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            window.location.href = '/login.html';
        });
    }
};

// Check authentication on page load
window.addEventListener('DOMContentLoaded', async () => {
    const isAuthenticated = await auth.checkAuth();
    if (isAuthenticated) {
        // Load models only after authentication is confirmed
        loadModels();
    }
});

// Global state
let scannedFiles = [];
let activeWebSocket = null;
let scanWebSocket = null;
let currentScanId = null;

// Rolling average calculator for metrics
class RollingAverage {
    constructor(windowSize = 5) {
        this.values = [];
        this.windowSize = windowSize;
    }
    
    add(value) {
        this.values.push(value);
        if (this.values.length > this.windowSize) {
            this.values.shift();
        }
    }
    
    getAverage() {
        if (this.values.length === 0) return 0;
        const sum = this.values.reduce((a, b) => a + b, 0);
        return sum / this.values.length;
    }
    
    clear() {
        this.values = [];
    }
}

// Store rolling averages for resource metrics
const resourceMetricsAverages = {
    cpu: new RollingAverage(5),
    memory: new RollingAverage(5),
    gpuMemory: new RollingAverage(5),
    gpuUtil: new RollingAverage(5)
};

// Helper function to show toast notifications
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg transform transition-all duration-300 z-50`;
    
    // Set color based on type
    const colors = {
        'success': 'bg-green-600 text-white',
        'error': 'bg-red-600 text-white',
        'warning': 'bg-yellow-600 text-white',
        'info': 'bg-blue-600 text-white'
    };
    
    toast.className += ` ${colors[type] || colors.info}`;
    toast.textContent = message;
    
    // Add to body
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => toast.classList.add('translate-x-0'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.add('translate-x-full');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Helper function to update job progress without full refresh
function updateJobProgress(jobId, data) {
    const jobCard = document.querySelector(`[data-job-id="${jobId}"]`);
    if (!jobCard) return;
    
    // Update progress bar
    const progressBar = jobCard.querySelector('.progress-bar');
    if (progressBar && data.processed_files !== undefined && data.total_files) {
        const percentage = Math.round((data.processed_files / data.total_files) * 100);
        progressBar.style.width = `${percentage}%`;
        
        const percentText = jobCard.querySelector('.progress-percentage');
        if (percentText) {
            percentText.textContent = `${percentage}%`;
        }
    }
    
    // Update current file
    const currentFileElement = jobCard.querySelector('.current-file');
    if (currentFileElement && data.current_file) {
        currentFileElement.innerHTML = `
            <i class="fas fa-spinner fa-spin mr-1"></i>
            ${data.status || 'Processing'}: ${data.current_file.split('/').pop()}
        `;
    }
    
    // Update processed count
    const processedElement = jobCard.querySelector('.processed-count');
    if (processedElement && data.processed_files !== undefined) {
        processedElement.textContent = data.processed_files;
    }
}

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

// Scan directory with WebSocket progress
async function scanDirectory() {
    const path = document.getElementById('directory-path').value;
    if (!path) {
        alert('Please enter a directory path');
        return;
    }
    
    // Generate unique scan ID
    currentScanId = Date.now().toString();
    
    // Show progress UI
    showScanProgress();
    
    // Close existing WebSocket if any
    if (scanWebSocket && scanWebSocket.readyState === WebSocket.OPEN) {
        scanWebSocket.close();
    }
    
    // Create WebSocket connection
    const wsUrl = `ws://${window.location.host}/ws/scan/${currentScanId}`;
    scanWebSocket = new WebSocket(wsUrl);
    
    scanWebSocket.onopen = () => {
        // Send scan request
        scanWebSocket.send(JSON.stringify({
            action: 'scan',
            path: path,
            recursive: true
        }));
    };
    
    scanWebSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleScanProgress(data);
    };
    
    scanWebSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        hideScanProgress();
        alert('Scan failed: Connection error');
    };
    
    scanWebSocket.onclose = () => {
        console.log('Scan WebSocket closed');
    };
}

// Cancel scan
function cancelScan() {
    if (scanWebSocket && scanWebSocket.readyState === WebSocket.OPEN) {
        scanWebSocket.send(JSON.stringify({ action: 'cancel' }));
        scanWebSocket.close();
    }
    hideScanProgress();
}

// Show scan progress UI
function showScanProgress() {
    const scanButton = document.getElementById('scan-button');
    const resultsDiv = document.getElementById('scan-results');
    
    // Hide results if shown
    resultsDiv.classList.add('hidden');
    
    // Replace scan button with progress UI
    scanButton.outerHTML = `
        <div id="scan-progress" class="space-y-3">
            <div class="flex items-center space-x-3">
                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span id="scan-status" class="text-sm">Starting scan...</span>
                <button onclick="cancelScan()" class="text-red-600 hover:text-red-800 text-sm">
                    <i class="fas fa-times-circle mr-1"></i>Cancel
                </button>
            </div>
            <div class="bg-gray-200 rounded-full h-2">
                <div id="scan-progress-bar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <div id="scan-details" class="text-xs text-gray-600"></div>
        </div>
    `;
}

// Hide scan progress UI
function hideScanProgress() {
    const progressDiv = document.getElementById('scan-progress');
    if (progressDiv) {
        progressDiv.outerHTML = `
            <button id="scan-button" onclick="scanDirectory()" 
                    class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                <i class="fas fa-folder-open mr-2"></i>Scan Directory
            </button>
        `;
    }
}

// Handle scan progress updates
function handleScanProgress(data) {
    const statusEl = document.getElementById('scan-status');
    const progressBar = document.getElementById('scan-progress-bar');
    const detailsEl = document.getElementById('scan-details');
    
    switch (data.type) {
        case 'started':
            statusEl.textContent = `Scanning ${data.path}...`;
            break;
            
        case 'counting':
            statusEl.textContent = `Counting files... (${data.count} found)`;
            break;
            
        case 'progress':
            const percent = Math.round((data.scanned / data.total) * 100);
            statusEl.textContent = `Scanning files... (${data.scanned}/${data.total})`;
            progressBar.style.width = `${percent}%`;
            detailsEl.textContent = `Current: ${data.current_path}`;
            break;
            
        case 'completed':
            scannedFiles = data.files;
            hideScanProgress();
            displayScanResults({ files: data.files, count: data.count });
            break;
            
        case 'error':
            hideScanProgress();
            alert(`Scan failed: ${data.error}`);
            break;
            
        case 'cancelled':
            hideScanProgress();
            break;
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
    
    const vectorDimValue = document.getElementById('vector-dim').value;
    const instructionValue = document.getElementById('instruction').value;
    
    try {
        // First, get a new job ID
        const idResponse = await fetch('/api/jobs/new-id', {
            headers: auth.getHeaders()
        });
        
        if (!idResponse.ok) {
            throw new Error('Failed to generate job ID');
        }
        
        const { job_id } = await idResponse.json();
        
        // Show scanning progress modal
        showScanningProgress();
        
        // Connect to scan WebSocket with the actual job ID
        const scanWs = connectToScanWebSocket(job_id);
        
        // Small delay to ensure WebSocket connection is established
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const jobData = {
            name: document.getElementById('job-name').value,
            description: document.getElementById('job-description').value,
            directory_path: document.getElementById('directory-path').value,
            model_name: modelName,
            chunk_size: parseInt(document.getElementById('chunk-size').value),
            chunk_overlap: parseInt(document.getElementById('chunk-overlap').value),
            batch_size: 96,
            vector_dim: vectorDimValue ? parseInt(vectorDimValue) : null,
            quantization: document.getElementById('quantization').value,
            instruction: instructionValue || null,
            job_id: job_id  // Include the pre-generated job ID
        };
        
        const response = await fetch('/api/jobs', {
            method: 'POST',
            headers: auth.getHeaders(),
            body: JSON.stringify(jobData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail);
        }
        
        const job = await response.json();
        
        // Close scan WebSocket and hide scanning progress
        if (scanWs && scanWs.readyState === WebSocket.OPEN) {
            scanWs.close();
        }
        hideScanningProgress();
        
        alert(`Job created successfully! ID: ${job.id}`);
        
        // Switch to jobs tab
        switchTab('jobs');
        
        // Connect to WebSocket for updates
        connectToJobWebSocket(job.id);
        
    } catch (error) {
        // Close scan WebSocket and hide scanning progress on error
        if (scanWs && scanWs.readyState === WebSocket.OPEN) {
            scanWs.close();
        }
        hideScanningProgress();
        alert(`Failed to create job: ${error.message}`);
    }
});

// Refresh jobs list
async function refreshJobs() {
    try {
        const response = await fetch('/api/jobs', {
            headers: { 'Authorization': `Bearer ${auth.getToken()}` }
        });
        const jobs = await response.json();
        
        const jobsList = document.getElementById('jobs-list');
        
        if (jobs.length === 0) {
            jobsList.innerHTML = '<p class="text-gray-500 text-center py-8">No jobs found</p>';
            return;
        }
        
        jobsList.innerHTML = jobs.map(job => {
            const progress = job.total_files > 0 ? (job.processed_files / job.total_files * 100).toFixed(1) : 0;
            const isRunning = job.status === 'processing' || job.status === 'scanning';
            
            return `
            <div class="border ${isRunning ? 'border-blue-300 shadow-lg' : 'border-gray-200'} rounded-lg p-4 ${isRunning ? 'job-card-running' : ''} transition-all duration-300" data-job-id="${job.id}">
                <div class="flex justify-between items-start mb-2">
                    <div>
                        <h3 class="font-semibold text-lg">${job.name}</h3>
                        <p class="text-sm text-gray-600">${job.directory_path}</p>
                    </div>
                    <div class="flex items-center space-x-2">
                        ${getStatusIndicator(job.status)}
                        <span class="px-3 py-1 rounded-full text-sm ${getStatusBadgeClass(job.status)} ${isRunning ? 'animate-pulse' : ''}">
                            ${job.status}
                        </span>
                    </div>
                </div>
                
                <div class="grid grid-cols-3 gap-4 mb-3">
                    <div>
                        <p class="text-sm text-gray-500">Total Files</p>
                        <p class="font-medium">${job.total_files}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Processed</p>
                        <p class="font-medium ${isRunning ? 'text-blue-600' : ''} processed-count">${job.processed_files}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Failed</p>
                        <p class="font-medium ${job.failed_files > 0 ? 'text-red-600' : ''}">${job.failed_files}</p>
                    </div>
                </div>
                
                ${job.status === 'processing' || job.status === 'scanning' ? `
                    <div class="mb-3">
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-gray-600">Progress</span>
                            <span class="font-medium text-blue-600 progress-percentage">${progress}%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div class="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full progress-bar-animated transition-all duration-500 ease-out progress-bar" 
                                 style="width: ${progress}%">
                                <div class="h-full bg-white/20 animate-shimmer"></div>
                            </div>
                        </div>
                        ${job.current_file ? `
                            <div class="mt-2 flex items-center space-x-2">
                                <div class="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                                <p class="text-sm text-gray-600 truncate current-file">Processing: ${job.current_file.split('/').pop()}</p>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
                
                ${job.status === 'completed' ? `
                    <div class="mb-3">
                        <div class="w-full bg-green-100 rounded-full h-3">
                            <div class="bg-green-500 h-3 rounded-full" style="width: 100%"></div>
                        </div>
                    </div>
                ` : ''}
                
                ${job.error ? `
                    <div class="bg-red-50 border border-red-200 rounded p-2 mb-3">
                        <p class="text-sm text-red-700">${job.error}</p>
                    </div>
                ` : ''}
                
                <div class="grid grid-cols-2 gap-2 text-sm text-gray-500 mb-2">
                    <div>Model: ${job.model_name}</div>
                    <div>Created: ${new Date(job.created_at).toLocaleString()}</div>
                    ${job.vector_size ? `<div>Vector size: ${job.vector_size}d</div>` : ''}
                    ${job.quantization ? `<div>Quantization: ${job.quantization}</div>` : ''}
                </div>
                
                <div class="mt-3 flex space-x-2">
                    ${job.status === 'processing' ? `
                        <button onclick="showJobMetrics('${job.id}')"
                                class="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors">
                            <i class="fas fa-chart-line mr-1"></i>Monitor
                        </button>
                    ` : ''}
                    ${job.status === 'processing' || job.status === 'scanning' ? `
                        <button onclick="cancelJob('${job.id}')"
                                class="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors">
                            <i class="fas fa-times-circle mr-1"></i>Cancel
                        </button>
                    ` : ''}
                    ${job.status === 'completed' ? `
                        <button onclick="useJobForSearch('${job.id}', '${job.name}')"
                                class="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700 transition-colors">
                            <i class="fas fa-search mr-1"></i>Search
                        </button>
                    ` : ''}
                </div>
            </div>
        `}).join('');
        
    } catch (error) {
        console.error('Failed to load jobs:', error);
    }
}

// Cancel a job
async function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/jobs/${jobId}/cancel`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${auth.getToken()}` }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to cancel job');
        }
        
        const result = await response.json();
        alert(result.message || 'Job cancellation requested');
        
        // Refresh the jobs list to show updated status
        refreshJobs();
        
    } catch (error) {
        alert(`Failed to cancel job: ${error.message}`);
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
        
        if (data.type === 'job_started') {
            console.log(`Job ${jobId} started processing ${data.total_files} files`);
            refreshJobs();
        } else if (data.type === 'file_processing') {
            // Update specific job card with detailed status
            updateJobProgress(jobId, data);
        } else if (data.type === 'file_completed') {
            // Update progress without full refresh
            updateJobProgress(jobId, data);
        } else if (data.type === 'job_completed') {
            showToast('Job completed successfully!', 'success');
            refreshJobs();
            ws.close();
        } else if (data.type === 'job_cancelled') {
            showToast('Job cancelled', 'warning');
            refreshJobs();
            ws.close();
        } else if (data.type === 'error') {
            showToast(`Job error: ${data.message}`, 'error');
            refreshJobs();
            ws.close();
        } else if (data.type === 'progress') {
            // Legacy progress update
            refreshJobs();
        } else if (data.type === 'completed') {
            // Legacy completed
            alert('Job completed successfully!');
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
        const response = await fetch('/api/jobs', {
            headers: { 'Authorization': `Bearer ${auth.getToken()}` }
        });
        const jobs = await response.json();
        
        const select = document.getElementById('search-collection');
        select.innerHTML = '';
        
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

// Toggle search mode
function toggleSearchMode() {
    const useHybridSearch = document.getElementById('use-hybrid-search').checked;
    const hybridOptions = document.getElementById('hybrid-search-options');
    
    if (useHybridSearch) {
        hybridOptions.classList.remove('hidden');
    } else {
        hybridOptions.classList.add('hidden');
    }
}

// Search form submission
document.getElementById('search-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const useHybridSearch = document.getElementById('use-hybrid-search').checked;
    
    if (!useHybridSearch) {
        // Vector search
        const searchData = {
            query: document.getElementById('search-query').value,
            k: parseInt(document.getElementById('search-k').value),
            job_id: document.getElementById('search-collection').value || null
        };
        
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: auth.getHeaders(),
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
    } else {
        // Hybrid search
        const searchData = {
            query: document.getElementById('search-query').value,
            k: parseInt(document.getElementById('search-k').value),
            job_id: document.getElementById('search-collection').value || null,
            mode: document.getElementById('hybrid-mode').value,
            keyword_mode: document.getElementById('keyword-mode').value
        };
        
        try {
            const response = await fetch('/api/hybrid_search', {
                method: 'POST',
                headers: auth.getHeaders(),
                body: JSON.stringify(searchData)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail);
            }
            
            const results = await response.json();
            displayHybridSearchResults(results);
            
        } catch (error) {
            alert(`Hybrid search failed: ${error.message}`);
        }
    }
});

// Group search results by document path
function groupResultsByDocument(results) {
    const grouped = {};
    
    results.forEach(result => {
        const path = result.payload.path;
        if (!grouped[path]) {
            grouped[path] = {
                path: path,
                filename: path.split('/').pop(),
                chunks: [],
                maxScore: 0
            };
        }
        grouped[path].chunks.push(result);
        grouped[path].maxScore = Math.max(grouped[path].maxScore, result.score);
    });
    
    // Sort documents by their highest chunk score
    return Object.values(grouped).sort((a, b) => b.maxScore - a.maxScore);
}

// Toggle document expansion
function toggleDocument(docId) {
    const content = document.getElementById(`doc-content-${docId}`);
    const icon = document.getElementById(`doc-icon-${docId}`);
    
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.classList.remove('fa-chevron-right');
        icon.classList.add('fa-chevron-down');
    } else {
        content.classList.add('hidden');
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-right');
    }
}

// Display search results
function displayHybridSearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const container = document.getElementById('results-container');
    
    // Store search context for view button
    const searchQuery = document.getElementById('search-query').value;
    const currentJobId = document.getElementById('search-collection').value;
    
    resultsDiv.classList.remove('hidden');
    
    if (data.results.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-8">No results found</p>';
        return;
    }
    
    // Convert hybrid results format to match regular results for grouping
    const convertedResults = data.results.map(result => ({
        payload: {
            path: result.path || result.payload?.path,
            chunk_id: result.chunk_id || result.payload?.chunk_id,
            doc_id: result.doc_id || result.payload?.doc_id,
            text: result.payload?.content?.substring(0, 200) || result.payload?.text,
            matched_keywords: result.matched_keywords
        },
        score: result.combined_score || result.score,
        keyword_score: result.keyword_score
    }));
    
    // Group results by document
    const groupedDocs = groupResultsByDocument(convertedResults);
    
    container.innerHTML = groupedDocs.map((doc, docIdx) => {
        // Sort chunks within each document by score
        const sortedChunks = doc.chunks.sort((a, b) => b.score - a.score);
        
        return `
        <div class="border border-gray-200 rounded-lg overflow-hidden mb-4 transition-shadow hover:shadow-lg">
            <!-- Document Header (Always Visible) -->
            <div class="bg-gray-50 p-4 cursor-pointer hover:bg-gray-100 transition-colors"
                 onclick="toggleDocument(${docIdx})">
                <div class="flex justify-between items-start">
                    <div class="flex items-center flex-1">
                        <i id="doc-icon-${docIdx}" class="fas fa-chevron-right mr-2 text-gray-500 transition-transform"></i>
                        <div class="flex-1">
                            <h4 class="font-semibold text-lg">
                                <i class="fas fa-file-alt mr-2 text-gray-500"></i>
                                ${doc.filename}
                            </h4>
                            <p class="text-sm text-gray-600 mt-1">${doc.path}</p>
                        </div>
                    </div>
                    <div class="flex flex-col items-end ml-4">
                        <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                            ${doc.chunks.length} ${doc.chunks.length === 1 ? 'chunk' : 'chunks'}
                        </span>
                        <span class="text-xs text-gray-500 mt-1">
                            Max score: ${doc.maxScore.toFixed(3)}
                        </span>
                    </div>
                </div>
            </div>
            
            <!-- Document Chunks (Collapsible) -->
            <div id="doc-content-${docIdx}" class="hidden border-t border-gray-200">
                <div class="p-4 space-y-3">
                    ${sortedChunks.map((chunk, chunkIdx) => `
                        <div class="border border-gray-200 rounded-lg p-4 bg-white hover:bg-gray-50 transition-colors">
                            <div class="flex justify-between items-start mb-2">
                                <h5 class="font-medium text-gray-700">
                                    Chunk ${chunkIdx + 1}
                                </h5>
                                <div class="flex gap-2">
                                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                                        Score: ${chunk.score.toFixed(3)}
                                    </span>
                                    ${chunk.keyword_score !== undefined ? `
                                        <span class="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm">
                                            Keywords: ${chunk.keyword_score.toFixed(3)}
                                        </span>
                                    ` : ''}
                                    ${chunk.payload.doc_id && chunk.payload.metadata ? `
                                        <button onclick="viewDocument('${currentJobId || ''}', '${chunk.payload.doc_id}', '${doc.filename.replace(/'/g, "\\'")}', '${searchQuery.replace(/'/g, "\\'")}', ${chunk.payload.metadata.page_number || 1})"
                                                class="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm transition-colors">
                                            <i class="fas fa-eye mr-1"></i>View
                                        </button>
                                    ` : ''}
                                </div>
                            </div>
                            
                            ${chunk.payload.text ? `
                                <div class="bg-gray-50 rounded p-3 mt-2">
                                    <p class="text-sm text-gray-700">${chunk.payload.text}...</p>
                                </div>
                            ` : ''}
                            
                            <div class="mt-2 text-xs text-gray-500">
                                Chunk ID: ${chunk.payload.chunk_id}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        `;
    }).join('');
}

function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const container = document.getElementById('results-container');
    
    // Store search context for view button
    const searchQuery = document.getElementById('search-query').value;
    const currentJobId = document.getElementById('search-collection').value;
    
    resultsDiv.classList.remove('hidden');
    
    if (data.results.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-8">No results found</p>';
        return;
    }
    
    // Group results by document
    const groupedDocs = groupResultsByDocument(data.results);
    
    container.innerHTML = groupedDocs.map((doc, docIdx) => {
        // Sort chunks within each document by score
        const sortedChunks = doc.chunks.sort((a, b) => b.score - a.score);
        
        return `
        <div class="border border-gray-200 rounded-lg overflow-hidden mb-4 transition-shadow hover:shadow-lg">
            <!-- Document Header (Always Visible) -->
            <div class="bg-gray-50 p-4 cursor-pointer hover:bg-gray-100 transition-colors"
                 onclick="toggleDocument(${docIdx})">
                <div class="flex justify-between items-start">
                    <div class="flex items-center flex-1">
                        <i id="doc-icon-${docIdx}" class="fas fa-chevron-right mr-2 text-gray-500 transition-transform"></i>
                        <div class="flex-1">
                            <h4 class="font-semibold text-lg">
                                <i class="fas fa-file-alt mr-2 text-gray-500"></i>
                                ${doc.filename}
                            </h4>
                            <p class="text-sm text-gray-600 mt-1">${doc.path}</p>
                        </div>
                    </div>
                    <div class="flex flex-col items-end ml-4">
                        <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                            ${doc.chunks.length} ${doc.chunks.length === 1 ? 'chunk' : 'chunks'}
                        </span>
                        <span class="text-xs text-gray-500 mt-1">
                            Max score: ${doc.maxScore.toFixed(3)}
                        </span>
                    </div>
                </div>
            </div>
            
            <!-- Document Chunks (Collapsible) -->
            <div id="doc-content-${docIdx}" class="hidden border-t border-gray-200">
                <div class="p-4 space-y-3">
                    ${sortedChunks.map((chunk, chunkIdx) => `
                        <div class="border border-gray-200 rounded-lg p-4 bg-white hover:bg-gray-50 transition-colors">
                            <div class="flex justify-between items-start mb-2">
                                <h5 class="font-medium text-gray-700">
                                    Chunk ${chunkIdx + 1}
                                </h5>
                                <div class="flex items-center space-x-2">
                                    <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                                        Score: ${chunk.score.toFixed(3)}
                                    </span>
                                    ${chunk.payload.doc_id && chunk.payload.metadata ? `
                                        <button onclick="viewDocument('${currentJobId || ''}', '${chunk.payload.doc_id}', '${doc.filename.replace(/'/g, "\\'")}', '${searchQuery.replace(/'/g, "\\'")}', ${chunk.payload.metadata.page_number || 1})"
                                                class="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm transition-colors">
                                            <i class="fas fa-eye mr-1"></i>View
                                        </button>
                                    ` : ''}
                                </div>
                            </div>
                            
                            ${chunk.payload.text ? `
                                <div class="bg-gray-50 rounded p-3 mt-2">
                                    <p class="text-sm text-gray-700">${chunk.payload.text}...</p>
                                </div>
                            ` : ''}
                            
                            <div class="mt-2 text-xs text-gray-500">
                                Chunk ID: ${chunk.payload.chunk_id}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
        `;
    }).join('');
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

function getStatusIndicator(status) {
    switch(status) {
        case 'processing':
            return `<div class="relative">
                        <div class="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-blue-400 opacity-75"></div>
                        <div class="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></div>
                    </div>`;
        case 'scanning':
            return `<div class="relative">
                        <div class="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-yellow-400 opacity-75"></div>
                        <div class="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></div>
                    </div>`;
        case 'completed':
            return `<div class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></div>`;
        case 'failed':
            return `<div class="relative inline-flex rounded-full h-3 w-3 bg-red-500"></div>`;
        case 'created':
            return `<div class="relative inline-flex rounded-full h-3 w-3 bg-gray-400"></div>`;
        default:
            return '';
    }
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
        const response = await fetch('/api/models', {
            headers: { 'Authorization': `Bearer ${auth.getToken()}` }
        });
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

// Scanning progress functions
function showScanningProgress() {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.id = 'scanning-modal';
    modal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 class="text-lg font-semibold mb-4">Scanning Directory...</h3>
            <div class="space-y-4">
                <div id="scan-status" class="text-sm text-gray-600">
                    Initializing scan...
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div id="scan-progress-bar" class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
                </div>
                <div id="scan-details" class="text-xs text-gray-500">
                    <span id="scan-current-path"></span>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function hideScanningProgress() {
    const modal = document.getElementById('scanning-modal');
    if (modal) {
        modal.remove();
    }
}

function connectToScanWebSocket(scanId) {
    const ws = new WebSocket(`ws://${window.location.host}/ws/scan/${scanId}`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        const statusEl = document.getElementById('scan-status');
        const progressBar = document.getElementById('scan-progress-bar');
        const currentPathEl = document.getElementById('scan-current-path');
        
        if (!statusEl || !progressBar) return;
        
        if (data.type === 'counting') {
            statusEl.textContent = `Counting files... (${data.count} found)`;
        } else if (data.type === 'progress') {
            const percentage = (data.scanned / data.total * 100).toFixed(1);
            statusEl.textContent = `Scanning files: ${data.scanned} / ${data.total}`;
            progressBar.style.width = `${percentage}%`;
            if (currentPathEl && data.current_path) {
                // Truncate long paths
                const path = data.current_path;
                const maxLength = 60;
                const displayPath = path.length > maxLength 
                    ? '...' + path.slice(-(maxLength - 3))
                    : path;
                currentPathEl.textContent = displayPath;
            }
        }
    };
    
    ws.onerror = (error) => {
        console.error('Scan WebSocket error:', error);
    };
    
    return ws;
}

// Show job metrics modal
async function showJobMetrics(jobId) {
    // First connect to WebSocket for real-time updates
    connectToJobWebSocket(jobId);
    
    // Create metrics modal
    const modal = document.createElement('div');
    modal.id = 'metrics-modal';
    modal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 shadow-xl max-h-[90vh] overflow-y-auto">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-semibold">Job Metrics - ${jobId}</h3>
                <button onclick="closeMetricsModal()" class="text-gray-500 hover:text-gray-700">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            
            <div class="space-y-6">
                <!-- Real-time Progress -->
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 class="font-semibold text-blue-900 mb-2">Real-time Progress</h4>
                    <div id="metrics-progress" class="space-y-2">
                        <div class="text-sm text-gray-600">Connecting to job...</div>
                    </div>
                </div>
                
                <!-- Performance Metrics -->
                <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h4 class="font-semibold text-green-900 mb-2">Performance Metrics</h4>
                    <div id="metrics-performance" class="grid grid-cols-2 gap-4">
                        <div class="text-sm text-gray-600">Loading metrics...</div>
                    </div>
                </div>
                
                <!-- Resource Usage -->
                <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <h4 class="font-semibold text-purple-900 mb-2">Resource Usage</h4>
                    <div id="metrics-resources" class="grid grid-cols-2 gap-4">
                        <div class="text-sm text-gray-600">Loading resource data...</div>
                    </div>
                </div>
                
                <!-- Error Tracking -->
                <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h4 class="font-semibold text-red-900 mb-2">Error Tracking</h4>
                    <div id="metrics-errors" class="space-y-2">
                        <div class="text-sm text-gray-600">Loading error data...</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // Fetch and display metrics
    await updateMetricsDisplay(jobId);
    
    // Set up periodic refresh
    const metricsInterval = setInterval(() => {
        updateMetricsDisplay(jobId);
    }, 1000); // Update every 1 second for smoother metrics
    
    // Store interval ID for cleanup
    modal.dataset.intervalId = metricsInterval;
}

// Close metrics modal
function closeMetricsModal() {
    const modal = document.getElementById('metrics-modal');
    if (modal) {
        // Clear refresh interval
        const intervalId = modal.dataset.intervalId;
        if (intervalId) {
            clearInterval(intervalId);
        }
        modal.remove();
        
        // Clear rolling averages
        Object.values(resourceMetricsAverages).forEach(avg => avg.clear());
    }
}

// Update metrics display
async function updateMetricsDisplay(jobId) {
    try {
        // Fetch job details
        const jobResponse = await fetch(`/api/jobs/${jobId}`, {
            headers: auth.getHeaders()
        });
        
        if (!jobResponse.ok) {
            throw new Error('Failed to fetch job details');
        }
        
        const job = await jobResponse.json();
        
        // Update progress section
        const progressDiv = document.getElementById('metrics-progress');
        if (progressDiv) {
            const progress = job.total_files > 0 ? (job.processed_files / job.total_files * 100).toFixed(1) : 0;
            progressDiv.innerHTML = `
                <div class="space-y-2">
                    <div class="flex justify-between text-sm">
                        <span>Files Processed:</span>
                        <span class="font-medium">${job.processed_files} / ${job.total_files}</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: ${progress}%"></div>
                    </div>
                    ${job.current_file ? `
                        <div class="text-sm text-gray-600">
                            <i class="fas fa-spinner fa-spin mr-1"></i>
                            Current: ${job.current_file.split('/').pop()}
                        </div>
                    ` : ''}
                    ${job.failed_files > 0 ? `
                        <div class="text-sm text-red-600">
                            <i class="fas fa-exclamation-triangle mr-1"></i>
                            Failed files: ${job.failed_files}
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        // Fetch metrics from the API endpoint
        try {
            const metricsResponse = await fetch('/api/metrics', {
                headers: auth.getHeaders()
            });
            
            if (metricsResponse.ok) {
                const metricsData = await metricsResponse.json();
                if (metricsData.available && metricsData.data) {
                    parseAndDisplayMetrics(metricsData.data, job);
                } else {
                    displayJobBasedMetrics(job);
                }
            } else {
                displayJobBasedMetrics(job);
            }
        } catch (error) {
            console.log('Failed to fetch metrics:', error);
            displayJobBasedMetrics(job);
        }
        
    } catch (error) {
        console.error('Failed to update metrics:', error);
    }
}

// Parse Prometheus metrics and display relevant ones
function parseAndDisplayMetrics(metricsText, job) {
    // Parse relevant metrics from Prometheus format
    const lines = metricsText.split('\n');
    const metrics = {};
    
    lines.forEach(line => {
        if (line.startsWith('#') || !line.trim()) return;
        
        // More precise regex for Prometheus metrics
        const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?(.*?)\}?\s+(.+)$/);
        if (match) {
            const [, name, labels, value] = match;
            
            if (!metrics[name]) metrics[name] = [];
            metrics[name].push({ labels, value: parseFloat(value) });
        }
    });
    
    // Debug: Log available metrics
    console.log('Available metrics:', Object.keys(metrics).filter(k => k.includes('embedding')));
    
    // Update performance metrics
    const performanceDiv = document.getElementById('metrics-performance');
    if (performanceDiv) {
        const embedRate = calculateEmbeddingRate(job);
        
        // Calculate average from sum and count for histograms
        const embedSum = getMetricValue(metrics, 'embedding_batch_duration_seconds_sum', 'last', 0);
        const embedCount = getMetricValue(metrics, 'embedding_batch_duration_seconds_count', 'last', 0);
        const avgEmbedTime = embedCount > 0 ? (embedSum / embedCount * 1000) : 0;
        
        // Debug log
        console.log('Embedding metrics:', { embedSum, embedCount, avgEmbedTime });
        
        performanceDiv.innerHTML = `
            <div>
                <div class="text-sm text-gray-600">Processing Rate</div>
                <div class="text-lg font-semibold">${embedRate} files/min</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Avg Embedding Time</div>
                <div class="text-lg font-semibold">${avgEmbedTime.toFixed(1)} ms</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Total Chunks</div>
                <div class="text-lg font-semibold">${getMetricValue(metrics, 'embedding_chunks_created_total', 'sum')}</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Vectors Generated</div>
                <div class="text-lg font-semibold">${getMetricValue(metrics, 'embedding_vectors_generated_total', 'sum')}</div>
            </div>
        `;
    }
    
    // Update resource metrics
    const resourcesDiv = document.getElementById('metrics-resources');
    if (resourcesDiv) {
        const cpuUsage = getMetricValue(metrics, 'embedding_cpu_utilization_percent', 'last');
        const memUsage = getMetricValue(metrics, 'embedding_memory_utilization_percent', 'last');
        const gpuMemoryBytes = getMetricValue(metrics, 'embedding_gpu_memory_used_bytes', 'last', 0);
        const gpuMemoryTotalBytes = getMetricValue(metrics, 'embedding_gpu_memory_total_bytes', 'last', 0);
        const gpuMemoryGB = gpuMemoryBytes / (1024 * 1024 * 1024);
        const gpuMemoryTotalGB = gpuMemoryTotalBytes / (1024 * 1024 * 1024);
        const gpuUtil = getMetricValue(metrics, 'embedding_gpu_utilization_percent', 'last', 0);
        
        // Calculate GPU memory percentage
        const gpuMemoryPercent = gpuMemoryTotalBytes > 0 ? (gpuMemoryBytes / gpuMemoryTotalBytes) * 100 : 0;
        
        // Add to rolling averages
        resourceMetricsAverages.cpu.add(cpuUsage);
        resourceMetricsAverages.memory.add(memUsage);
        resourceMetricsAverages.gpuMemory.add(gpuMemoryPercent);
        resourceMetricsAverages.gpuUtil.add(gpuUtil);
        
        // Get averaged values
        const avgCpuUsage = resourceMetricsAverages.cpu.getAverage();
        const avgMemUsage = resourceMetricsAverages.memory.getAverage();
        const avgGpuUtil = resourceMetricsAverages.gpuUtil.getAverage();
        
        resourcesDiv.innerHTML = `
            <div>
                <div class="text-sm text-gray-600">CPU Usage</div>
                <div class="text-lg font-semibold">${avgCpuUsage.toFixed(1)}%</div>
                <div class="text-xs text-gray-400">5s avg</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">System Memory</div>
                <div class="text-lg font-semibold">${avgMemUsage.toFixed(1)}%</div>
                <div class="text-xs text-gray-400">5s avg</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">GPU Memory</div>
                <div class="text-lg font-semibold">${gpuMemoryGB.toFixed(2)} / ${gpuMemoryTotalGB.toFixed(1)} GB</div>
                <div class="text-xs text-gray-500">${gpuMemoryPercent.toFixed(1)}% used</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">GPU Utilization</div>
                <div class="text-lg font-semibold">${avgGpuUtil.toFixed(1)}%</div>
                <div class="text-xs text-gray-400">5s avg</div>
            </div>
        `;
    }
    
    // Update error metrics
    const errorsDiv = document.getElementById('metrics-errors');
    if (errorsDiv) {
        const oomErrors = getMetricValue(metrics, 'embedding_oom_errors_total', 'sum');
        const batchReductions = getMetricValue(metrics, 'embedding_batch_size_reductions_total', 'sum');
        
        errorsDiv.innerHTML = `
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <div class="text-sm text-gray-600">OOM Errors</div>
                    <div class="text-lg font-semibold ${oomErrors > 0 ? 'text-red-600' : ''}">${oomErrors}</div>
                </div>
                <div>
                    <div class="text-sm text-gray-600">Batch Size Reductions</div>
                    <div class="text-lg font-semibold ${batchReductions > 0 ? 'text-yellow-600' : ''}">${batchReductions}</div>
                </div>
            </div>
            ${oomErrors > 0 || batchReductions > 0 ? `
                <div class="mt-2 text-sm text-yellow-700 bg-yellow-50 p-2 rounded">
                    <i class="fas fa-info-circle mr-1"></i>
                    The system has automatically adjusted batch sizes to handle memory constraints.
                </div>
            ` : ''}
        `;
    }
}

// Display metrics based on job data when Prometheus is not available
function displayJobBasedMetrics(job) {
    const performanceDiv = document.getElementById('metrics-performance');
    if (performanceDiv) {
        const embedRate = calculateEmbeddingRate(job);
        performanceDiv.innerHTML = `
            <div>
                <div class="text-sm text-gray-600">Processing Rate</div>
                <div class="text-lg font-semibold">${embedRate} files/min</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Batch Size</div>
                <div class="text-lg font-semibold">${job.batch_size || 96}</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Model</div>
                <div class="text-sm font-medium">${job.model_name}</div>
            </div>
            <div>
                <div class="text-sm text-gray-600">Quantization</div>
                <div class="text-lg font-semibold">${job.quantization || 'float32'}</div>
            </div>
        `;
    }
    
    const resourcesDiv = document.getElementById('metrics-resources');
    if (resourcesDiv) {
        resourcesDiv.innerHTML = `
            <div class="col-span-2 text-sm text-gray-500">
                <i class="fas fa-info-circle mr-1"></i>
                Resource metrics require Prometheus integration.
                <br>Metrics will be available once the job starts processing.
            </div>
        `;
    }
    
    const errorsDiv = document.getElementById('metrics-errors');
    if (errorsDiv) {
        errorsDiv.innerHTML = `
            <div class="text-sm">
                <div class="text-gray-600">Failed Files</div>
                <div class="text-lg font-semibold ${job.failed_files > 0 ? 'text-red-600' : ''}">${job.failed_files}</div>
            </div>
        `;
    }
}

// Helper function to calculate embedding rate
function calculateEmbeddingRate(job) {
    console.log('Job data:', { start_time: job.start_time, processed_files: job.processed_files });
    
    if (!job.start_time || job.processed_files === 0) return '0';
    
    const startTime = new Date(job.start_time);
    const currentTime = new Date();
    const elapsedMinutes = (currentTime - startTime) / (1000 * 60);
    
    console.log('Time calculation:', { startTime, currentTime, elapsedMinutes });
    
    if (elapsedMinutes < 0.1) return '0';
    
    const rate = job.processed_files / elapsedMinutes;
    return rate.toFixed(1);
}

// Helper function to extract metric value
function getMetricValue(metrics, metricName, aggregation = 'last', defaultValue = 0) {
    if (!metrics[metricName] || metrics[metricName].length === 0) {
        return defaultValue;
    }
    
    const values = metrics[metricName].map(m => m.value);
    
    switch (aggregation) {
        case 'sum':
            return values.reduce((a, b) => a + b, 0);
        case 'avg':
            return values.reduce((a, b) => a + b, 0) / values.length;
        case 'last':
            return values[values.length - 1];
        default:
            return defaultValue;
    }
}

// Document viewer function
function viewDocument(jobId, docId, filename, searchQuery, pageNumber) {
    // Ensure the document viewer is loaded
    if (!window.documentViewer) {
        console.error('Document viewer not loaded');
        return;
    }
    
    // Open the document viewer with the specified parameters
    window.documentViewer.open(jobId, docId, filename, {
        searchQuery: searchQuery,
        pageNumber: pageNumber || 1
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Models are loaded after authentication check
    // Set default tab
    switchTab('create');
});