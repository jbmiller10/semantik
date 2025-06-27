// Global state
let scannedFiles = [];
let activeWebSocket = null;
let scanWebSocket = null;
let currentScanId = null;

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
        instruction: instructionValue || null
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
                        <button onclick="connectToJobWebSocket('${job.id}')"
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
            method: 'POST'
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
function displaySearchResults(data) {
    const resultsDiv = document.getElementById('search-results');
    const container = document.getElementById('results-container');
    
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
                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                                    Score: ${chunk.score.toFixed(3)}
                                </span>
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