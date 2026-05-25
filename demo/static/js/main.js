// Preset dataset Vietnamese variables configurations
const PRESET_VARIABLES = {
    playground: ['Caffeine', 'Nhịp Tim', 'Tập Trung'],
    boston: ['Tỷ Lệ Tội Phạm', 'Khí Thải Độc Hại', 'Số Phòng Trung Bình', 'Thâm Niên Nhà Ở', 'Thuế Bất Động Sản', 'Tỷ Lệ Học Sinh/Thầy Cô', 'Dân Cư Lao Động', 'Giá Nhà'],
    mpg: ['Hiệu Suất Xăng (MPG)', 'Số Xi Lanh', 'Dung Tích Động Cơ', 'Mã Lực', 'Trọng Lượng Xe', 'Gia Tốc', 'Năm Sản Xuất'],
    california: ['Tuổi Thọ Nhà', 'Số Phòng Trung Bình', 'Số Phòng Ngủ', 'Dân Số', 'Thu Nhập Trung Vị', 'Giá Nhà']
};

// State Management
let appState = {
    dataset: 'custom',
    nodes: [],
    adjMatrix: null,
    ateMatrix: null,
    metrics: null,
    running: false,
    
    // SVG and Graph layout
    svgWidth: 800,
    svgHeight: 550,
    nodeRadius: 28,
    nodePositions: {}, // Map nodeName -> {x, y}
    isDragging: false,
    draggedNode: null,
    
    // Intervention Lab state
    interveneNodeIdx: 0,
    interveneVal: 0.0,
    activeImpacts: {}
};

// DOM Elements
const elements = {
    uploadDropzone: document.getElementById('upload-dropzone'),
    csvFileInput: document.getElementById('csv-file-input'),
    uploadStatus: document.getElementById('upload-status'),
    uploadedFilename: document.getElementById('uploaded-filename'),
    uploadedVarCount: document.getElementById('uploaded-var-count'),
    uploadedSampleCount: document.getElementById('uploaded-sample-count'),
    
    btnTrain: document.getElementById('btn-train'),
    
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressText: document.getElementById('progress-text'),
    logBox: document.getElementById('log-box'),
    
    metricNll: document.getElementById('metric-nll'),
    metricHsic: document.getElementById('metric-hsic'),
    metricHval: document.getElementById('metric-hval'),
    
    graphSvg: document.getElementById('graph-svg'),
    
    labNodeSelect: document.getElementById('lab-node-select'),
    labNodeName: document.getElementById('lab-node-name'),
    labSlider: document.getElementById('lab-slider'),
    labSliderVal: document.getElementById('lab-slider-val'),
    impactBox: document.getElementById('impact-box'),
    
    // Tabbed Lab Elements
    tabLabIntervene: document.getElementById('tab-lab-intervene'),
    tabLabCluster: document.getElementById('tab-lab-cluster'),
    interveneInterface: document.getElementById('intervene-interface'),
    clusterInterface: document.getElementById('cluster-interface'),
    clusterCountSelect: document.getElementById('cluster-count-select'),
    btnRunCluster: document.getElementById('btn-run-cluster'),
    clusterResultsBox: document.getElementById('cluster-results-box'),
    
    explainerOverlay: document.getElementById('explainer-overlay'),
    explainerContent: document.getElementById('explainer-content'),
    explainerClose: document.getElementById('explainer-close')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    // Default load
    loadInitialState();
    if (window.lucide) {
        lucide.createIcons();
    }
});

function setupEventListeners() {
    // Custom CSV File Upload Trigger
    elements.uploadDropzone.addEventListener('click', () => {
        elements.csvFileInput.click();
    });

    elements.csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and Drop
    elements.uploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadDropzone.style.borderColor = 'var(--accent-blue)';
        elements.uploadDropzone.style.background = 'rgba(96, 165, 250, 0.04)';
    });

    elements.uploadDropzone.addEventListener('dragleave', () => {
        elements.uploadDropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        elements.uploadDropzone.style.background = 'rgba(255, 255, 255, 0.02)';
    });

    elements.uploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadDropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        elements.uploadDropzone.style.background = 'rgba(255, 255, 255, 0.02)';
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Train button
    elements.btnTrain.addEventListener('click', startCausalDiscovery);
    
    // Intervention controls
    elements.labNodeSelect.addEventListener('change', (e) => {
        appState.interveneNodeIdx = parseInt(e.target.value);
        elements.labNodeName.textContent = appState.nodes[appState.interveneNodeIdx] || 'None';
        elements.labSlider.value = 0.0;
        elements.labSliderVal.textContent = "+0.00";
        runIntervention();
    });
    
    elements.labSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        appState.interveneVal = val;
        elements.labSliderVal.textContent = val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
        
        clearTimeout(elements.sliderTimeout);
        elements.sliderTimeout = setTimeout(runIntervention, 80);
    });
    
    // Close explainer
    elements.explainerClose.addEventListener('click', () => {
        elements.explainerOverlay.style.display = 'none';
    });
    
    // Tabbed Lab Panel Switching
    elements.tabLabIntervene.addEventListener('click', () => {
        elements.tabLabIntervene.classList.add('active');
        elements.tabLabCluster.classList.remove('active');
        elements.interveneInterface.style.display = 'flex';
        elements.clusterInterface.style.display = 'none';
    });
    
    elements.tabLabCluster.addEventListener('click', () => {
        elements.tabLabCluster.classList.add('active');
        elements.tabLabIntervene.classList.remove('active');
        elements.clusterInterface.style.display = 'flex';
        elements.interveneInterface.style.display = 'none';
    });
    
    // Trigger Latent Subgroup Discovery
    elements.btnRunCluster.addEventListener('click', runLatentClustering);
}

// Load initial status (if already running or loaded)
function loadInitialState() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            updateUIFromStatus(data);
            if (data.running) {
                appState.running = true;
                setTimeout(pollStatus, 300);
            } else if (!data.adj_matrix) {
                // First load - empty state for custom CSV uploading
                appState.nodes = [];
                appState.adjMatrix = null;
                appState.ateMatrix = null;
                renderCausalGraph();
                populateLabDropdowns();
            }
        });
}

// Helper to restore button state with Lucide icon
function setBtnReady() {
    appState.running = false;
    elements.btnTrain.disabled = false;
    elements.btnTrain.innerHTML = '<i data-lucide="play" style="width: 15px; height: 15px;"></i> Chạy';
    if (window.lucide) {
        lucide.createIcons();
    }
}

// API: Start Causal Discovery training
function startCausalDiscovery() {
    if (appState.running) return;
    
    appState.running = true;
    elements.btnTrain.disabled = true;
    elements.btnTrain.innerHTML = '<span class="spinner"></span> Đang huấn luyện...';
    
    const params = {
        dataset: appState.dataset,
        sparsity: 0.002,
        strictness: 0.005,
        fast_mode: true
    };
    
    appState.activeImpacts = {}; // Reset impacts
    
    fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            elements.logBox.innerHTML = '';
            pollStatus();
        } else {
            alert('Lỗi: ' + data.message);
            setBtnReady();
        }
    })
    .catch(err => {
        console.error(err);
        setBtnReady();
    });
}

// API: Poll status periodically
function pollStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            updateUIFromStatus(data);
            if (data.running) {
                setTimeout(pollStatus, 300);
            } else {
                setBtnReady();
            }
        })
        .catch(err => {
            console.error(err);
            setBtnReady();
        });
}

// Update DOM elements using JSON data
function updateUIFromStatus(data) {
    // Status text
    if (data.running) {
        elements.statusIndicator.className = 'status-indicator active';
        elements.statusText.textContent = `Đang chạy (Vòng ${data.epoch}/${data.total_epochs})`;
    } else {
        elements.statusIndicator.className = 'status-indicator';
        elements.statusText.textContent = data.adj_matrix ? 'Đã hoàn thành' : 'Chưa khởi chạy';
    }
    
    // Progress
    elements.progressBarFill.style.width = `${data.progress}%`;
    elements.progressText.textContent = `${data.progress}%`;
    
    // Metrics
    elements.metricNll.textContent = data.nll !== 0 ? data.nll.toFixed(3) : '-';
    elements.metricHsic.textContent = data.hsic !== 0 ? data.hsic.toFixed(4) : '-';
    elements.metricHval.textContent = data.h_val !== 0 ? data.h_val.toFixed(5) : '-';
    
    // Update logs
    if (data.logs && data.logs.length > 0) {
        elements.logBox.innerHTML = data.logs.map(log => `<div>${log}</div>`).join('');
        elements.logBox.scrollTop = elements.logBox.scrollHeight;
    }
    
    // Update graph if weights exist and match the size of variables
    if (data.adj_matrix && data.node_names && data.adj_matrix.length === data.node_names.length) {
        appState.nodes = data.node_names;
        appState.adjMatrix = data.adj_matrix;
        appState.ateMatrix = data.ate_matrix;
        appState.metrics = data.metrics;
        
        populateLabDropdowns();
        renderCausalGraph();
    }
}

// API: Run causal intervention simulation
function runIntervention() {
    if (appState.nodes.length === 0) return;
    
    const params = {
        source_idx: appState.interveneNodeIdx,
        value: appState.interveneVal,
        dataset: appState.dataset
    };
    
    fetch('/api/intervene', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            appState.activeImpacts = data.impacts;
            renderImpactIndicators();
            updateNodeVisualsByImpact();
        }
    });
}

// API: Run latent space subgroup clustering
function runLatentClustering() {
    elements.btnRunCluster.disabled = true;
    elements.btnRunCluster.innerHTML = '<span class="spinner"></span> Đang phân nhóm...';
    
    const params = {
        n_clusters: parseInt(elements.clusterCountSelect.value)
    };
    
    fetch('/api/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        elements.btnRunCluster.disabled = false;
        elements.btnRunCluster.innerHTML = '<i data-lucide="zoom-in" style="width: 15px; height: 15px;"></i> Bắt Đầu Phân Nhóm';
        
        if (data.status === 'success') {
            const colors = ['var(--accent-blue)', 'var(--accent-green)', 'var(--accent-gold)', 'var(--accent-crimson)', '#818cf8'];
            
            elements.clusterResultsBox.innerHTML = `
                <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px; padding: 2px 4px;">
                    Dữ liệu đang xem: <b style="color: var(--accent-blue);">${data.dataset_name}</b> | Tổng số dòng: <b>${data.total_samples}</b>.
                </div>
                ${data.clusters.map((c, i) => {
                    const color = colors[i % colors.length];
                    return `
                        <div class="impact-row" style="flex-direction: column; align-items: stretch; gap: 6px; background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.03); margin-bottom: 4px; padding: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 11.5px; font-weight: 600;">
                                <span style="display: flex; align-items: center; gap: 6px;">
                                    <span style="width: 8px; height: 8px; border-radius: 50%; background: ${color};"></span>
                                    Nhóm #${c.id}
                                </span>
                                <span style="color: ${color}; font-weight: 700;">${c.percentage}% <span style="font-size: 10px; color: var(--text-muted); font-weight: 500;">(${c.count} dòng)</span></span>
                            </div>
                            <div class="progress-bar-container" style="height: 4px; background: rgba(255,255,255,0.04); border-radius: 4px; overflow: hidden; margin-top: 4px;">
                                <div style="height: 100%; width: ${c.percentage}%; background: ${color}; border-radius: 4px;"></div>
                            </div>
                        </div>
                    `;
                }).join('')}
                <div style="font-size: 9.5px; color: var(--text-muted); line-height: 1.4; margin-top: 6px; text-align: justify; padding: 6px 10px; background: rgba(255,255,255,0.01); border-radius: 6px; border: 1px solid rgba(255,255,255,0.03);">
                    💡 <b>Giải thích đơn giản:</b> Hệ thống sẽ tự động quét qua dữ liệu và chia các đối tượng (như bệnh nhân hoặc ngôi nhà) thành những nhóm nhỏ có đặc điểm tương đồng với nhau (ví dụ: nhóm bệnh nhân có nguy cơ cao, hoặc nhóm nhà giá rẻ). Điều này giúp bạn hiểu rõ từng phân khúc cụ thể mà không cần phải phân loại thủ công.
                </div>
            `;
            
            if (window.lucide) {
                lucide.createIcons();
            }
        } else {
            alert('Lỗi phân nhóm: ' + data.message);
        }
    })
    .catch(err => {
        console.error(err);
        elements.btnRunCluster.disabled = false;
        elements.btnRunCluster.innerHTML = '<i data-lucide="zoom-in" style="width: 15px; height: 15px;"></i> Bắt Đầu Phân Nhóm';
        if (window.lucide) {
            lucide.createIcons();
        }
    });
}

// ----------------- SIDEBAR LAB INTERFACES -----------------

function populateLabDropdowns() {
    // Populate node selectors
    const currentSelectVal = elements.labNodeSelect.value;
    elements.labNodeSelect.innerHTML = appState.nodes.map((node, i) => 
        `<option value="${i}" ${parseInt(currentSelectVal) === i ? 'selected' : ''}>${node}</option>`
    ).join('');
    
    appState.interveneNodeIdx = parseInt(elements.labNodeSelect.value) || 0;
    elements.labNodeName.textContent = appState.nodes[appState.interveneNodeIdx] || 'None';
}

function renderImpactIndicators() {
    if (!appState.activeImpacts || Object.keys(appState.activeImpacts).length === 0) {
        elements.impactBox.innerHTML = '<div style="color:var(--text-muted);font-size:12px;text-align:center;">Can thiệp biến nguồn để đo lường tác động hạ nguồn.</div>';
        return;
    }
    
    elements.impactBox.innerHTML = appState.nodes.map(node => {
        const delta = appState.activeImpacts[node] || 0.0;
        let cls = 'neutral';
        let prefix = '';
        if (delta > 0.005) { cls = 'positive'; prefix = '+'; }
        else if (delta < -0.005) { cls = 'negative'; }
        
        return `
            <div class="impact-row">
                <span class="impact-name">${node}</span>
                <span class="impact-val ${cls}">${prefix}${delta.toFixed(2)}</span>
            </div>
        `;
    }).join('');
}

// ----------------- SVG GRAPH RENDERER -----------------

function renderCausalGraph() {
    const svg = elements.graphSvg;
    svg.innerHTML = ''; // Clear SVG

    const n = appState.nodes.length;
    if (n === 0) {
        svg.innerHTML = `
            <g transform="translate(${appState.svgWidth / 2}, ${appState.svgHeight / 2 - 20})" text-anchor="middle">
                <rect x="-240" y="-55" width="480" height="110" rx="15" fill="rgba(255,255,255,0.01)" stroke="rgba(255,255,255,0.06)" stroke-width="1.2" />
                <text y="-10" fill="var(--accent-blue)" font-size="14.5px" font-weight="700" style="letter-spacing: 1px;">HỆ THỐNG SẴN SÀNG</text>
                <text y="18" fill="var(--text-main)" font-size="12px" font-weight="600">Vui lòng tải lên tệp dữ liệu CSV ở cột bên trái</text>
                <text y="38" fill="var(--text-muted)" font-size="11px">để bắt đầu phân tích tự động và khám phá mối quan hệ nhân quả.</text>
            </g>
        `;
        return;
    }
    
    // Layout Calculation: Circle layout (stable, doesn't overlap)
    // For Coffee Playground (3 nodes), make a simple horizontal sequence: Coffee -> HR -> Focus
    const centerX = appState.svgWidth / 2;
    const centerY = appState.svgHeight / 2;
    const radius = Math.min(centerX, centerY) - 75;
    
    appState.nodes.forEach((node, i) => {
        // If node already has a user-dragged position, preserve it
        if (appState.nodePositions[node]) return;
        
        if (n === 3) {
            // Horizontal line layout for simplicity
            const spacing = appState.svgWidth / 4;
            appState.nodePositions[node] = {
                x: spacing * (i + 1),
                y: centerY
            };
        } else {
            // Circle layout for larger biological networks
            const angle = (i * 2 * Math.PI) / n - Math.PI / 2; // Start from top
            appState.nodePositions[node] = {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            };
        }
    });
    
    // 1. Create Arrow marker markers for directed links
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <marker id="arrow-default" viewBox="0 0 10 10" refX="24" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="rgba(255,255,255,0.4)" />
        </marker>
        <marker id="arrow-blue" viewBox="0 0 10 10" refX="24" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent-blue)" />
        </marker>
        <marker id="arrow-crimson" viewBox="0 0 10 10" refX="24" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent-crimson)" />
        </marker>
    `;
    svg.appendChild(defs);
    
    // 2. Draw Causal Directed Edges (Arrows)
    const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    edgeGroup.setAttribute('id', 'edge-group');
    svg.appendChild(edgeGroup);
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const w = appState.adjMatrix ? appState.adjMatrix[i][j] : 0.0;
            if (Math.abs(w) > 0.05) { // Visual threshold
                const sourceNode = appState.nodes[i];
                const targetNode = appState.nodes[j];
                
                const p1 = appState.nodePositions[sourceNode];
                const p2 = appState.nodePositions[targetNode];
                
                const edgePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                edgePath.setAttribute('class', 'edge-path');
                edgePath.setAttribute('id', `edge-${i}-${j}`);
                
                // Add curves to prevent overlaps
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                
                // Slight curve: Quadratic bezier with control point slightly offset
                const offset = 22; // Pixels offset
                const mx = (p1.x + p2.x) / 2;
                const my = (p1.y + p2.y) / 2;
                const ux = -dy / dr; // Perpendicular unit vector
                const uy = dx / dr;
                const cx = mx + ux * offset;
                const cy = my + uy * offset;
                
                edgePath.setAttribute('d', `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`);
                
                // Determine color by weight strength
                if (w > 0) {
                    edgePath.style.stroke = 'rgba(96, 165, 250, 0.45)'; // Soft Slate Blue
                } else {
                    edgePath.style.stroke = 'rgba(251, 113, 133, 0.45)'; // Soft Rose/Crimson
                }
                
                // Adjust thickness based on absolute weight
                edgePath.style.strokeWidth = `${2.0 + Math.abs(w) * 3}px`;
                edgePath.setAttribute('marker-end', 'url(#arrow-default)');
                
                // Add click event for explanation drawer
                edgePath.addEventListener('click', (e) => {
                    e.stopPropagation();
                    showCausalExplanation(i, j, w);
                });
                
                edgeGroup.appendChild(edgePath);
                
                // Render text weights on edge center
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('class', 'edge-text');
                text.setAttribute('x', cx);
                text.setAttribute('y', cy - 5);
                text.setAttribute('text-anchor', 'middle');
                text.textContent = w >= 0 ? `+${w.toFixed(2)}` : w.toFixed(2);
                
                text.addEventListener('click', (e) => {
                    e.stopPropagation();
                    showCausalExplanation(i, j, w);
                });
                
                edgeGroup.appendChild(text);
            }
        }
    }
    
    // 3. Draw Nodes (Circles & Texts)
    const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    nodeGroup.setAttribute('id', 'node-group');
    svg.appendChild(nodeGroup);
    
    appState.nodes.forEach((node, i) => {
        const pos = appState.nodePositions[node];
        
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'node-group-item');
        g.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);
        g.setAttribute('id', `node-container-${i}`);
        
        // Background node circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('class', 'node-circle');
        circle.setAttribute('r', appState.nodeRadius);
        circle.setAttribute('id', `node-circle-${i}`);
        
        // Add click node to quickly select for intervention
        circle.addEventListener('click', (e) => {
            e.stopPropagation();
            elements.labNodeSelect.value = i;
            elements.labNodeSelect.dispatchEvent(new Event('change'));
            
            // Highlight node visually
            document.querySelectorAll('.node-circle').forEach(c => c.classList.remove('active-source'));
            circle.classList.add('active-source');
        });
        
        // Hover node to highlight parents and children paths
        circle.addEventListener('mouseenter', () => {
            highlightCausalPaths(i);
        });
        circle.addEventListener('mouseleave', () => {
            resetCausalHighlights();
        });
        
        // DRAG & DROP Handlers
        circle.addEventListener('mousedown', (e) => {
            appState.isDragging = true;
            appState.draggedNode = { name: node, index: i, group: g };
            svg.classList.add('dragging');
        });
        
        // Label text inside circle
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('class', 'node-label');
        text.setAttribute('y', 4);
        text.textContent = node;
        
        g.appendChild(circle);
        g.appendChild(text);
        nodeGroup.appendChild(g);
    });
    
    // Drag handlers on SVG canvas to track movement
    svg.addEventListener('mousemove', (e) => {
        if (!appState.isDragging || !appState.draggedNode) return;
        
        // Get mouse coordinates relative to SVG
        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Keep inside bounds
        const boundedX = Math.max(appState.nodeRadius + 10, Math.min(appState.svgWidth - appState.nodeRadius - 10, x));
        const boundedY = Math.max(appState.nodeRadius + 10, Math.min(appState.svgHeight - appState.nodeRadius - 10, y));
        
        // Update state positions
        const nodeName = appState.draggedNode.name;
        appState.nodePositions[nodeName] = { x: boundedX, y: boundedY };
        
        // Update group translate transformation
        appState.draggedNode.group.setAttribute('transform', `translate(${boundedX}, ${boundedY})`);
        
        // Rapidly redraw connecting edges
        redrawEdges();
    });
    
    svg.addEventListener('mouseup', () => {
        if (appState.isDragging) {
            appState.isDragging = false;
            appState.draggedNode = null;
            svg.classList.remove('dragging');
        }
    });
    
    // Apply visual effects if active intervention is active
    updateNodeVisualsByImpact();
}

// Redraw connecting lines during node dragging
function redrawEdges() {
    const edgeGroup = document.getElementById('edge-group');
    if (!edgeGroup) return;
    
    const n = appState.nodes.length;
    let paths = edgeGroup.getElementsByTagName('path');
    let texts = edgeGroup.getElementsByTagName('text');
    
    let pathIdx = 0;
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const w = appState.adjMatrix[i][j];
            if (Math.abs(w) > 0.05) {
                const p1 = appState.nodePositions[appState.nodes[i]];
                const p2 = appState.nodePositions[appState.nodes[j]];
                
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                
                const offset = 22;
                const mx = (p1.x + p2.x) / 2;
                const my = (p1.y + p2.y) / 2;
                const ux = -dy / dr;
                const uy = dx / dr;
                const cx = mx + ux * offset;
                const cy = my + uy * offset;
                
                if (paths[pathIdx]) {
                    paths[pathIdx].setAttribute('d', `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`);
                }
                
                if (texts[pathIdx]) {
                    texts[pathIdx].setAttribute('x', cx);
                    texts[pathIdx].setAttribute('y', cy - 5);
                }
                
                pathIdx++;
            }
        }
    }
}

// Highlight parent/child relations on hovering
function highlightCausalPaths(nodeIdx) {
    const n = appState.nodes.length;
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const w = appState.adjMatrix[i][j];
            if (Math.abs(w) > 0.05) {
                const edge = document.getElementById(`edge-${i}-${j}`);
                if (edge) {
                    if (i === nodeIdx) {
                        // Hovered node is cause (parent) -> highlight downstream children paths
                        edge.classList.add('highlight-child');
                        edge.setAttribute('marker-end', 'url(#arrow-crimson)');
                    } else if (j === nodeIdx) {
                        // Hovered node is effect (child) -> highlight upstream parent paths
                        edge.classList.add('highlight-parent');
                        edge.setAttribute('marker-end', 'url(#arrow-blue)');
                    }
                }
            }
        }
    }
}

function resetCausalHighlights() {
    document.querySelectorAll('.edge-path').forEach(edge => {
        edge.classList.remove('highlight-child');
        edge.classList.remove('highlight-parent');
        edge.setAttribute('marker-end', 'url(#arrow-default)');
    });
}

// Interventions visual effect: scaling nodes and colorizing them
function updateNodeVisualsByImpact() {
    if (!appState.activeImpacts || Object.keys(appState.activeImpacts).length === 0) {
        // Reset back to defaults
        appState.nodes.forEach((node, i) => {
            const circle = document.getElementById(`node-circle-${i}`);
            if (circle) {
                circle.style.transform = 'scale(1.0)';
                circle.style.stroke = 'var(--accent-blue)';
                circle.classList.remove('target-affected');
            }
        });
        return;
    }
    
    appState.nodes.forEach((node, i) => {
        const circle = document.getElementById(`node-circle-${i}`);
        if (!circle) return;
        
        const delta = appState.activeImpacts[node] || 0.0;
        
        if (i === appState.interveneNodeIdx) {
            // Source of intervention
            circle.classList.add('active-source');
            circle.style.stroke = 'var(--accent-crimson)';
            circle.style.transform = `scale(1.15)`;
        } else if (Math.abs(delta) > 0.02) {
            // Affected child nodes: swell up if positive, shrink if negative
            circle.classList.add('target-affected');
            
            if (delta > 0) {
                circle.style.stroke = 'var(--accent-green)';
                circle.style.transform = `scale(${1.0 + Math.min(delta * 0.15, 0.45)})`;
            } else {
                circle.style.stroke = 'var(--accent-crimson)';
                circle.style.transform = `scale(${1.0 - Math.min(Math.abs(delta) * 0.12, 0.35)})`;
            }
        } else {
            // Unaffected nodes
            circle.classList.remove('active-source');
            circle.classList.remove('target-affected');
            circle.style.stroke = 'rgba(255,255,255,0.2)';
            circle.style.transform = 'scale(0.9)';
        }
    });
}

// Translate mathematical weights into clear layperson explanation
function showCausalExplanation(sourceIdx, targetIdx, weight) {
    const sourceNode = appState.nodes[sourceIdx];
    const targetNode = appState.nodes[targetIdx];
    
    // Retrieve estimated ATE
    let ateVal = 0.0;
    if (appState.ateMatrix) {
        ateVal = appState.ateMatrix[sourceIdx][targetIdx];
    }
    
    const direction = weight > 0 ? "TĂNG" : "GIẢM";
    const strength = Math.abs(weight) > 0.4 ? "mạnh mẽ" : "vừa phải";
    
    // Construct rich text explainer
    let text = `
        <p style="margin-bottom:10px;">Đã phát hiện mối quan hệ <b>Nhân quả trực tiếp</b> từ nút <b>${sourceNode}</b> sang nút <b>${targetNode}</b>.</p>
        <p style="margin-bottom:10px;">Trọng số liên kết là <b style="color:${weight > 0 ? 'var(--accent-green)':'var(--accent-crimson)'}">${weight >= 0 ? `+${weight.toFixed(3)}` : weight.toFixed(3)}</b>, cho thấy <b>${sourceNode}</b> tác động <b>${direction}</b> và ${strength} lên <b>${targetNode}</b>.</p>
    `;
    
    if (ateVal !== 0.0) {
        text += `
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:10px;border-radius:10px;font-size:11.5px;line-height:1.5;margin-top:12px; display:flex; flex-direction:column; gap:6px;">
                <div style="display:flex; align-items:center; gap:6px; font-weight:700;"><i data-lucide="zoom-in" style="width:14px; height:14px; color:var(--accent-blue);"></i> <span>Mô phỏng Can thiệp (ATE):</span></div>
                <span>Khi dùng toán tử <i>do-calculus</i> để ép buộc tăng <b>${sourceNode}</b> lên 1 độ lệch chuẩn, <b>${targetNode}</b> dự kiến sẽ biến thiên <b>${ateVal >= 0 ? `+${ateVal.toFixed(2)}` : ateVal.toFixed(2)}</b> đơn vị.
            </div>
        `;
    }
    
    // Concrete examples for common-sense playground
    if (appState.dataset === 'playground') {
        if (sourceNode === 'Caffeine' && targetNode === 'Nhịp Tim') {
            text += `<p style="margin-top:10px;font-size:11.5px;color:var(--text-muted); display:flex; align-items:center; gap:6px;"><i data-lucide="coffee" style="width:13px; height:13px; color:var(--text-muted); flex-shrink:0;"></i> <span><b>Liên tưởng:</b> Caffeine hấp thụ vào máu kích hoạt hệ thần kinh giao cảm làm tăng tốc nhịp đập của tim.</span></p>`;
        } else if (sourceNode === 'Nhịp Tim' && targetNode === 'Tập Trung') {
            text += `<p style="margin-top:10px;font-size:11.5px;color:var(--text-muted); display:flex; align-items:center; gap:6px;"><i data-lucide="brain" style="width:13px; height:13px; color:var(--text-muted); flex-shrink:0;"></i> <span><b>Liên tưởng:</b> Nhịp tim ở mức vừa phải giúp tăng lưu lượng oxy lên não, tăng tập trung. Nhưng nhịp tim quá cao sẽ gây bồn chồn (jittery) và làm tụt giảm Tập Trung.</span></p>`;
        }
    }
    
    elements.explainerContent.innerHTML = text;
    elements.explainerOverlay.style.display = 'block';
    if (window.lucide) {
        lucide.createIcons();
    }
}

// Upload custom CSV to backend
function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Chỉ chấp nhận file định dạng .CSV!');
        return;
    }

    elements.logBox.innerHTML = `<div>[UPLOAD] Đang tải lên và xử lý tệp ${file.name}...</div>`;
    elements.btnTrain.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        elements.btnTrain.disabled = false;
        if (data.status === 'success') {
            // Show status panel
            elements.uploadStatus.style.display = 'block';
            elements.uploadedFilename.textContent = data.filename;
            elements.uploadedVarCount.textContent = data.variables.length;
            elements.uploadedSampleCount.textContent = data.samples;

            // Log status
            elements.logBox.innerHTML = `
                <div style="color: var(--accent-green); font-weight: 700;">[SUCCESS] Tải lên dữ liệu "${data.filename}" thành công!</div>
                <div style="margin-top: 4px;">[DATA] Phát hiện được <b>${data.variables.length} biến số</b>: ${data.variables.join(', ')}</div>
                <div>[DATA] Số lượng mẫu dữ liệu: <b>${data.samples} dòng</b>.</div>
                <div style="color: var(--accent-blue); font-weight: 600; margin-top: 5px;">[TIP] Nhấp nút "Chạy" để bắt đầu khám phá cấu trúc nhân quả bằng CausalFlowNet!</div>
            `;
            
            // Set initial state nodes
            appState.nodes = data.variables;
            appState.adjMatrix = null;
            appState.ateMatrix = null;
            renderCausalGraph();
            populateLabDropdowns();
        } else {
            elements.uploadStatus.style.display = 'none';
            elements.logBox.innerHTML = `<div style="color: var(--accent-crimson);">[ERROR] Tải lên thất bại: ${data.message}</div>`;
            alert('Lỗi tải file: ' + data.message);
        }
    })
    .catch(err => {
        elements.btnTrain.disabled = false;
        elements.uploadStatus.style.display = 'none';
        elements.logBox.innerHTML = `<div style="color: var(--accent-crimson);">[ERROR] Lỗi kết nối server khi tải file.</div>`;
        console.error(err);
    });
}
