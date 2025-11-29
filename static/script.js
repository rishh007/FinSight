let ws = null;
        let sessionId = null;
        const messagesContainer = document.getElementById('messagesContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const connectionStatus = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        const sessionInfo = document.getElementById('sessionInfo');
        const clearChatBtn = document.getElementById('clearChatBtn');
        const filesBtn = document.getElementById('filesBtn');
        const filesSidebar = document.getElementById('filesSidebar');
        const closeSidebarBtn = document.getElementById('closeSidebar');

// Toggle files sidebar
filesBtn.addEventListener('click', async () => {
    filesSidebar.classList.toggle('open');
    if (filesSidebar.classList.contains('open')) {
        await loadSessionFiles();
    }
});

closeSidebarBtn.addEventListener('click', () => {
    filesSidebar.classList.remove('open');
});

// Load session files
async function loadSessionFiles() {
    try {
        const response = await fetch(`http://localhost:5500/api/sessions/${sessionId}/files`);
        const data = await response.json();
        
        const filesList = document.getElementById('filesList');
        filesList.innerHTML = '';
        
        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                const icon = file.type === 'chart' ? '📊' : '📄';
                const fileName = file.name;
                
                fileItem.innerHTML = `
                    <div class="file-info">
                        <span class="file-icon">${icon}</span>
                        <span class="file-name">${fileName}</span>
                    </div>
                    <a href="${file.path}" download="${fileName}" class="file-download">⬇️</a>
                `;
                
                filesList.appendChild(fileItem);
            });
        } else {
            filesList.innerHTML = '<p class="no-files">No files generated yet</p>';
        }
    } catch (error) {
        console.error('Error loading files:', error);
    }
}


// Generate particles
function createParticles() {
    const bgAnimation = document.querySelector('.bg-animation');
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = particle.style.height = Math.random() * 4 + 2 + 'px';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        bgAnimation.appendChild(particle);
    }
}

createParticles();


// Initialize particle system when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new ParticleSystem('particleCanvas');
    });
} else {
    new ParticleSystem('particleCanvas');
}

        // Generate particles
        function createParticles() {

            const bgAnimation = document.querySelector('.bg-animation');
            for (let i = 0; i < 20; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.width = particle.style.height = Math.random() * 4 + 2 + 'px';
                particle.style.animationDelay = Math.random() * 15 + 's';
                particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
                bgAnimation.appendChild(particle);
            }
            
        }

        createParticles();

        // Initialize WebSocket
        function connectWebSocket() {
            sessionId = 'session_' + Date.now();
            // adjust ws URL if needed
            ws = new WebSocket(`ws://localhost:5500/ws/${sessionId}`);

            ws.onopen = () => {
                connectionStatus.classList.remove('disconnected');
                statusText.textContent = 'Connected';
                sessionInfo.textContent = `Session: ${sessionId.substring(0, 20)}...`;
                console.log('WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('Received:', data);

                    if (data.type === 'session_created') {
                        sessionId = data.session_id;
                        sessionInfo.textContent = `Session: ${sessionId.substring(0, 20)}...`;
                    } else if (data.type === 'status') {
                        showTypingIndicator(true);
                    } else if (data.type === 'message') {
                        showTypingIndicator(false);
                        addMessage('assistant', data.content, data.intent, data.data);
                    } else if (data.type === 'error') {
                        showTypingIndicator(false);
                        addMessage('assistant', '❌ ' + data.content);
                    } else {
                        // handle other message types gracefully
                        showTypingIndicator(false);
                        if (data.content) addMessage('assistant', data.content);
                    }
                } catch (err) {
                    console.error('Invalid message format', err);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                connectionStatus.classList.add('disconnected');
                statusText.textContent = 'Connection Error';
            };

            ws.onclose = () => {
                connectionStatus.classList.add('disconnected');
                statusText.textContent = 'Disconnected';
                console.log('WebSocket disconnected');
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!ws || ws.readyState === WebSocket.CLOSED) {
                        connectWebSocket();
                    }
                }, 3000);
            };
        }

        // Send message
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

            addMessage('user', message);

            ws.send(JSON.stringify({ message: message }));

            messageInput.value = '';
            showTypingIndicator(true);
        }

        // Quick message
        function sendQuickMessage(message) {
            messageInput.value = message;
            sendMessage();
        }
        // Expose to global scope for inline onclick handlers
        window.sendQuickMessage = sendQuickMessage;

        // Add message to chat
        // function addMessage(role, content, intent = null, data = null) {
        //     // Remove welcome message if exists
        //     const welcomeMsg = document.querySelector('.welcome-message');
        //     if (welcomeMsg) {
        //         welcomeMsg.remove();
        //     }

        //     const messageDiv = document.createElement('div');
        //     messageDiv.className = `message ${role}`;

        //     const avatar = document.createElement('div');
        //     avatar.className = 'message-avatar';
        //     avatar.textContent = role === 'user' ? '👤' : '🤖';

        //     const contentDiv = document.createElement('div');
        //     contentDiv.className = 'message-content';

        //     // Format content
        //     let formattedContent = content;

        //     // Add extra info for certain intents
        //     if (data) {
        //         if (data.chart_path) {
        //             formattedContent += `\n\n📊 Chart saved to: ${data.chart_path}`;
        //         }
        //         if (data.metrics && Object.keys(data.metrics).length > 0) {
        //             formattedContent += '\n\n📈 Key Metrics Retrieved';
        //         }
        //         if (data.news && data.news.length > 0) {
        //             formattedContent += `\n\n📰 Found ${data.news.length} news articles`;
        //         }
        //     }

        //     // Use innerHTML to preserve formatting, but sanitize to prevent XSS
        //     contentDiv.innerHTML = formattedContent
        //         .replace(/&/g, "&amp;")
        //         .replace(/</g, "&lt;")
        //         .replace(/>/g, "&gt;")
        //         .replace(/\n/g, "<br>");

        //     messageDiv.appendChild(avatar);
        //     messageDiv.appendChild(contentDiv);

        //     messagesContainer.appendChild(messageDiv);
        //     messagesContainer.scrollTop = messagesContainer.scrollHeight;
        // }
        function addMessage(role, content, intent = null, data = null) {
    // Remove welcome message if exists
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Format basic content
    let htmlContent = content
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\n/g, "<br>");

    contentDiv.innerHTML = htmlContent;

    // Handle different intents with rich content
    if (role === 'assistant' && data) {
        
        // 1. STOCK CHART - Display image inline
        if (intent === 'get_stock_data_and_chart' && data.chart_url) {
            const chartImg = document.createElement('img');
            chartImg.src = data.chart_url;
            chartImg.className = 'stock-chart-image';
            chartImg.alt = 'Stock Chart';
            contentDiv.appendChild(chartImg);
            
            // Add metrics table if available
            if (data.metrics) {
                contentDiv.appendChild(createMetricsTable(data.metrics));
            }
        }

        // 2. NEWS - Create carousel widget
        if (intent === 'get_financial_news' && data.news && data.news.length > 0) {
            contentDiv.appendChild(createNewsCarousel(data.news));
        }

        // 3. SEC FILING - Format as expandable section
        if (intent === 'get_sec_filing_section' && data.filing_content) {
            contentDiv.appendChild(createFilingSection(data.filing_content, data.filing_info));
        }

        // 4. REPORT - Add download button
        if (intent === 'get_report' && data.report_url) {
            contentDiv.appendChild(createDownloadButton(data.report_url));
        }
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Create metrics table
function createMetricsTable(metrics) {
    const table = document.createElement('div');
    table.className = 'metrics-table';
    
    const rows = [
        { label: '💵 Current Price', value: metrics.current_price ? `$${metrics.current_price}` : 'N/A' },
        { label: '📊 Market Cap', value: formatMarketCap(metrics.market_cap) },
        { label: '📈 P/E Ratio', value: metrics.pe_ratio || 'N/A' },
        { label: '⬆️ 52W High', value: metrics['52_week_high'] ? `$${metrics['52_week_high']}` : 'N/A' },
        { label: '⬇️ 52W Low', value: metrics['52_week_low'] ? `$${metrics['52_week_low']}` : 'N/A' }
    ];
    
    rows.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'metric-row';
        rowDiv.innerHTML = `<span class="metric-label">${row.label}</span><span class="metric-value">${row.value}</span>`;
        table.appendChild(rowDiv);
    });
    
    return table;
}

function formatMarketCap(value) {
    if (!value || value === 'N/A') return 'N/A';
    if (value >= 1e12) return `$${(value/1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value/1e9).toFixed(2)}B`;
    return `$${(value/1e6).toFixed(2)}M`;
}

// Create news carousel
function createNewsCarousel(newsArticles) {
    const carousel = document.createElement('div');
    carousel.className = 'news-carousel';
    
    newsArticles.forEach((article, index) => {
        const card = document.createElement('div');
        card.className = 'news-card';
        
        const title = document.createElement('h4');
        title.className = 'news-title';
        title.textContent = article.title;
        
        const meta = document.createElement('div');
        meta.className = 'news-meta';
        meta.innerHTML = `<span>📰 ${article.source}</span> • <span>📅 ${formatDate(article.published_at)}</span>`;
        
        const snippet = document.createElement('p');
        snippet.className = 'news-snippet';
        snippet.textContent = article.content_snippet || 'No preview available';
        
        const link = document.createElement('a');
        link.href = article.url;
        link.target = '_blank';
        link.className = 'news-link';
        link.textContent = 'Read More →';
        
        card.appendChild(title);
        card.appendChild(meta);
        card.appendChild(snippet);
        card.appendChild(link);
        carousel.appendChild(card);
    });
    
    return carousel;
}

function formatDate(dateStr) {
    if (!dateStr) return 'Unknown';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// Create filing section (expandable)
function createFilingSection(content, info) {
    const section = document.createElement('div');
    section.className = 'filing-section';
    
    const preview = document.createElement('div');
    preview.className = 'filing-preview';
    preview.textContent = content.substring(0, 800) + (content.length > 800 ? '...' : '');
    
    if (content.length > 800) {
        const expandBtn = document.createElement('button');
        expandBtn.className = 'expand-btn';
        expandBtn.textContent = 'Show More';
        expandBtn.onclick = () => {
            if (preview.classList.contains('expanded')) {
                preview.textContent = content.substring(0, 800) + '...';
                preview.classList.remove('expanded');
                expandBtn.textContent = 'Show More';
            } else {
                preview.textContent = content;
                preview.classList.add('expanded');
                expandBtn.textContent = 'Show Less';
            }
        };
        section.appendChild(preview);
        section.appendChild(expandBtn);
    } else {
        section.appendChild(preview);
    }
    
    return section;
}

// Create download button for reports
function createDownloadButton(reportUrl) {
    const btnContainer = document.createElement('div');
    btnContainer.className = 'download-container';
    
    const btn = document.createElement('a');
    btn.href = reportUrl;
    btn.download = reportUrl.split('/').pop();
    btn.className = 'download-btn';
    btn.innerHTML = '📄 Download Full Report';
    
    btnContainer.appendChild(btn);
    return btnContainer;
}

        // Show/hide typing indicator
        function showTypingIndicator(show) {
            typingIndicator.classList.toggle('active', show);
            if (show) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        // Clear chat
        clearChatBtn.addEventListener('click', () => {
            // preserve session info but clear messages area and re-show welcome
            messagesContainer.innerHTML = '';
            const welcome = document.createElement('div');
            welcome.className = 'welcome-message';
            welcome.innerHTML = `
                <div class="welcome-title">👋 Welcome to FinSight</div>
                <p>Your AI-powered financial analyst assistant. Ask me anything about stocks, financial news, SEC filings, or request comprehensive analyst reports.</p>
                <div class="quick-actions" id="quickActions">
                    <button class="quick-action-btn" onclick="sendQuickMessage('Show me Apple stock chart')">📊 Apple Stock Chart</button>
                    <button class="quick-action-btn" onclick="sendQuickMessage('Recent news about Tesla')">📰 Tesla News</button>
                    <button class="quick-action-btn" onclick="sendQuickMessage('What is the P/E ratio for NVIDIA?')">💹 NVIDIA P/E Ratio</button>
                    <button class="quick-action-btn" onclick="sendQuickMessage('Generate report for Microsoft')">📄 Microsoft Report</button>
                </div>
            `;
            messagesContainer.appendChild(welcome);
            messagesContainer.scrollTop = 0;
        });

        // Enter key to send
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Connect on load
        window.addEventListener('load', () => {
            connectWebSocket();
        });
    // </script>