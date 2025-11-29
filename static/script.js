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
        function addMessage(role, content, intent = null, data = null) {
            // Remove welcome message if exists
            const welcomeMsg = document.querySelector('.welcome-message');
            if (welcomeMsg) {
                welcomeMsg.remove();
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? '👤' : '🤖';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            // Format content
            let formattedContent = content;

            // Add extra info for certain intents
            if (data) {
                if (data.chart_path) {
                    formattedContent += `\n\n📊 Chart saved to: ${data.chart_path}`;
                }
                if (data.metrics && Object.keys(data.metrics).length > 0) {
                    formattedContent += '\n\n📈 Key Metrics Retrieved';
                }
                if (data.news && data.news.length > 0) {
                    formattedContent += `\n\n📰 Found ${data.news.length} news articles`;
                }
            }

            // Use innerHTML to preserve formatting, but sanitize to prevent XSS
            contentDiv.innerHTML = formattedContent
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/\n/g, "<br>");

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
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