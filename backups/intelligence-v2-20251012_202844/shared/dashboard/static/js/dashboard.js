// Agent Zero Dashboard - Real-time WebSocket Client

class AgentDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 2000;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.loadAgents();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Wyślij ping co 30s żeby utrzymać połączenie
                this.pingInterval = setInterval(() => {
                    if (this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send('ping');
                    }
                }, 30000);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.updateDashboard(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                clearInterval(this.pingInterval);
                
                // Próba reconnect
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connectWebSocket(), this.reconnectDelay);
                }
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-indicator');
        const text = document.getElementById('connection-text');
        
        if (connected) {
            indicator.className = 'indicator connected';
            text.textContent = 'Połączono';
        } else {
            indicator.className = 'indicator disconnected';
            text.textContent = 'Rozłączono';
        }
    }
    
    updateDashboard(data) {
        // Update timestamp
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString('pl-PL');
        
        // System status
        const status = data.status || 'unknown';
        const statusEl = document.getElementById('system-status');
        statusEl.textContent = this.translateStatus(status);
        statusEl.className = `metric-value status-${status}`;
        
        // Metrics
        document.getElementById('total-agents').textContent = data.total_agents || 0;
        document.getElementById('total-tasks').textContent = data.total_tasks_completed || 0;
        document.getElementById('total-messages').textContent = data.total_messages || 0;
        document.getElementById('total-errors').textContent = data.total_errors || 0;
        
        // Error rate
        const errorRate = (data.error_rate || 0) * 100;
        document.getElementById('error-rate').textContent = errorRate.toFixed(2) + '%';
        
        // Agent states distribution
        if (data.state_distribution) {
            this.updateStatesGrid(data.state_distribution);
        }
        
        // Reload agents list
        this.loadAgents();
    }
    
    updateStatesGrid(states) {
        const grid = document.getElementById('states-grid');
        grid.innerHTML = '';
        
        const stateLabels = {
            'created': 'Utworzone',
            'initializing': 'Inicjalizacja',
            'ready': 'Gotowe',
            'busy': 'Zajęte',
            'idle': 'Bezczynne',
            'paused': 'Wstrzymane',
            'error': 'Błąd',
            'terminated': 'Zakończone'
        };
        
        for (const [state, count] of Object.entries(states)) {
            if (count === 0) continue;
            
            const badge = document.createElement('div');
            badge.className = 'state-badge';
            badge.innerHTML = `
                <div class="state-name">${stateLabels[state] || state}</div>
                <div class="state-count">${count}</div>
            `;
            grid.appendChild(badge);
        }
        
        if (grid.children.length === 0) {
            grid.innerHTML = '<p class="loading">Brak aktywnych agentów</p>';
        }
    }
    
    async loadAgents() {
        try {
            const response = await fetch('/api/agents');
            const data = await response.json();
            
            const container = document.getElementById('agents-container');
            
            if (!data.agents || data.agents.length === 0) {
                container.innerHTML = '<p class="loading">Brak aktywnych agentów</p>';
                return;
            }
            
            container.innerHTML = '';
            
            data.agents.forEach(agent => {
                const card = this.createAgentCard(agent);
                container.appendChild(card);
            });
            
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }
    
    createAgentCard(agent) {
        const card = document.createElement('div');
        card.className = 'agent-card';
        
        const stateClass = agent.state.replace('_', '-');
        
        card.innerHTML = `
            <div class="agent-card-header">
                <span class="agent-id">${agent.agent_id}</span>
                <span class="agent-state ${stateClass}">${this.translateState(agent.state)}</span>
            </div>
            <div class="agent-type">
                <strong>Typ:</strong> ${agent.agent_type}
            </div>
            <div class="agent-metrics">
                <div class="agent-metric">
                    <span>Zadania:</span>
                    <span class="value">${agent.tasks_completed}</span>
                </div>
                <div class="agent-metric">
                    <span>Błędy:</span>
                    <span class="value">${agent.tasks_failed}</span>
                </div>
                <div class="agent-metric">
                    <span>Wysłane:</span>
                    <span class="value">${agent.messages_sent}</span>
                </div>
                <div class="agent-metric">
                    <span>Odebrane:</span>
                    <span class="value">${agent.messages_received}</span>
                </div>
                <div class="agent-metric">
                    <span>Uptime:</span>
                    <span class="value">${this.formatUptime(agent.uptime)}</span>
                </div>
                <div class="agent-metric">
                    <span>Błędy sys:</span>
                    <span class="value">${agent.error_count}</span>
                </div>
            </div>
        `;
        
        return card;
    }
    
    translateStatus(status) {
        const translations = {
            'healthy': '✅ Zdrowy',
            'degraded': '⚠️ Obniżona wydajność',
            'no_agents': '💤 Brak agentów',
            'unknown': '❓ Nieznany'
        };
        return translations[status] || status;
    }
    
    translateState(state) {
        const translations = {
            'created': 'Utworzony',
            'initializing': 'Inicjalizacja',
            'ready': 'Gotowy',
            'busy': 'Zajęty',
            'idle': 'Bezczynny',
            'paused': 'Wstrzymany',
            'error': 'Błąd',
            'terminated': 'Zakończony'
        };
        return translations[state] || state;
    }
    
    formatUptime(seconds) {
        if (!seconds || seconds < 1) return '0s';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    setupEventListeners() {
        // Refresh button (można dodać)
        // Auto-reconnect on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.ws.readyState !== WebSocket.OPEN) {
                this.connectWebSocket();
            }
        });
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Agent Zero Dashboard initializing...');
    new AgentDashboard();
});
