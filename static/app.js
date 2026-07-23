// AVF Research Assistant 前端应用
class AVFResearchApp {
    constructor() {
        this.apiBaseUrl = '/api';
        this.currentMode = 'quick';
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.currentChatHistory = [];
        this.chatHistories = this.loadChatHistories();
        this.isCurrentChatFromHistory = false;

        this.initializeElements();
        this.bindEvents();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    initMarkdown() {
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
                if (typeof hljs !== 'undefined') {
                    marked.setOptions({
                        highlight: function(code, lang) {
                            if (lang && hljs.getLanguage(lang)) {
                                try { return hljs.highlight(code, { language: lang }).value; } catch (err) {}
                            }
                            return code;
                        }
                    });
                }
            } else {
                setTimeout(checkMarked, 100);
            }
        };
        checkMarked();
    }

    renderMarkdown(content) {
        if (!content) return '';
        if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
            return this.escapeHtml(content);
        }
        try {
            const rendered = marked.parse(content);
            return DOMPurify.sanitize(rendered, {
                USE_PROFILES: { html: true },
                FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
                FORBID_ATTR: ['style']
            });
        } catch (e) {
            return this.escapeHtml(content);
        }
    }

    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            container.querySelectorAll('pre code').forEach(block => {
                if (!block.classList.contains('hljs')) hljs.highlightElement(block);
            });
        }
    }

    initializeElements() {
        this.sidebar = document.querySelector('.sidebar');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.toolsBtn = document.getElementById('toolsBtn');
        this.toolsMenu = document.getElementById('toolsMenu');
        this.uploadFileItem = document.getElementById('uploadFileItem');
        this.modeSelectorBtn = document.getElementById('modeSelectorBtn');
        this.modeDropdown = document.getElementById('modeDropdown');
        this.currentModeText = document.getElementById('currentModeText');
        this.fileInput = document.getElementById('fileInput');
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.chatContainer = document.querySelector('.chat-container');
        this.welcomeGreeting = document.getElementById('welcomeGreeting');
        this.chatHistoryList = document.getElementById('chatHistoryList');
        this.checkAndSetCentered();
    }

    bindEvents() {
        if (this.newChatBtn) this.newChatBtn.addEventListener('click', () => this.newChat());
        if (this.modeSelectorBtn) {
            this.modeSelectorBtn.addEventListener('click', (e) => { e.stopPropagation(); this.toggleModeDropdown(); });
        }
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = item.getAttribute('data-mode');
                this.selectMode(mode);
                this.closeModeDropdown();
            });
        });
        document.addEventListener('click', (e) => {
            if (this.modeSelectorBtn && this.modeDropdown &&
                !this.modeSelectorBtn.contains(e.target) && !this.modeDropdown.contains(e.target)) {
                this.closeModeDropdown();
            }
        });
        if (this.sendButton) this.sendButton.addEventListener('click', () => this.sendMessage());
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
            });
        }
        if (this.toolsBtn) {
            this.toolsBtn.addEventListener('click', (e) => { e.stopPropagation(); this.toggleToolsMenu(); });
        }
        if (this.uploadFileItem) {
            this.uploadFileItem.addEventListener('click', () => { if (this.fileInput) this.fileInput.click(); this.closeToolsMenu(); });
        }
        document.addEventListener('click', (e) => {
            if (this.toolsBtn && this.toolsMenu &&
                !this.toolsBtn.contains(e.target) && !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });
        if (this.fileInput) this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    toggleToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) wrapper.classList.toggle('active');
        }
    }

    closeToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) wrapper.classList.remove('active');
        }
    }

    newChat() {
        if (this.isStreaming) { this.showNotification('请等待当前对话完成后再新建对话', 'warning'); return; }
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) { this.updateCurrentChatHistory(); }
            else { this.saveCurrentChat(); }
        }
        this.isStreaming = false;
        if (this.messageInput) this.messageInput.value = '';
        this.currentChatHistory = [];
        this.isCurrentChatFromHistory = false;
        if (this.chatMessages) this.chatMessages.innerHTML = '';
        this.sessionId = this.generateSessionId();
        this.currentMode = 'quick';
        this.updateUI();
        this.checkAndSetCentered();
        if (this.chatContainer) this.chatContainer.style.transition = 'all 0.5s ease';
        this.renderChatHistory();
    }

    saveCurrentChat() {
        if (this.currentChatHistory.length === 0) return;
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex !== -1) { this.updateCurrentChatHistory(); return; }
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        const title = firstUserMessage ? (firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')) : '新对话';
        const chatHistory = { id: this.sessionId, title: title, messages: [...this.currentChatHistory], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
        this.chatHistories.unshift(chatHistory);
        if (this.chatHistories.length > 50) this.chatHistories = this.chatHistories.slice(0, 50);
        this.saveChatHistories();
    }

    updateCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) return;
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) { this.saveCurrentChat(); return; }
        const history = this.chatHistories[existingIndex];
        history.messages = [...this.currentChatHistory];
        history.updatedAt = new Date().toISOString();
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            const newTitle = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
            if (history.title !== newTitle) history.title = newTitle;
        }
        this.saveChatHistories();
    }

    loadChatHistories() {
        try { const stored = localStorage.getItem('chatHistories'); return stored ? JSON.parse(stored) : []; }
        catch (e) { return []; }
    }

    saveChatHistories() {
        try { localStorage.setItem('chatHistories', JSON.stringify(this.chatHistories)); }
        catch (e) { console.error('保存历史对话失败:', e); }
    }

    persistCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) return;
        this.updateCurrentChatHistory();
        this.renderChatHistory();
    }

    renderChatHistory() {
        if (!this.chatHistoryList) return;
        this.chatHistoryList.innerHTML = '';
        if (this.chatHistories.length === 0) return;
        this.chatHistories.forEach((history) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.historyId = history.id;
            historyItem.innerHTML =
                '<div class="history-item-content"><span class="history-item-title">' + this.escapeHtml(history.title) + '</span></div>' +
                '<button class="history-item-delete" data-history-id="' + history.id + '" title="删除">' +
                '<svg viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>' +
                '</button>';
            historyItem.addEventListener('click', (e) => { if (!e.target.closest('.history-item-delete')) this.loadChatHistory(history.id); });
            const deleteBtn = historyItem.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', (e) => { e.stopPropagation(); this.deleteChatHistory(history.id); });
            this.chatHistoryList.appendChild(historyItem);
        });
    }

    async loadChatHistory(historyId) {
        const history = this.chatHistories.find(h => h.id === historyId);
        if (!history) return;
        if (this.currentChatHistory.length > 0 && this.sessionId !== historyId) {
            if (this.isCurrentChatFromHistory) this.updateCurrentChatHistory();
            else this.saveCurrentChat();
        }
        try {
            const response = await fetch('/api/chat/session/' + historyId);
            if (response.ok) {
                const data = await response.json();
                const backendHistory = data.history || [];
                this.sessionId = history.id;
                this.isCurrentChatFromHistory = true;
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';
                    if (backendHistory.length > 0) {
                        this.currentChatHistory = [];
                        backendHistory.forEach(msg => {
                            const messageType = msg.role === 'user' ? 'user' : 'assistant';
                            this.addMessage(messageType, msg.content, false, false);
                        });
                    } else {
                        this.currentChatHistory = [...history.messages];
                        history.messages.forEach(msg => this.addMessage(msg.type, msg.content, false, false));
                    }
                }
            } else {
                this.sessionId = history.id;
                this.currentChatHistory = [...history.messages];
                this.isCurrentChatFromHistory = true;
                if (this.chatMessages) { this.chatMessages.innerHTML = ''; history.messages.forEach(msg => this.addMessage(msg.type, msg.content, false, false)); }
            }
        } catch (error) {
            this.sessionId = history.id;
            this.currentChatHistory = [...history.messages];
            this.isCurrentChatFromHistory = true;
            if (this.chatMessages) { this.chatMessages.innerHTML = ''; history.messages.forEach(msg => this.addMessage(msg.type, msg.content, false, false)); }
        }
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    async deleteChatHistory(historyId) {
        try {
            const response = await fetch('/api/chat/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: historyId }) });
            if (!response.ok) throw new Error('清空会话失败');
            const result = await response.json();
            if (result.status === 'success') {
                this.chatHistories = this.chatHistories.filter(h => h.id !== historyId);
                this.saveChatHistories();
                this.renderChatHistory();
                if (this.sessionId === historyId) {
                    this.currentChatHistory = [];
                    if (this.chatMessages) this.chatMessages.innerHTML = '';
                    this.sessionId = this.generateSessionId();
                    this.checkAndSetCentered();
                }
                this.showNotification('会话已清空', 'success');
            }
        } catch (error) { this.showNotification('删除失败: ' + error.message, 'error'); }
    }

    toggleModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) wrapper.classList.toggle('active');
        }
    }

    closeModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) wrapper.classList.remove('active');
        }
    }

    selectMode(mode) {
        if (this.isStreaming) { this.showNotification('请等待当前对话完成后再切换模式', 'warning'); return; }
        this.currentMode = mode;
        this.updateUI();
        this.showNotification('已切换到' + (mode === 'quick' ? '快速' : '流式') + '模式', 'info');
    }

    updateUI() {
        if (this.currentModeText) {
            this.currentModeText.textContent = this.currentMode === 'quick' ? '快速' : '流式';
        }
        document.querySelectorAll('.dropdown-item').forEach(item => {
            const mode = item.getAttribute('data-mode');
            if (mode === this.currentMode) item.classList.add('active'); else item.classList.remove('active');
        });
        if (this.sendButton) this.sendButton.disabled = this.isStreaming;
        if (this.messageInput) {
            this.messageInput.disabled = this.isStreaming;
            this.messageInput.placeholder = '输入你的科研问题...';
        }
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    async sendMessage() {
        var message = '';
        if (this.messageInput) message = this.messageInput.value.trim();
        if (!message) { this.showNotification('请输入消息内容', 'warning'); return; }
        if (this.isStreaming) { this.showNotification('请等待当前对话完成', 'warning'); return; }
        this.addMessage('user', message);
        // 新对话在首条用户消息发出后立即进入“近期对话”，无需等待切换会话。
        this.persistCurrentChatHistory();
        if (this.messageInput) this.messageInput.value = '';
        this.isStreaming = true;
        this.updateUI();
        try {
            if (this.currentMode === 'quick') await this.sendQuickMessage(message);
            else if (this.currentMode === 'stream') await this.sendStreamMessage(message);
        } catch (error) {
            this.addMessage('assistant', '抱歉，发送消息时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();
            // 无论是新对话还是从历史列表打开的对话，都保存本轮最终状态。
            this.persistCurrentChatHistory();
        }
    }

    async sendQuickMessage(message) {
        var loadingMessage = this.addLoadingMessage('正在思考...');
        try {
            var response = await fetch(this.apiBaseUrl + '/chat', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ Id: this.sessionId, Question: message })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            var data = await response.json();
            if (loadingMessage && loadingMessage.parentNode) loadingMessage.parentNode.removeChild(loadingMessage);
            if (data.code === 200 || data.message === 'success') {
                var chatResponse = data.data;
                if (chatResponse && chatResponse.success) {
                    this.addMessage('assistant', chatResponse.answer || '（无回复内容）');
                } else if (chatResponse && chatResponse.errorMessage) {
                    throw new Error(chatResponse.errorMessage);
                } else {
                    this.addMessage('assistant', (chatResponse && chatResponse.answer) || (chatResponse && chatResponse.errorMessage) || '服务返回了空内容');
                }
            } else {
                throw new Error(data.message || '请求失败');
            }
        } catch (error) {
            if (loadingMessage && loadingMessage.parentNode) loadingMessage.parentNode.removeChild(loadingMessage);
            throw error;
        }
    }

    async sendStreamMessage(message) {
        try {
            var response = await fetch(this.apiBaseUrl + '/chat_stream', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ Id: this.sessionId, Question: message })
            });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            var assistantMessageElement = this.addMessage('assistant', '', true);
            var fullResponse = '';
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';
            var self = this;
            try {
                while (true) {
                    var result = await reader.read();
                    if (result.done) { self.handleStreamComplete(assistantMessageElement, fullResponse); break; }
                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        if (line.trim() === '') continue;
                        if (line.indexOf('data:') === 0) {
                            var rawData = line.substring(5).trim();
                            if (rawData === '[DONE]') { self.handleStreamComplete(assistantMessageElement, fullResponse); return; }
                            try {
                                var sseMessage = JSON.parse(rawData);
                                if (sseMessage && sseMessage.type === 'content') {
                                    fullResponse += sseMessage.data || '';
                                    if (assistantMessageElement) {
                                        var mc = assistantMessageElement.querySelector('.message-content');
                                        mc.innerHTML = self.renderMarkdown(fullResponse);
                                        self.highlightCodeBlocks(mc);
                                        self.scrollToBottom();
                                    }
                                } else if (sseMessage.type === 'tool_start') {
                                    if (assistantMessageElement && !fullResponse) {
                                        assistantMessageElement.querySelector('.message-content').textContent = '正在检索知识库...';
                                    }
                                } else if (sseMessage.type === 'retrieval_complete') {
                                    if (assistantMessageElement && !fullResponse) {
                                        var selectedCount = sseMessage.data && sseMessage.data.selected_count;
                                        assistantMessageElement.querySelector('.message-content').textContent = selectedCount ? ('已找到 ' + selectedCount + ' 条证据，正在生成回答...') : '检索完成，正在生成回答...';
                                    }
                                } else if (sseMessage.type === 'done') {
                                    self.handleStreamComplete(assistantMessageElement, fullResponse);
                                    return;
                                } else if (sseMessage.type === 'error') {
                                    if (assistantMessageElement) {
                                        assistantMessageElement.querySelector('.message-content').innerHTML = self.renderMarkdown('错误: ' + (sseMessage.data || '未知错误'));
                                    }
                                    return;
                                }
                            } catch (e) {
                                fullResponse += rawData;
                                if (assistantMessageElement) {
                                    assistantMessageElement.querySelector('.message-content').innerHTML = self.renderMarkdown(fullResponse);
                                    self.scrollToBottom();
                                }
                            }
                        }
                    }
                }
            } finally { reader.releaseLock(); }
        } catch (error) { throw error; }
    }

    addMessage(type, content, isStreaming, saveToHistory) {
        if (isStreaming === undefined) isStreaming = false;
        if (saveToHistory === undefined) saveToHistory = true;
        var isFirstMessage = this.chatMessages && this.chatMessages.querySelectorAll('.message').length === 0;
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({ type: type, content: content, timestamp: new Date().toISOString() });
        }
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + type + (isStreaming ? ' streaming' : '');
        if (type === 'assistant') {
            var messageAvatar = document.createElement('div');
            messageAvatar.className = 'message-avatar';
            messageAvatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/></svg>';
            messageDiv.appendChild(messageAvatar);
        }
        var messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';
        var messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        if (type === 'assistant' && !isStreaming) {
            messageContent.innerHTML = this.renderMarkdown(content);
            this.highlightCodeBlocks(messageContent);
        } else {
            messageContent.textContent = content;
        }
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);
        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            this.scrollToBottom();
        }
        return messageDiv;
    }

    addLoadingMessage(content) {
        var messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        var messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/></svg>';
        messageDiv.appendChild(messageAvatar);
        var messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';
        var messageContent = document.createElement('div');
        messageContent.className = 'message-content loading-message-content';
        var textSpan = document.createElement('span');
        textSpan.textContent = content;
        var loadingIcon = document.createElement('span');
        loadingIcon.className = 'loading-spinner-icon';
        loadingIcon.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="currentColor" opacity="0.2"/><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c1.54 0 3-.36 4.28-1l-1.5-2.6C13.64 19.62 12.84 20 12 20c-4.41 0-8-3.59-8-8s3.59-8 8-8c.84 0 1.64.38 2.18 1l1.5-2.6C13 2.36 12.54 2 12 2z" fill="currentColor"/></svg>';
        messageContent.appendChild(textSpan);
        messageContent.appendChild(loadingIcon);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);
        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            var isFirstMessage = this.chatMessages.querySelectorAll('.message').length === 1;
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            this.scrollToBottom();
        }
        return messageDiv;
    }

    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            var hasMessages = this.chatMessages.querySelectorAll('.message').length > 0;
            if (!hasMessages) this.chatContainer.classList.add('centered');
            else this.chatContainer.classList.remove('centered');
        }
    }

    scrollToBottom() {
        if (this.chatMessages) this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    handleStreamComplete(assistantMessageElement, fullResponse) {
        if (assistantMessageElement) {
            assistantMessageElement.classList.remove('streaming');
            var messageContent = assistantMessageElement.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = this.renderMarkdown(fullResponse);
                this.highlightCodeBlocks(messageContent);
            }
        }
        if (fullResponse) {
            this.currentChatHistory.push({ type: 'assistant', content: fullResponse, timestamp: new Date().toISOString() });
            if (this.isCurrentChatFromHistory) { this.updateCurrentChatHistory(); this.renderChatHistory(); }
        }
    }

    showNotification(message, type) {
        if (type === undefined) type = 'info';
        var notification = document.createElement('div');
        notification.className = 'notification ' + type;
        notification.textContent = message;
        notification.style.cssText = 'position:fixed;top:20px;right:20px;padding:15px 20px;border-radius:8px;color:white;font-weight:500;z-index:10000;animation:slideIn 0.3s ease;max-width:300px;';
        var colors = { info: '#1a73e8', success: '#34a853', warning: '#fbbc04', error: '#ea4335' };
        notification.style.backgroundColor = colors[type] || colors.info;
        document.body.appendChild(notification);
        var self = this;
        setTimeout(function() {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(function() { if (notification.parentNode) notification.parentNode.removeChild(notification); }, 300);
        }, 3000);
    }

    handleFileSelect(event) {
        var file = event.target.files[0];
        if (file) {
            if (!this.validateFileType(file)) { this.showNotification('只支持上传 TXT、Markdown (.md) 或 PDF 文件', 'error'); this.fileInput.value = ''; return; }
            this.uploadFile(file);
        }
    }

    validateFileType(file) {
        var fileName = file.name.toLowerCase();
        var allowedExtensions = ['.txt', '.md', '.pdf'];
        return allowedExtensions.some(function(ext) { return fileName.endsWith(ext); });
    }

    async uploadFile(file) {
        if (!this.validateFileType(file)) { this.showNotification('只支持上传 TXT、Markdown (.md) 或 PDF 文件', 'error'); return; }
        var maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) { this.showNotification('文件大小不能超过10MB', 'error'); return; }
        this.isStreaming = true;
        this.updateUI();
        this.showUploadOverlay(true, file.name);
        try {
            var formData = new FormData();
            formData.append('file', file);
            var response = await fetch(this.apiBaseUrl + '/upload', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('HTTP错误: ' + response.status);
            var data = await response.json();
            if (data.data && data.code === 201 && data.data.status === 'uploaded') {
                this.addMessage('assistant', file.name + ' 上传成功，当前等待解析入库。请在对话中要求我解析并加入知识库。', false, true);
            } else if (data.data && data.data.index_success === true) {
                this.addMessage('assistant', file.name + ' 上传并写入知识库成功', false, true);
            } else if (data.data && data.data.upload_success === true) {
                throw new Error(data.data.index_error || '文件已保存，但索引失败');
            } else {
                throw new Error(data.message || '上传失败');
            }
        } catch (error) {
            this.showNotification('文件上传失败: ' + error.message, 'error');
        } finally {
            if (this.fileInput) this.fileInput.value = '';
            this.isStreaming = false;
            this.showUploadOverlay(false);
            this.updateUI();
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        var k = 1024;
        var sizes = ['Bytes', 'KB', 'MB', 'GB'];
        var i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    showLoadingOverlay(show) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                var loadingText = this.loadingOverlay.querySelector('.loading-text');
                var loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '处理中，请稍候...';
                if (loadingSubtext) loadingSubtext.textContent = '正在处理你的请求';
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                document.body.style.overflow = '';
            }
        }
    }

    showUploadOverlay(show, fileName) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                var loadingText = this.loadingOverlay.querySelector('.loading-text');
                var loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '正在上传文件...';
                if (loadingSubtext) loadingSubtext.textContent = fileName ? '上传: ' + fileName : '请稍候';
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                document.body.style.overflow = '';
            }
        }
    }

    escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// CSS 动画
var style = document.createElement('style');
style.textContent = '@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } } @keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }';
document.head.appendChild(style);

// 启动应用
document.addEventListener('DOMContentLoaded', function() {
    new AVFResearchApp();
});
