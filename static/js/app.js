class ImageDeduplicationApp {
    constructor() {
        this.currentFile = null;
        this.initializeElements();
        this.bindEvents();
        this.loadImageHistory();
    }

    initializeElements() {
        this.uploadBox = document.getElementById('uploadBox');
        this.fileInput = document.getElementById('fileInput');
        this.previewSection = document.getElementById('previewSection');
        this.previewImg = document.getElementById('previewImg');
        this.previewInfo = document.getElementById('previewInfo');
        this.resultSection = document.getElementById('resultSection');
        this.resultSummary = document.getElementById('resultSummary');
        this.similarList = document.getElementById('similarList');
        this.imageGrid = document.getElementById('imageGrid');
        this.loadingModal = document.getElementById('loadingModal');
        this.loadingText = document.getElementById('loadingText');
        this.toast = document.getElementById('toast');
        this.refreshBtn = document.getElementById('refreshBtn');
    }

    bindEvents() {
        this.uploadBox.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        this.uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadBox.classList.add('dragover');
        });

        this.uploadBox.addEventListener('dragleave', () => {
            this.uploadBox.classList.remove('dragover');
        });

        this.uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadBox.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });

        this.refreshBtn.addEventListener('click', () => this.loadImageHistory());
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
        const allowedExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];
        
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            this.showToast('不支持的文件格式', 'error');
            return;
        }

        if (file.size > 50 * 1024 * 1024) {
            this.showToast('文件大小超过50MB限制', 'error');
            return;
        }

        this.currentFile = file;
        this.previewFile(file);
    }

    previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.previewSection.style.display = 'block';
            
            this.previewInfo.innerHTML = `
                <p><strong>文件名:</strong> ${file.name}</p>
                <p><strong>文件大小:</strong> ${this.formatFileSize(file.size)}</p>
                <p><strong>文件类型:</strong> ${file.type || '未知'}</p>
            `;
            
            this.uploadFile(file);
        };
        reader.readAsDataURL(file);
    }

    async uploadFile(file) {
        this.showLoading('正在上传并处理图片...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/images/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result);
                this.loadImageHistory();
                this.showToast('图片上传成功', 'success');
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('上传失败，请重试', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        this.resultSection.style.display = 'block';

        const hasDuplicates = result.similar_images && 
            result.similar_images.some(img => img.is_duplicate === 1);

        let summaryClass = 'success';
        let summaryTitle = '检测完成';
        let summaryMessage = result.message;

        if (hasDuplicates) {
            summaryClass = 'warning';
            summaryTitle = '⚠️ 检测到重复图片！';
        }

        this.resultSummary.className = `result-summary ${summaryClass}`;
        this.resultSummary.innerHTML = `
            <h3>${summaryTitle}</h3>
            <p>${summaryMessage}</p>
            <p>相似度阈值: 50% (超过此值视为有重复嫌疑)</p>
        `;

        if (result.similar_images && result.similar_images.length > 0) {
            this.similarList.innerHTML = result.similar_images.map(img => {
                const similarityPercent = (img.similarity_score * 100).toFixed(2);
                const isDuplicate = img.is_duplicate === 1;
                const itemClass = isDuplicate ? 'duplicate' : 'safe';
                
                let progressClass = 'low';
                if (img.similarity_score >= 0.7) {
                    progressClass = 'high';
                } else if (img.similarity_score >= 0.5) {
                    progressClass = 'medium';
                }

                return `
                    <div class="similar-item ${itemClass}">
                        <div class="filename">${img.image_id_2 || '图片ID: ' + img.image_id_2}</div>
                        <div class="similarity">
                            <span>相似度:</span>
                            <span style="font-weight: bold; color: ${isDuplicate ? '#dc3545' : '#28a745'}">
                                ${similarityPercent}%
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${progressClass}" 
                                 style="width: ${similarityPercent}%"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: ${isDuplicate ? '#dc3545' : '#28a745'}">
                            ${isDuplicate ? '⚠️ 有重复嫌疑' : '✓ 相似度较低'}
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            this.similarList.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #666;">未检测到相似图片</p>';
        }
    }

    async loadImageHistory() {
        this.imageGrid.innerHTML = '<div class="loading">加载中...</div>';

        try {
            const response = await fetch('/api/images/all');
            const images = await response.json();

            if (images.length === 0) {
                this.imageGrid.innerHTML = `
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                        <p>暂无上传的图片</p>
                        <p style="font-size: 0.9rem;">请上传第一张图片开始检测</p>
                    </div>
                `;
                return;
            }

            this.imageGrid.innerHTML = images.map(img => `
                <div class="image-card" data-id="${img.id}">
                    <div class="card-image">
                        <img src="/api/images/file/${img.id}" alt="${img.original_filename}" 
                             onerror="this.style.display='none'">
                    </div>
                    <div class="card-info">
                        <div class="filename">${img.original_filename}</div>
                        <div class="size">${this.formatFileSize(img.file_size)}</div>
                        <div class="card-actions">
                            <button class="btn-check" onclick="app.checkSimilarity(${img.id})">
                                检测相似
                            </button>
                            <button class="btn-delete" onclick="app.deleteImage(${img.id})">
                                删除
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');

        } catch (error) {
            console.error('Load history error:', error);
            this.imageGrid.innerHTML = '<div class="loading" style="color: #dc3545;">加载失败，请刷新重试</div>';
        }
    }

    async checkSimilarity(imageId) {
        this.showLoading('正在检测相似图片...');

        try {
            const response = await fetch(`/api/images/${imageId}/similar`);
            const result = await response.json();

            if (result.success) {
                this.displaySimilarityResult(result);
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Check similarity error:', error);
            this.showToast('检测失败，请重试', 'error');
        } finally {
            this.hideLoading();
        }
    }

    displaySimilarityResult(result) {
        this.resultSection.style.display = 'block';
        this.previewSection.style.display = 'none';

        let summaryClass = result.is_duplicate ? 'warning' : 'success';
        let summaryTitle = result.is_duplicate ? '⚠️ 检测到重复图片！' : '检测完成';

        this.resultSummary.className = `result-summary ${summaryClass}`;
        this.resultSummary.innerHTML = `
            <h3>${summaryTitle}</h3>
            <p>${result.message}</p>
            <p>相似度阈值: 50% (超过此值视为有重复嫌疑)</p>
        `;

        if (result.similar_images && result.similar_images.length > 0) {
            this.similarList.innerHTML = result.similar_images.map(img => {
                const similarityPercent = img.similarity_percentage;
                const isDuplicate = img.is_duplicate;
                const itemClass = isDuplicate ? 'duplicate' : 'safe';
                
                let progressClass = 'low';
                if (img.similarity_score >= 0.7) {
                    progressClass = 'high';
                } else if (img.similarity_score >= 0.5) {
                    progressClass = 'medium';
                }

                return `
                    <div class="similar-item ${itemClass}">
                        <div class="filename">${img.original_filename}</div>
                        <div class="similarity">
                            <span>相似度:</span>
                            <span style="font-weight: bold; color: ${isDuplicate ? '#dc3545' : '#28a745'}">
                                ${similarityPercent}%
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${progressClass}" 
                                 style="width: ${similarityPercent}%"></div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: ${isDuplicate ? '#dc3545' : '#28a745'}">
                            ${isDuplicate ? '⚠️ 有重复嫌疑' : '✓ 相似度较低'}
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            this.similarList.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #666;">未检测到相似图片</p>';
        }

        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    async deleteImage(imageId) {
        if (!confirm('确定要删除这张图片吗？')) {
            return;
        }

        try {
            const response = await fetch(`/api/images/${imageId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.showToast('图片已删除', 'success');
                this.loadImageHistory();
            } else {
                this.showToast('删除失败', 'error');
            }
        } catch (error) {
            console.error('Delete error:', error);
            this.showToast('删除失败，请重试', 'error');
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showLoading(text = '加载中...') {
        this.loadingText.textContent = text;
        this.loadingModal.classList.add('show');
    }

    hideLoading() {
        this.loadingModal.classList.remove('show');
    }

    showToast(message, type = 'success') {
        this.toast.textContent = message;
        this.toast.className = `toast ${type} show`;

        setTimeout(() => {
            this.toast.classList.remove('show');
        }, 3000);
    }
}

const app = new ImageDeduplicationApp();
