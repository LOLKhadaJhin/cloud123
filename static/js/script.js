// 全局变量
let statusInterval = null;

// 更新任务状态的函数
function updateTaskStatus() {
    fetch('/task_status')
        .then(response => response.json())
        .then(data => {
            const statusDiv = document.getElementById('statusMessage');
            const progressBar = document.getElementById('progressBar');
            const progressDetails = document.getElementById('progressDetails');
            const cancelBtn = document.getElementById('cancelBtn');

            if (data.status === 'idle') {
                statusDiv.innerHTML = '没有正在进行的任务';
                progressBar.style.width = '0%';
                progressDetails.innerHTML = '';
                cancelBtn.style.display = 'none';
            } else {
                // 更新状态消息
                statusDiv.innerHTML = `<strong>状态:</strong> <span class="${getStatusClass(data.status)}">${data.status}</span>`;

                // 更新进度条
                const progress = data.processed / data.total_files * 100;
                progressBar.style.width = `${progress}%`;

                // 更新进度详情
                let rangeInfo = '';
                if (data.start_index && data.end_index) {
                    rangeInfo = `<div class="detail-item">
                        <div class="label">文件范围</div>
                        <div class="value">${data.start_index} - ${data.end_index}</div>
                    </div>`;
                }

                progressDetails.innerHTML = `
                    ${rangeInfo}
                    <div class="detail-item">
                        <div class="label">总文件数</div>
                        <div class="value">${data.total_files}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">已处理</div>
                        <div class="value">${data.processed}</div>
                    </div>
                    <div class="detail-item success">
                        <div class="label">成功</div>
                        <div class="value">${data.success}</div>
                    </div>
                    <div class="detail-item error">
                        <div class="label">失败</div>
                        <div class="value">${data.failed}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">当前文件</div>
                        <div class="value file-path">${data.current_file || '-'}</div>
                    </div>
                `;

                // 显示取消按钮
                if (data.status === 'running') {
                    cancelBtn.style.display = 'flex';
                } else {
                    cancelBtn.style.display = 'none';

                    // 如果任务完成，显示浮动提示
                    if (data.status === 'completed') {
                        showToast(`任务完成！成功: ${data.success}, 失败: ${data.failed}`, 'success');
                    } else if (data.status === 'cancelled') {
                        showToast('任务已取消', 'warning');
                    } else if (data.status === 'failed') {
                        showToast('任务失败', 'error');
                    }

                    // 刷新历史记录
                    refreshHistory();
                }
            }
        });
}

// 刷新历史记录
function refreshHistory() {
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#historyTable tbody');
            tbody.innerHTML = '';

            if (data.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="empty-row">暂无历史记录</td>
                    </tr>
                `;
                return;
            }

            data.forEach((task, index) => {
                // 确定状态徽章
                let statusBadge = '';
                if (task.status === 'completed') {
                    statusBadge = '<span class="status-badge success">已完成</span>';
                } else if (task.status === 'cancelled') {
                    statusBadge = '<span class="status-badge warning">已取消</span>';
                } else if (task.status === 'failed') {
                    statusBadge = '<span class="status-badge error">失败</span>';
                } else {
                    statusBadge = `<span class="status-badge">${task.status}</span>`;
                }

                // 添加查看失败按钮
                let viewFailBtn = '';
                if (task.failed > 0) {
                    viewFailBtn = `<button class="btn-view-fail" data-id="${index}">查看失败</button>`;
                }

                tbody.innerHTML += `
                    <tr class="history-row">
                        <td>${task.start_time}</td>
                        <td>${task.end_time || '-'}</td>
                        <td>${task.start_index || 1} - ${task.end_index || task.total_files}</td>
                        <td>${task.total_files}</td>
                        <td class="success">${task.success}</td>
                        <td class="error">${task.failed}</td>
                        <td>${statusBadge}</td>
                        <td>${viewFailBtn}</td>
                    </tr>
                    <tr class="fail-details" id="fail-details-${index}" style="display: none;">
                        <td colspan="8">
                            <div class="fail-list">
                                <h3>失败文件列表 (${task.failed})</h3>
                                <ul>
                                    ${task.fail_list.map(fail => `
                                        <li>
                                            <div class="fail-path">${fail.path}</div>
                                            <div class="fail-error">错误: ${fail.error}</div>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        </td>
                    </tr>
                `;
            });

            // 添加查看失败按钮事件
            document.querySelectorAll('.btn-view-fail').forEach(btn => {
                btn.addEventListener('click', function() {
                    const id = this.getAttribute('data-id');
                    const details = document.getElementById(`fail-details-${id}`);
                    if (details.style.display === 'none') {
                        details.style.display = 'table-row';
                        this.textContent = '隐藏失败';
                    } else {
                        details.style.display = 'none';
                        this.textContent = '查看失败';
                    }
                });
            });
        });
}

// 根据状态获取CSS类
function getStatusClass(status) {
    switch(status) {
        case 'running': return 'warning';
        case 'completed': return 'success';
        case 'cancelled':
        case 'failed': return 'error';
        default: return '';
    }
}

// 显示浮动提示
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast show';

    // 设置类型
    if (type === 'error') {
        toast.classList.add('error');
    } else if (type === 'warning') {
        toast.classList.add('warning');
    } else {
        toast.classList.remove('error', 'warning');
    }

    // 2秒后隐藏
    setTimeout(() => {
        toast.className = 'toast';
    }, 2000);
}

// 取消任务
document.getElementById('cancelBtn').addEventListener('click', function() {
    fetch('/cancel_task', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                showToast(data.message, 'warning');
                // 刷新历史记录
                refreshHistory();
            } else if (data.error) {
                showToast(data.error, 'error');
            }
        });
});

// 清空历史记录
document.getElementById('clearHistory').addEventListener('click', function() {
    if (confirm('确定要清空所有历史记录吗？')) {
        fetch('/clear_history', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.message) {
                    showToast(data.message);
                    // 刷新历史记录
                    refreshHistory();
                }
            });
    }
});

// 文件上传区域交互 - 修复版本
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('json_file');
const fileName = document.getElementById('fileName');

// 修复：直接使用label关联，无需JS触发
fileInput.addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        fileName.textContent = e.target.files[0].name;
        fileName.style.display = 'block';
    }
});

// 拖放功能
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileName.textContent = e.dataTransfer.files[0].name;
        fileName.style.display = 'block';

        // 自动触发change事件
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
});

// 表单提交处理
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();

    if (!fileInput.files.length) {
        showToast('请选择JSON文件', 'error');
        return;
    }

    const formData = new FormData(this);

    // 显示上传中提示
    showToast('任务启动中...');

    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showToast('错误: ' + data.error, 'error');
        } else {
            showToast(data.message);
            // 开始轮询任务状态
            clearInterval(statusInterval);
            statusInterval = setInterval(updateTaskStatus, 2000);
            updateTaskStatus();
        }
    })
    .catch(error => {
        showToast('请求失败: ' + error, 'error');
    });
});

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始更新任务状态
    updateTaskStatus();

    // 设置定时器每2秒更新一次状态
    statusInterval = setInterval(updateTaskStatus, 2000);

    // 初始化文件上传区域
    fileName.style.display = 'none';
    document.getElementById('cancelBtn').style.display = 'none';

    // 初始化历史记录
    refreshHistory();
});