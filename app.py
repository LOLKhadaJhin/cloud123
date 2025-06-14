import os
import requests
import time
import threading
import sqlite3
import json
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify, redirect
from collections import deque
import re

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库文件路径
DB_FILE = "pan_cache.db"


# 初始化SQLite数据库
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 创建目录缓存表
    c.execute('''CREATE TABLE IF NOT EXISTS directories (
                 id INTEGER PRIMARY KEY,
                 parent_id INTEGER,
                 file_id INTEGER UNIQUE,
                 last_updated REAL,
                 data TEXT)''')

    # 创建文件缓存表
    c.execute('''CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY,
                 file_id INTEGER UNIQUE,
                 name TEXT,
                 parent_id INTEGER,
                 size INTEGER,
                 type INTEGER,
                 etag TEXT,
                 download_url TEXT,
                 expires REAL,
                 last_updated REAL)''')

    # 创建路径映射表
    c.execute('''CREATE TABLE IF NOT EXISTS path_mapping (
                 id INTEGER PRIMARY KEY,
                 path TEXT UNIQUE,
                 file_id INTEGER,
                 is_directory INTEGER)''')

    # 创建索引
    c.execute("CREATE INDEX IF NOT EXISTS idx_path ON path_mapping (path)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_parent ON directories (parent_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON files (file_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mapping_file_id ON path_mapping (file_id)")

    conn.commit()
    conn.close()


# 初始化数据库
init_db()

# 全局变量存储访问令牌及其过期时间
access_token = None
token_expiry = 0
token_lock = threading.Lock()  # 用于线程安全的锁


# 速率限制器（令牌桶算法）
class RateLimiter:
    def __init__(self, max_tokens, refill_rate):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # 每秒补充的令牌数
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens=1):
        with self.lock:
            # 补充令牌
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now

            # 检查是否有足够的令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# 创建文件列表API的限流器（10次/秒）
file_list_limiter = RateLimiter(max_tokens=10, refill_rate=10)


def wait_for_rate_limit():
    """等待获取API调用令牌"""
    while not file_list_limiter.acquire():
        time.sleep(0.1)  # 等待100毫秒后重试


def get_access_token():
    """获取或刷新访问令牌"""
    global access_token, token_expiry

    # 如果令牌未过期，直接返回现有令牌
    if access_token and time.time() < token_expiry - 60:  # 提前60秒刷新
        return access_token

    # 从环境变量获取客户端凭证
    client_id = os.environ.get("PAN123_CLIENT_ID")
    client_secret = os.environ.get("PAN123_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise Exception("未配置PAN123_CLIENT_ID或PAN123_CLIENT_SECRET")

    with token_lock:  # 加锁确保线程安全
        # 再次检查，避免多个线程同时刷新令牌
        if access_token and time.time() < token_expiry - 60:
            return access_token

        # 调用API获取新令牌
        url = 'https://open-api.123pan.com/api/v1/access_token'
        headers = {
            'Platform': 'open_platform',
            'Content-Type': 'application/json'
        }
        data = {
            "clientID": client_id,
            "clientSecret": client_secret
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"获取access_token失败: HTTP {response.status_code}")

        result = response.json()
        if result['code'] != 0:
            raise Exception(f"获取access_token错误: {result['message']}")

        # 更新令牌和过期时间
        access_token = result['data']['accessToken']
        expired_str = result['data']['expiredAt']

        # 解析并转换过期时间
        try:
            # 解析ISO格式的时间字符串
            dt = datetime.fromisoformat(expired_str)

            # 如果时区信息缺失，添加UTC时区
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            # 转换为UNIX时间戳
            token_expiry = dt.timestamp()
            logger.info(f"获取新访问令牌，有效期至: {dt}")
        except ValueError:
            # 如果解析失败，使用默认有效期（1小时）
            token_expiry = time.time() + 3600
            logger.warning(f"解析过期时间失败: {expired_str}，使用默认1小时有效期")

        return access_token


def get_file_list_from_api(parent_file_id):
    """从API获取目录文件列表（原始API调用）"""
    token = get_access_token()
    items = []
    last_file_id = None

    while True:
        # 遵守API速率限制
        wait_for_rate_limit()

        # 构建请求URL
        url = f"https://open-api.123pan.com/api/v2/file/list?parentFileId={parent_file_id}&limit=100"
        if last_file_id is not None:
            url += f"&lastFileId={last_file_id}"

        headers = {
            'Content-Type': 'application/json',
            'Platform': 'open_platform',
            'Authorization': f'Bearer {token}'
        }

        # 发送API请求
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API请求失败: HTTP {response.status_code}")

        data = response.json()
        if data['code'] != 0:
            raise Exception(f"API错误: {data['message']}")

        # 添加当前页数据
        items.extend(data['data']['fileList'])

        # 检查是否还有下一页
        last_file_id = data['data']['lastFileId']
        if last_file_id == -1:  # 无更多数据
            break

    return items


def cache_directory_contents(parent_id, items):
    """缓存目录内容到SQLite"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # 缓存目录内容
        c.execute("""
            INSERT OR REPLACE INTO directories (parent_id, file_id, last_updated, data)
            VALUES (?, ?, ?, ?)
        """, (parent_id, parent_id, time.time(), json.dumps(items)))

        # 缓存每个文件和子目录
        for item in items:
            # 缓存到文件表
            c.execute("""
                INSERT OR REPLACE INTO files (
                    file_id, name, parent_id, size, type, etag, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                item['fileId'],
                item['filename'],
                parent_id,
                item['size'],
                item['type'],
                item['etag'],
                time.time()
            ))

            # 缓存路径映射
            is_dir = 1 if item['type'] == 1 else 0
            path = get_path_for_file_id(item['fileId'], parent_id)

            if path:
                c.execute("""
                    INSERT OR REPLACE INTO path_mapping (path, file_id, is_directory)
                    VALUES (?, ?, ?)
                """, (path.lower(), item['fileId'], is_dir))

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {str(e)}")
    finally:
        conn.close()


def get_cached_directory(parent_id):
    """从SQLite获取缓存的目录内容"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT data, last_updated FROM directories WHERE parent_id = ?", (parent_id,))
    row = c.fetchone()
    conn.close()

    if row:
        data, last_updated = row
        # 检查缓存是否过期（5分钟）
        if time.time() - last_updated < 300:
            return json.loads(data)
    return None


def get_cached_file_info(file_id):
    """从SQLite获取缓存的文件信息"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT name, parent_id, size, type, etag, last_updated 
        FROM files WHERE file_id = ?
    """, (file_id,))
    row = c.fetchone()
    conn.close()

    if row:
        name, parent_id, size, ftype, etag, last_updated = row
        # 检查缓存是否过期（1小时）
        if time.time() - last_updated < 3600:
            return {
                'fileId': file_id,
                'filename': name,
                'parentFileId': parent_id,
                'size': size,
                'type': ftype,
                'etag': etag
            }
    return None


def get_path_for_file_id(file_id, parent_id):
    """构建文件/目录的完整路径"""
    if parent_id == 0:  # 根目录下的项目
        return f"/{file_id}"

    # 获取父目录路径
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT path FROM path_mapping WHERE file_id = ?", (parent_id,))
    row = c.fetchone()
    conn.close()

    if row:
        parent_path = row[0]
        # 获取文件名
        file_info = get_cached_file_info(file_id)
        if file_info:
            # 移除可能存在的重复斜杠
            return f"{parent_path.rstrip('/')}/{file_info['filename']}"
    return None


def get_file_list(parent_file_id, target_name=None, target_type=None, force_refresh=False):
    """
    获取目录文件列表（带缓存优化）
    可选参数：
      target_name: 要查找的文件/目录名
      target_type: 要查找的类型 (0=文件, 1=目录)
      force_refresh: 强制刷新缓存
    """
    # 尝试从缓存获取
    if not force_refresh:
        cached_data = get_cached_directory(parent_file_id)
        if cached_data:
            logger.info(f"从缓存获取目录: {parent_file_id} ({len(cached_data)}项)")

            # 如果有目标查找条件，检查缓存
            if target_name:
                for item in cached_data:
                    name_match = f"{item['filename']}" == f"{target_name}"
                    type_match = (target_type is None) or (item['type'] == target_type)
                    if name_match and type_match:
                        return item
            return cached_data

    # 缓存未命中或强制刷新，调用API
    logger.info(f"调用API获取目录: {parent_file_id}")
    items = get_file_list_from_api(parent_file_id)

    # 存储到缓存
    cache_directory_contents(parent_file_id, items)

    # 如果有目标查找条件，检查结果
    if target_name:
        for item in items:
            name_match = item['filename'] == target_name
            type_match = (target_type is None) or (item['type'] == target_type)
            if name_match and type_match:
                return item
        return None

    return items


def find_file_id_by_path(path, force_refresh=False, folder_type=False):
    """根据路径查找文件ID（带缓存优化）"""
    # 清理路径格式
    path = re.sub(r'/{2,}', '/', path.strip().rstrip('/'))

    # 尝试从路径缓存获取
    if not force_refresh:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT file_id FROM path_mapping WHERE path = ?", (path.lower(),))
        row = c.fetchone()
        conn.close()

        if row:
            file_id = row[0]
            logger.info(f"从缓存获取路径映射: {path} -> {file_id}")
            return file_id

    parts = [p for p in path.strip('/').split('/') if p]  # 拆分路径并移除空部分
    if not parts:
        raise ValueError("路径无效")

    current_parent_id = 0  # 从根目录开始
    current_path = ""

    for idx, part in enumerate(parts):
        is_last_component = (idx == len(parts) - 1)
        target_type = 0 if is_last_component and not folder_type else 1

        # 构建当前路径
        current_path = f"{current_path}/{part}" if current_path else f"/{part}"

        # 查找当前项
        found = get_file_list(
            current_parent_id,
            target_name=part,
            target_type=target_type,
            force_refresh=force_refresh
        )

        if not found:
            raise FileNotFoundError(f"路径不存在: {current_path}")

        # 更新当前父ID
        current_parent_id = found['fileId']

        # 缓存当前路径映射
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO path_mapping (path, file_id, is_directory)
            VALUES (?, ?, ?)
        """, (current_path.lower(), current_parent_id, 1 if target_type == 1 else 0))
        conn.commit()
        conn.close()

    return current_parent_id


def save_folder_to_db(path, max_depth=8):
    """保存指定文件夹到SQLite数据库（永久存储）"""
    logger.info(f"永久保存文件夹: {path} (深度: {max_depth})")

    # 获取目录ID
    dir_id = find_file_id_by_path(path, folder_type=True)

    # 使用BFS遍历目录结构
    queue = deque([(dir_id, 0, path)])
    saved_count = 0

    while queue:
        parent_id, depth, current_path = queue.popleft()

        # 获取目录内容（强制刷新）
        items = get_file_list(parent_id, force_refresh=True)

        # 遍历目录项
        for item in items:
            item_path = f"{current_path.rstrip('/')}/{item['filename']}"

            # 如果是目录且未达到深度限制，加入队列
            if item['type'] == 1 and depth < max_depth:
                queue.append((item['fileId'], depth + 1, item_path))

        saved_count += len(items)

    return saved_count


def delete_cached_path(path):
    """删除指定路径的缓存"""
    # 清理路径格式
    path = re.sub(r'/{2,}', '/', path.strip().rstrip('/')).lower()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # 1. 删除路径映射
        c.execute("DELETE FROM path_mapping WHERE path = ? OR path LIKE ?",
                  (path, f"{path}/%"))

        # 2. 获取关联的文件ID
        c.execute("SELECT file_id FROM path_mapping WHERE path = ?", (path,))
        file_ids = [row[0] for row in c.fetchall()]

        # 3. 删除目录缓存
        if file_ids:
            placeholders = ','.join('?' * len(file_ids))
            c.execute(f"DELETE FROM directories WHERE parent_id IN ({placeholders})", file_ids)

        # 4. 删除文件缓存
        if file_ids:
            c.execute(f"DELETE FROM files WHERE file_id IN ({placeholders})", file_ids)

        conn.commit()
        deleted_count = c.rowcount
        logger.info(f"删除缓存: {path} ({deleted_count}项)")

        return deleted_count
    except sqlite3.Error as e:
        logger.error(f"删除缓存失败: {str(e)}")
        return 0
    finally:
        conn.close()


def get_download_url(file_id):
    """根据文件ID获取下载链接"""
    # 检查缓存
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT download_url, expires FROM files WHERE file_id = ?", (file_id,))
    row = c.fetchone()

    if row and row[0] and row[1] and time.time() < row[1]:
        conn.close()
        return row[0]

    conn.close()

    # 缓存未命中或过期，调用API
    token = get_access_token()
    url = f"https://open-api.123pan.com/api/v1/file/download_info?fileId={file_id}"
    headers = {
        'Content-Type': 'application/json',
        'Platform': 'open_platform',
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("下载API请求失败")

    data = response.json()
    if data['code'] != 0:
        raise Exception(f"下载API错误: {data['message']}")

    download_url = data['data']['downloadUrl']

    # 缓存下载链接（有效1小时）
    expires = time.time() + 3600
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        UPDATE files SET download_url = ?, expires = ?, last_updated = ?
        WHERE file_id = ?
    """, (download_url, expires, time.time(), file_id))
    conn.commit()
    conn.close()

    return download_url


def get_download_url_by_path(path):
    """根据路径获取下载链接（带缓存优化）"""
    try:
        file_id = find_file_id_by_path(path)
        return get_download_url(file_id)
    except Exception as e:
        logger.error(f"获取下载链接失败: {str(e)}")
        raise


@app.route('/get_download_url', methods=['GET'])
def handle_download_request():
    """API端点：返回JSON格式的下载链接"""
    path = request.args.get('path')
    if not path:
        return jsonify({'error': '缺少 path 参数'}), 400

    try:
        download_url = get_download_url_by_path(path)
        return jsonify({
            'path': path,
            'download_url': download_url
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get/<path:file_path>')
def handle_direct_download(file_path):
    """直接下载端点：302重定向到实际下载链接"""
    try:
        # 添加前导斜杠确保路径格式正确
        full_path = f"/{file_path}"
        download_url = get_download_url_by_path(full_path)
        return redirect(download_url, code=302)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/save_folder', methods=['GET'])
def save_folder():
    """永久保存文件夹到数据库"""
    path = request.args.get('path', '/')
    max_depth = int(request.args.get('depth', 8))

    try:
        count = save_folder_to_db(path, max_depth)
        return jsonify({
            'status': 'success',
            'path': path,
            'depth': max_depth,
            'items_saved': count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/delete', methods=['GET'])
def delete_cache():
    """删除指定路径的缓存"""
    path = request.args.get('path')
    if not path:
        return jsonify({'error': '缺少 path 参数'}), 400

    try:
        count = delete_cached_path(path)
        return jsonify({
            'status': 'success',
            'path': path,
            'items_deleted': count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    """清除所有缓存"""
    try:
        # 删除所有缓存数据
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM directories")
        c.execute("DELETE FROM files")
        c.execute("DELETE FROM path_mapping")
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'message': '所有缓存已清除'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/stats')
def cache_stats():
    """获取缓存统计信息"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # 获取目录缓存统计
        c.execute("SELECT COUNT(*) FROM directories")
        dir_count = c.fetchone()[0]

        # 获取文件缓存统计
        c.execute("SELECT COUNT(*) FROM files")
        file_count = c.fetchone()[0]

        # 获取路径映射统计
        c.execute("SELECT COUNT(*) FROM path_mapping")
        path_count = c.fetchone()[0]

        conn.close()

        return jsonify({
            'directories': dir_count,
            'files': file_count,
            'path_mappings': path_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)
