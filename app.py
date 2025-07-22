import os
import requests
import time
import threading
import sqlite3
import json
import logging
from datetime import datetime, timezone
from flask import Flask, render_template,request, jsonify, redirect
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
                 is_directory INTEGER,
                 last_verified REAL)''')

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
task_lock = threading.Lock()
current_task = None
task_history = deque(maxlen=20)  # 保存最近20条历史记录
history_updated = False  # 跟踪历史记录是否更新
# 默认配置
DEFAULT_PARENT_ID = os.environ.get("PAN123_PARENT_ID", 0)
# Base62 字符集
BASE62_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def base62_to_hex(base62_str):
    """将 Base62 编码的字符串转换为十六进制"""
    # 验证输入是否为有效的 Base62 字符串
    if not all(char in BASE62_CHARS for char in base62_str):
        raise ValueError(f"Invalid Base62 string: {base62_str}")

    # 转换为大整数
    num = 0
    for char in base62_str:
        num = num * 62 + BASE62_CHARS.index(char)

    # 转换为十六进制并补齐32位
    hex_str = hex(num)[2:]  # 去掉 '0x' 前缀
    hex_str = hex_str.lower()

    # 补齐到32个字符
    if len(hex_str) < 32:
        hex_str = '0' * (32 - len(hex_str)) + hex_str

    return hex_str


def convert_etag_if_needed(file_info, uses_base62):
    """如果需要，将 Base62 ETag 转换为十六进制格式"""
    if not uses_base62:
        return file_info["etag"]

    try:
        # print(f"转换 ETag: {file_info['etag']} -> {base62_to_hex(file_info['etag'])}")
        return base62_to_hex(file_info["etag"])
    except Exception as e:
        print(f"ETag 转换失败: {e}")
        return file_info["etag"]  # 返回原始值，让API处理

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
            # 如果是404错误，可能是目录被删除
            if response.status_code == 404:
                logger.warning(f"目录不存在: {parent_file_id}")
                raise FileNotFoundError(f"目录不存在: {parent_file_id}")
            raise Exception(f"API请求失败: HTTP {response.status_code}")

        data = response.json()
        if data['code'] != 0:
            # 特定错误码处理：文件不存在
            if data['code'] in [10004, 404]:
                logger.warning(f"API返回目录不存在: {parent_file_id}")
                raise FileNotFoundError(f"目录不存在: {parent_file_id}")
            raise Exception(f"API错误: {data['message']}")

        # 添加当前页数据，过滤掉已删除的项目
        for item in data['data']['fileList']:
            if item.get('trashed', 0) == 0:  # 只处理未删除的项目
                items.append(item)

        # 检查是否还有下一页
        last_file_id = data['data']['lastFileId']
        if last_file_id == -1:  # 无更多数据
            break

    return items


def get_parent_directory_id(file_id):
    """获取父目录ID"""
    logger.info(f"获取父目录ID: {file_id}")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 尝试从文件表获取
    c.execute("SELECT parent_id FROM files WHERE file_id = ?", (file_id,))
    row = c.fetchone()

    if not row:
        # 尝试从目录表获取
        c.execute("SELECT parent_id FROM directories WHERE file_id = ?", (file_id,))
        row = c.fetchone()

    conn.close()
    return row[0] if row else None


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

        # 缓存每个文件和子目录（只缓存未删除的项目）
        for item in items:
            if item.get('trashed', 0) == 1:
                continue  # 跳过已删除项目

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
                    INSERT OR REPLACE INTO path_mapping (path, file_id, is_directory, last_verified)
                    VALUES (?, ?, ?, ?)
                """, (path.lower(), item['fileId'], is_dir, time.time()))

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {str(e)}")
    finally:
        conn.close()


def get_cached_directory(parent_id):
    """从SQLite获取缓存的目录内容（过滤已删除项目）"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT data, last_updated FROM directories WHERE parent_id = ?", (parent_id,))
    row = c.fetchone()
    conn.close()

    if row:
        data, last_updated = row
        # 检查缓存是否过期（5分钟）
        if time.time() - last_updated < 300:
            items = json.loads(data)
            # 过滤掉已删除的项目
            valid_items = [item for item in items if item.get('trashed', 0) == 0]
            logger.info(f"从缓存获取有效目录项: {parent_id} ({len(valid_items)}项)")
            return valid_items
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
        # 获取文件名
        file_info = get_cached_file_info(file_id)
        if file_info:
            return f"/{file_info['filename']}"
        return f"/{file_id}"  # 回退方案

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
    # 尝试从缓存获取（已过滤已删除项目）
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
            else:
                return cached_data

    # 缓存未命中或强制刷新，调用API
    logger.info(f"调用API获取目录: {parent_file_id}")

    try:
        items = get_file_list_from_api(parent_file_id)
    except FileNotFoundError:
        # 目录不存在，清除相关缓存
        logger.warning(f"目录不存在，清除缓存: {parent_file_id}")
        delete_cached_path_by_id(parent_file_id)
        # 重新尝试API调用
        items = get_file_list_from_api(parent_file_id)

    # 新增：如果目录内容为空，强制刷新父目录
    if not items and parent_file_id != 0:
        logger.warning(f"目录 {parent_file_id} 内容为空，强制刷新父目录")
        parent_dir_id = get_parent_directory_id(parent_file_id)
        if parent_dir_id:
            logger.info(f"找到父目录ID: {parent_dir_id}")
            # 强制刷新父目录
            get_file_list(parent_dir_id, force_refresh=True)
            # 重新获取当前目录内容
            items = get_file_list_from_api(parent_file_id)
            logger.info(f"刷新后目录内容: {len(items)}项")

    # 存储到缓存（自动过滤已删除项目）
    cache_directory_contents(parent_file_id, items)

    # 如果有目标查找条件，检查结果
    if target_name:
        for item in items:
            logger.warning(f"文件{item} ")
            # 确保只处理未删除的项目
            if item.get('trashed', 0) == 1:
                continue

            name_match = f"{item['filename']}" == f"{target_name}"
            type_match = (target_type is None) or (item['type'] == target_type)
            if name_match and type_match:
                return item
        return None

    return items


def find_file_id_by_path(path, force_refresh=False, folder_type=False):
    """根据路径查找文件ID（带缓存优化），添加全局重试机制"""
    # 清理路径格式
    path = re.sub(r'/{2,}', '/', path.strip().rstrip('/'))
    logger.info(f"查找路径: {path} (强制刷新: {force_refresh})")

    # 第一次尝试
    try:
        return _find_file_id_by_path(path, force_refresh, folder_type)
    except FileNotFoundError as e:
        logger.warning(f"首次查找失败: {path}, 错误: {str(e)}")
        # 删除整个路径的缓存
        delete_cached_path(path)

        # 第二次尝试 - 从根目录强制刷新
        logger.info("从根目录重新尝试查找...")
        try:
            return _find_file_id_by_path(path, True, folder_type)
        except Exception as e2:
            logger.error(f"全局重试失败: {path}, 错误: {str(e2)}")
            raise FileNotFoundError(f"路径不存在: {path}") from e2

def _find_file_id_by_path(path, force_refresh=False, folder_type=False):
    """内部实现 - 根据路径查找文件ID"""
    # 尝试从路径缓存获取
    if not force_refresh:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT file_id, last_verified FROM path_mapping WHERE path = ?", (path.lower(),))
        row = c.fetchone()
        conn.close()

        if row:
            file_id, last_verified = row
            # 检查是否在最近验证过（5分钟内）
            if time.time() - last_verified < 300:
                logger.info(f"从缓存获取路径映射: {path} -> {file_id}")
                return file_id
            else:
                logger.info(f"路径验证过期，重新验证: {path}")

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
        logger.info(f"处理路径组件: {current_path} (类型: {target_type})")

        # 查找当前项（最多重试2次）
        found = None
        for attempt in range(2):
            try:
                found = get_file_list(
                    current_parent_id,
                    target_name=part,
                    target_type=target_type,
                    force_refresh=(force_refresh or attempt > 0)  # 强制刷新或重试时强制刷新
                )
                if found:
                    # 检查找到的项目是否已被删除
                    if found.get('trashed', 0) == 1:
                        logger.warning(f"找到的项目已被删除: {part} (ID: {found['fileId']})")
                        found = None
                    else:
                        break
                else:
                    # 如果未找到且不是最后一次尝试，继续重试
                    if attempt < 1:
                        logger.warning(f"首次查找未找到，强制刷新父目录: {current_parent_id}")
                    else:
                        raise FileNotFoundError(f"路径不存在: {current_path}")
            except FileNotFoundError as e:
                if attempt == 0:
                    logger.warning(f"首次查找失败，强制刷新父目录: {current_parent_id}")
                else:
                    logger.error(f"路径查找失败: {current_path}")
                    raise e

        if not found:
            raise FileNotFoundError(f"路径不存在: {current_path}")

        # 更新当前父ID
        current_parent_id = found['fileId']
        logger.info(f"找到文件ID: {current_parent_id}")

        # 缓存当前路径映射
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        is_directory = 1 if (target_type == 1 or (is_last_component and folder_type)) else 0
        c.execute("""
            INSERT OR REPLACE INTO path_mapping (path, file_id, is_directory, last_verified)
            VALUES (?, ?, ?, ?)
        """, (current_path.lower(), current_parent_id, is_directory, time.time()))
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

        # 缓存目录内容
        cache_directory_contents(parent_id, items)

        # 遍历目录项
        for item in items:
            # 跳过已删除项目
            if item.get('trashed', 0) == 1:
                continue

            item_path = f"{current_path.rstrip('/')}/{item['filename']}"

            # 如果是目录且未达到深度限制，加入队列
            if item['type'] == 1 and depth < max_depth:
                queue.append((item['fileId'], depth + 1, item_path))

        saved_count += len([item for item in items if item.get('trashed', 0) == 0])

    return saved_count


def delete_cached_path(path):
    """删除指定路径的缓存"""
    # 清理路径格式
    path = re.sub(r'/{2,}', '/', path.strip().rstrip('/')).lower()
    logger.info(f"删除缓存路径: {path}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # 1. 获取关联的文件ID
        c.execute("SELECT file_id FROM path_mapping WHERE path = ?", (path,))
        file_ids = [row[0] for row in c.fetchall()]

        # 2. 删除路径映射
        c.execute("DELETE FROM path_mapping WHERE path = ? OR path LIKE ?",
                  (path, f"{path}/%"))

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


def delete_cached_path_by_id(file_id):
    """根据文件ID删除缓存"""
    logger.info(f"根据ID删除缓存: {file_id}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        # 1. 获取关联的路径
        c.execute("SELECT path FROM path_mapping WHERE file_id = ?", (file_id,))
        paths = [row[0] for row in c.fetchall()]

        # 2. 删除路径映射
        c.execute("DELETE FROM path_mapping WHERE file_id = ?", (file_id,))

        # 3. 删除目录缓存
        c.execute("DELETE FROM directories WHERE file_id = ?", (file_id,))

        # 4. 删除文件缓存
        c.execute("DELETE FROM files WHERE file_id = ?", (file_id,))

        conn.commit()
        deleted_count = c.rowcount

        # 5. 递归删除子路径
        for path in paths:
            if path:
                c.execute("DELETE FROM path_mapping WHERE path LIKE ?", (f"{path}/%",))
                conn.commit()

        logger.info(f"根据ID删除缓存完成: {file_id} ({deleted_count}项)")
        return deleted_count
    except sqlite3.Error as e:
        logger.error(f"根据ID删除缓存失败: {str(e)}")
        return 0
    finally:
        conn.close()


def get_download_url(file_id):
    """根据文件ID获取下载链接"""
    logger.info(f"获取下载链接: {file_id}")

    # 检查缓存
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT download_url, expires FROM files WHERE file_id = ?", (file_id,))
    row = c.fetchone()

    if row and row[0] and row[1] and time.time() < row[1]:
        conn.close()
        logger.info(f"从缓存获取下载链接: {file_id}")
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
        # 文件不存在，清除相关缓存
        if response.status_code == 404:
            logger.warning(f"文件不存在，清除缓存: {file_id}")
            delete_cached_path_by_id(file_id)
        raise Exception("下载API请求失败")

    data = response.json()
    if data['code'] != 0:
        # 文件不存在，清除相关缓存
        if data['code'] in [10004, 404]:
            logger.warning(f"API返回文件不存在，清除缓存: {file_id}")
            delete_cached_path_by_id(file_id)
        raise Exception(f"下载API错误: {data['message']}")

    download_url = data['data']['downloadUrl']
    logger.info(f"获取到下载链接: {download_url[:50]}...")

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
        logger.info(f"获取路径下载链接: {path}")
        file_id = find_file_id_by_path(path)
        return get_download_url(file_id)
    except Exception as e:
        logger.error(f"获取下载链接失败: {str(e)}")
        # 添加详细错误日志
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
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

@app.route('/token_info')
def token_info():
    """查看当前令牌信息（调试用）"""
    try:
        token = get_access_token()
        return jsonify({
            'access_token': token[:20] + '...',  # 部分显示
            'expiry_time': datetime.fromtimestamp(token_expiry).isoformat(),
            'seconds_remaining': int(token_expiry - time.time())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 秒传功能
def sec_upload(parent_id, filename, etag, size, contain_dir=True):
    """执行秒传操作 - 添加详细日志"""
    token = get_access_token()
    url = 'https://open-api.123pan.com/upload/v2/file/create'
    headers = {
        'Content-Type': 'application/json',
        'Platform': 'open_platform',
        'Authorization': f'Bearer {token}'
    }
    data = {
        "parentFileID": parent_id,
        "filename": filename,
        "etag": etag,
        "size": size,
        "duplicate": 2,
        "containDir": contain_dir
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result
    except Exception as e:
        return {"code": -1, "message": str(e)}


def process_sec_upload_task(task_data):
    """处理秒传任务"""
    global current_task, history_updated

    try:
        # 解析文件范围
        start_index = int(task_data.get("start_index", 1)) - 1  # 转换为0-based索引
        end_index = int(task_data.get("end_index", 0))

        # 如果end_index为0，表示处理到最后一个文件
        if end_index == 0 or end_index > len(task_data["files"]):
            end_index = len(task_data["files"])

        # 确保索引在有效范围内
        start_index = max(0, start_index)
        end_index = min(len(task_data["files"]), end_index)

        # 获取要处理的文件子集
        files_to_process = task_data["files"][start_index:end_index]

        with task_lock:
            current_task = {
                "status": "running",
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": len(files_to_process),
                "processed": 0,
                "success": 0,
                "failed": 0,
                "fail_list": [],
                "current_file": "",
                "parent_id": task_data["parent_id"],
                "common_path": task_data["common_path"],
                "uses_base62": task_data.get("uses_base62", False),
                "cancelled": False,
                "start_index": start_index + 1,  # 显示给用户的1-based索引
                "end_index": end_index
            }

        # 处理每个文件
        for idx, file_info in enumerate(files_to_process):
            # 检查是否取消
            with task_lock:
                if current_task.get("cancelled", False):
                    current_task["status"] = "cancelled"
                    break

            with task_lock:
                current_task["current_file"] = file_info["path"]
                current_task["processed"] = idx + 1

            # 构建完整文件路径
            full_path = os.path.join(task_data["common_path"], file_info["path"]).replace("\\", "/")

            # 转换 ETag（如果需要）
            etag = convert_etag_if_needed(
                file_info,
                task_data.get("uses_base62", False)
            )

            # 执行秒传
            result = sec_upload(
                parent_id=task_data["parent_id"],
                filename=full_path,
                etag=etag,
                size=int(file_info["size"]),
                contain_dir=True
            )

            # 处理结果
            if result.get("code") == 0 and result.get("data", {}).get("reuse"):
                with task_lock:
                    current_task["success"] += 1
            else:
                error_msg = result.get("message", "未知错误")
                with task_lock:
                    current_task["failed"] += 1
                    current_task["fail_list"].append({
                        "path": full_path,
                        "error": error_msg
                    })

            # 控制请求频率（每秒最多5次）
            time.sleep(0.25)

        # 任务完成
        with task_lock:
            if not current_task.get("cancelled", False):
                current_task["status"] = "completed"
            current_task["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 添加到历史记录
            task_history.append({
                "start_time": current_task["start_time"],
                "end_time": current_task["end_time"],
                "total_files": current_task["total_files"],
                "success": current_task["success"],
                "failed": current_task["failed"],
                "fail_list": current_task["fail_list"],
                "status": current_task["status"],
                "uses_base62": current_task["uses_base62"],
                "start_index": current_task["start_index"],
                "end_index": current_task["end_index"]
            })
            history_updated = True

    except Exception as e:
        with task_lock:
            if current_task:
                current_task["status"] = "failed"
                current_task["error"] = str(e)
                current_task["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 添加到历史记录
                task_history.append({
                    "start_time": current_task["start_time"],
                    "end_time": current_task["end_time"],
                    "total_files": current_task["total_files"],
                    "success": current_task["success"],
                    "failed": current_task["failed"],
                    "fail_list": current_task["fail_list"],
                    "status": "failed",
                    "uses_base62": current_task["uses_base62"],
                    "start_index": current_task["start_index"],
                    "end_index": current_task["end_index"]
                })
                history_updated = True

    finally:
        # 重置当前任务
        with task_lock:
            current_task = None


@app.route('/', methods=['GET', 'POST'])
def index():
    global current_task, history_updated

    if request.method == 'POST':
        # 检查是否有任务正在运行
        with task_lock:
            if current_task:
                return jsonify({
                    "error": "当前已有任务运行中，请等待完成后再提交新任务"
                }), 400

        # 获取表单数据
        parent_id = request.form.get('parent_id', DEFAULT_PARENT_ID)
        start_index = request.form.get('start_index', '1')
        end_index = request.form.get('end_index', '0')
        file = request.files.get('json_file')

        if not file:
            return jsonify({"error": "请选择JSON文件"}), 400

        try:
            # 读取并解析JSON文件
            json_data = json.load(file.stream)

            # 验证JSON结构
            required_fields = ["files", "commonPath"]
            for field in required_fields:
                if field not in json_data:
                    return jsonify({"error": f"无效的JSON格式: 缺少 {field} 字段"}), 400

            # 检查是否需要转换ETag
            uses_base62 = json_data.get("usesBase62EtagsInExport", False)
            print(f"检测到 usesBase62EtagsInExport: {uses_base62}")

            # 启动后台任务
            task_data = {
                "parent_id": parent_id,
                "common_path": json_data["commonPath"],
                "files": json_data["files"],
                "uses_base62": uses_base62,
                "start_index": start_index,
                "end_index": end_index
            }

            threading.Thread(
                target=process_sec_upload_task,
                args=(task_data,),
                daemon=True
            ).start()

            return jsonify({
                "message": "秒传任务已启动",
                "total_files": len(json_data["files"]),
                "start_index": start_index,
                "end_index": end_index,
                "uses_base62": uses_base62
            })

        except json.JSONDecodeError:
            return jsonify({"error": "无效的JSON格式"}), 400
        except Exception as e:
            return jsonify({"error": f"处理错误: {str(e)}"}), 400

    # GET请求显示页面
    # 检查历史记录是否更新
    refresh_history = False
    with task_lock:
        if history_updated:
            refresh_history = True
            history_updated = False

    return render_template(
        'index.html',
        default_parent_id=DEFAULT_PARENT_ID,
        history=list(task_history),
        refresh_history=refresh_history
    )


@app.route('/task_status')
def task_status():
    """获取当前任务状态"""
    with task_lock:
        if current_task:
            return jsonify(current_task)
        return jsonify({"status": "idle"})


@app.route('/cancel_task', methods=['POST'])
def cancel_task():
    """取消当前任务"""
    with task_lock:
        if current_task and current_task["status"] == "running":
            current_task["cancelled"] = True
            current_task["status"] = "cancelled"
        #     # 添加到历史记录
            task_history.append({
                "start_time": current_task["start_time"],
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": current_task["total_files"],
                "success": current_task["success"],
                "failed": current_task["failed"],
                "fail_list": current_task["fail_list"],
                "status": "cancelled",
                "uses_base62": current_task["uses_base62"],
                "start_index": current_task["start_index"],
                "end_index": current_task["end_index"]
            })
        #     # 重置当前任务
        #     current_task = None
            return jsonify({"message": "任务已取消"})
        return jsonify({"error": "没有正在运行的任务"}), 400


@app.route('/history')
def get_history():
    """获取历史记录"""
    with task_lock:
        return jsonify(list(task_history))


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """清空历史记录"""
    global task_history, history_updated
    with task_lock:
        task_history = deque(maxlen=20)
        history_updated = True
    return jsonify({"message": "历史记录已清空"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)
