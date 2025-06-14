# 使用官方 Python 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录内容到容器中的 /app 目录
COPY . /app

# 使用阿里云镜像加速安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用运行的端口
EXPOSE 1234

# 启动命令
CMD ["python", "app.py"]