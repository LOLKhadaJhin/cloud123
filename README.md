更新：

```
2025-06-14
1、现在打开更慢了，因为官方api1秒内只能获取十次文件夹，所以做了限制。
2、但你先别急，好消息是增加了缓存，第二次访问比以前更快，无须访问文件夹。也可以事先缓存好！
3、端口号改成了1234
```

使用方法：
1、下载这三个文件（app.py、Dockerfile、requirements.txt）放入nas，并获取在nas目录，比如：`/volume1/docker/container/cloud123`

2、ssh连接nas，构建镜像

```
# 获取root权限
sudo -i
# 输入密码
# 构建docker
cd /volume1/docker/container/cloud123
docker pull python:3.10-slim
docker build -t cloud123:1.0 .
```

3、构建容器,端口我用的1234

```
version: '3'

services:
  cloud123:
    image: cloud123:1.0
    container_name: cloud123
    restart: always
    ports:
      - "1234:1234"
    environment:
      - PAN123_CLIENT_ID=**********
      - PAN123_CLIENT_SECRET=******
```

4、验证成功。假设123云盘根目录下abc目录下有一个1.txt

```
# 查看令牌信息（调试）
http://IP:1234/token_info

# JSON格式获取下载链接  http://IP:1234/get_download_url?path=123云盘文件路径
http://IP:1234/get_download_url?path=/abc/1.txt

# 直接重定向下载 http://IP:1234/get/ + 123云盘文件路径
http://IP:1234/get/abc/1.txt
```

5、缓存目录

```
当文件夹删除重建时最好删除缓存

假设根目录下有文件夹abc需要缓存
http://IP:1234/cache/save_folder?path=/adc
删除文件夹abc缓存
http://IP:1234/cache/delete?path=/adc
删除全部缓存
http://IP:1234/cache/clear
```

6、FastEmby 配置

```
专业路径映射，和之前填法一样
/CloudNAS/123云盘   http://192.168.68.123:1234/get
```

7、嘿嘿如果可以，请作者喝一杯咖啡

![1594ce86c4b32d809db1a701c39db9ee](1.png)
