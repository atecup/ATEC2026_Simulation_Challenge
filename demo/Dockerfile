# 基础镜像（选手可自由选择）
FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/pytorch:2.7.1.8-cuda12.8.1-py312-alinux3.2104

# 设置时区（不要改）
ENV TZ=Asia/Shanghai

# 基础设置（不要改）
RUN set -x && \
    groupadd -g 1000 admin && \
    useradd -u 1000 -g admin -m admin && \
    mkdir -p /home/admin/appspace/atec2026/robot/solution/ && \
    chown -R admin:admin /home/admin/appspace

# 将镜像自带的 python ac2 虚拟环境所有者改为 admin 用户（选手根据实际情况调整）
RUN chown -R admin:admin /opt/ac2

# 设置 pip 源（选手根据实际情况调整）
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn

# 使用 admin 用户运行（不要改）
USER admin

# 工作目录（不要改）
WORKDIR /home/admin/appspace/atec2026/robot

# 选手的文件都要放到 solution 目录下（不要改）

# 通用文件：run.sh，requirements.txt，server.py
COPY --chown=admin:admin run.sh ./solution/
COPY --chown=admin:admin requirements.txt ./solution/
COPY --chown=admin:admin server.py ./solution/

# demo1: 全 O demo
COPY --chown=admin:admin solution_zero.py ./solution/solution.py

# demo2: RL demo
# COPY --chown=admin:admin solution_rl.py ./solution/solution.py
# COPY --chown=admin:admin policy.pt ./solution/

# demo3: ACT demo
# COPY --chown=admin:admin solution_act.py ./solution/solution.py
# COPY --chown=admin:admin act ./solution/act
# COPY --chown=admin:admin policy_act.pt ./solution/policy_act.pt

# demo4: custom action
# COPY --chown=admin:admin solution_custom_action.py ./solution/solution.py

# 赋予 run.sh 可执行权限（不要改）
RUN set -x && \
    chmod +x ./solution/run.sh

# 在工作目录下创建虚拟环境软链接，复用镜像自带的 python ac2 虚拟环境（选手根据实际情况调整）
RUN set -x && \
    ln -s /opt/ac2 venv

# python 虚拟环境设置，并安装HTTP服务所需依赖（不要改）
RUN set -x && \
    source ./venv/bin/activate && \
    pip install --no-cache-dir fastapi==0.136.0 uvicorn==0.45.0 python-multipart==0.0.26

# 依赖安装（选手根据实际情况调整）
RUN set -x && \
    source ./venv/bin/activate && \
    if [ -s "solution/requirements.txt" ]; then \
        pip install --no-cache-dir -r solution/requirements.txt; \
    fi

# 选手镜像启动命令（不要改）
CMD ["./solution/run.sh"]
