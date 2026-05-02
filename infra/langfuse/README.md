# Langfuse 自托管（neoagent SDK 配套）

neoagent SDK 内置 OTEL 集成，这套 docker-compose 提供开箱即用的 Langfuse v3 本地后端，
任何使用 neoagent 的项目（trip-os 等）只需配 `.env` 即可使用。

## 1. 启动

```bash
cd infra/langfuse
cp .env.example .env   # 生产环境请修改 .env 内所有 secret
docker compose up -d
```

## 2. 等服务起来

Langfuse Web 需要约 1-2 分钟完成数据库迁移，可用以下命令查看 healthcheck 状态：

```bash
docker compose ps
```

所有服务均为 `healthy` 后，访问 http://localhost:3000

## 3. 创建账号 + 项目 + API Key

1. 浏览器访问 http://localhost:3000，点击 **Sign up** 创建账号
   （自托管模式无需邮件验证，直接可用）
2. 创建 **Organization** → 创建 **Project**
3. 进入 Project → **Settings** → **API Keys** → **Create new API key**
4. 复制 **Public Key**（`pk-lf-...`）和 **Secret Key**（`sk-lf-...`）

## 4. 配到 neoagent 项目的 .env

```bash
# 把 pk:sk 用 base64 编码（OTLP Basic auth header 格式）
echo -n "pk-lf-xxx:sk-lf-xxx" | base64
# 输出示例：cGstbGYteHh4OnNrLWxmLXh4eA==

# 在你的项目（如 trip-os）的 .env 加入：
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic cGstbGYteHh4OnNrLWxmLXh4eA==
```

## 5. 验证

跑 agent 后，回到 Langfuse Web UI 的 **Tracing** tab，确认出现 trace。

## 6. 停止 / 清理

```bash
docker compose down          # 停服务，保留数据
docker compose down -v       # 停服务 + 删 volumes（彻底清空）
```

## 7. 端口表

| 服务 | host:container | 用途 |
|------|----------------|------|
| Langfuse Web | 3000:3000 | UI + OTEL endpoint |
| Postgres | 5433:5432 | Langfuse 元数据 |
| ClickHouse HTTP | 8123:8123 | trace 查询（一般不直接用） |
| MinIO API | 9090:9000 | blob 存储 |
| MinIO Console | 9091:9001 | MinIO 管理 UI（可选） |
| Redis | 6380:6379 | 缓存（开发环境调试） |

端口设计刻意避开 trip-os 主 compose 占用的 5432 / 6379 / 8000，两套 compose 可以同时运行。

## 生产注意事项

- `.env` 内所有 `dev` 字样的 secret 在生产环境**必须替换**
- `LANGFUSE_ENCRYPTION_KEY` 必须是真实的 64 字符 hex：`openssl rand -hex 32`
- ClickHouse native port（9000）未暴露到 host，服务间通过 `langfuse_net` 内网通信
- 生产建议在 langfuse-web 前面加反向代理（nginx / caddy）并启用 TLS
