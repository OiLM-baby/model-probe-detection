# Model Probe Detection

从 API Test UI 单独拆出的模型探针 + 模型检测工具。

## 特点

- 不依赖 MySQL / SQLite 业务数据库
- Provider 配置、探针历史、检测历史保存为本地 JSON 文件
- 探针支持聊天、OpenAI 图片文生图、图生图、Responses 图片和 Banna 图片模型
- 检测复用内置 TokenStar runtime，报告仍按 TokenStar 模板生成

## 启动

```bash
pip install -r requirements.txt

cd frontend
npm install
cd ..

python run.py
```

默认地址：

```text
http://127.0.0.1:8090
```

开发前端：

```bash
cd frontend
npm run dev
```

Vite 会把 `/api` 代理到 `http://127.0.0.1:8090`。

## 数据文件

运行后会自动生成：

```text
data/probe_configs.json
data/probe_runs.json
data/detection_runs.json
```

这些文件就是本工具的轻量存储，不需要数据库服务。

## 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MODEL_TOOL_HOST` | 后端监听地址 | `0.0.0.0` |
| `MODEL_TOOL_PORT` | 后端监听端口 | `8090` |
| `MODEL_TOOL_SKIP_FRONTEND_BUILD` | 启动时跳过前端构建 | 空 |

## 页面

- `/probe`：保存 Provider 配置，按探针类型执行聊天首 Token、图片生成/编辑、Banna 图片模型探测。
- `/detection`：选择 Provider 和模型，执行 TokenStar 检测套件，查看历史并下载 Markdown 报告。
