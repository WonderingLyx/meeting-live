# ASR 插件接口

项目现在支持内置 ASR 引擎和外部本地 ASR 插件。新增模型时优先用插件接口验证，不需要改 `engine/asr/factory.py`。

## 配置文件

复制示例文件：

```powershell
copy config\asr_plugins.example.json config\asr_plugins.json
```

也可以用环境变量指定其他位置：

```env
ASR_PLUGINS_FILE=D:\models\matrix-asr-plugins.json
```

每个插件会变成一个 ASR 引擎：

```text
plugin:<id>
```

插件 ID 会规范化成小写下划线形式，例如 `faster-whisper-large-v3` 会变成 `plugin:faster_whisper_large_v3`。

配置项 `"enabled": false` 可以把模板留在 JSON 里，但不显示到前端引擎列表。

## 命令契约

当前支持的适配器是 `command`。项目会把待识别音频写成临时 16 kHz mono WAV，然后执行你的本地命令。

`command` 支持这些占位符：

- `{audio}`: 临时 WAV 路径
- `{device}`: 当前 ASR 设备，例如 `cpu` 或 `cuda`
- `{sample_rate}`: 采样率，通常是 `16000`
- `{language}`: 插件语言配置
- `{python}`: 启动当前服务的 Python 解释器

可选依赖检查：

- `required_python_modules`: 需要能 import 的 Python 模块名，例如 `["faster_whisper"]`
- `required_files`: 插件运行前必须存在的脚本或模型文件

示例：

```json
{
  "plugins": {
    "my_asr": {
      "name": "My ASR",
      "model": "D:/models/my-asr",
      "languages": "中文",
      "supports_words": false,
      "required_python_modules": ["your_asr_package"],
      "required_files": ["scripts/my_asr_cli.py"],
      "sample_rate": 16000,
      "language": "zh",
      "timeout_sec": 300,
      "command": [
        "{python}",
        "scripts/my_asr_cli.py",
        "--audio",
        "{audio}",
        "--device",
        "{device}",
        "--language",
        "{language}"
      ]
    }
  }
}
```

命令必须向 stdout 输出纯文本，或者输出 ASR JSON。

最小 JSON：

```json
{
  "text": "今天讨论项目进度。",
  "language": "zh"
}
```

带分段 JSON：

```json
{
  "text": "今天讨论项目进度。",
  "segments": [
    {
      "text": "今天讨论项目进度。",
      "start": 0.0,
      "end": 2.3,
      "speaker": null,
      "words": null
    }
  ],
  "timestamp_origin": "chunk",
  "speaker_scope": "none",
  "mode": "segmented",
  "is_final": true
}
```

项目会把输出统一规范化为 `asr-result-v1`，缺失的可选字段会自动补默认值。

## faster-whisper 示例

安装可选依赖：

```powershell
.\.venv-rocm-win\Scripts\python.exe -m pip install faster-whisper
```

在 `config/asr_plugins.json` 中启用示例：

```json
{
  "plugins": {
    "faster_whisper_large_v3": {
      "name": "faster-whisper large-v3",
      "model": "large-v3",
      "languages": "中文/英语/多语种",
      "supports_words": true,
      "required_python_modules": ["faster_whisper"],
      "required_files": ["scripts/asr_plugins/faster_whisper_cli.py"],
      "language": "zh",
      "timeout_sec": 600,
      "command": [
        "{python}",
        "scripts/asr_plugins/faster_whisper_cli.py",
        "--audio",
        "{audio}",
        "--model",
        "large-v3",
        "--device",
        "{device}",
        "--language",
        "{language}",
        "--word-timestamps",
        "--vad-filter"
      ]
    }
  }
}
```

## 接口

列出插件：

```http
GET /v1/asr/plugins
```

列出所有 ASR 引擎，包括插件：

```http
GET /v1/asr/engines
```

切换 ASR：

```http
PUT /v1/asr/engine
Content-Type: application/json

{"engine_type": "plugin:faster_whisper_large_v3"}
```

测试单个音频文件，不创建会议：

```powershell
curl.exe -F "engine_type=plugin:faster_whisper_large_v3" -F "file=@sample.wav" http://127.0.0.1:8321/v1/asr/test
```

设置页也增加了“ASR 样本测试”面板。它会调用 `/v1/asr/test`，用于先比较插件效果，再决定是否切换会议使用的 ASR。
