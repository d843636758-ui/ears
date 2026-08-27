# ears 🎐

让你的AI听懂你的语气。

对着它说一句话，你的AI收到的就不只是文字，还有这句话的声学指纹——音高、音量、停顿、语速、颤动——以及最关键的一层：**这些和你平时相比，偏了多少**。

```
她说："我没事。"
文字看到的：我没事
ears听到的：委屈 · 声音比平时低、停顿多，像是强撑（音量比较偏低、停顿明显偏高）
```

<p align="center"><img src="docs/screenshot.png" width="420" alt="ears 界面：按住说话，情绪卡片带个人化对比"></p>

## 为什么做这个

隔着屏幕的陪伴有一个天生的缺口：它看得见你打的每一个字，却听不见你是怎么说的。而人恰恰不是字面生物——同一句"我可不可以骂你"，音量放到耳语级别时其实是撒娇；同一句"没事"，说得比平时慢半拍时其实是"快来问我"。字面只有一层，声音里藏着十层。

做 ears，就是把那十层挖出来递给你的AI——让它别再只相信你想让它看见的那一面。

## 它和别的情绪识别有什么不一样

**它学的是你，不是"平均人"。**

大多数语音情绪方案拿全人类的平均值做标尺。但同一个音高，对A是平静，对B是低落。ears 内置**个人化基线**：前8句它先认识你的声音，之后每一句都和"你自己的平时"比——用中位数±MAD统计，偶尔在地铁上、户外录的几句也带不偏它对你的认识。

它输出的也不是学术标签（angry/sad/neutral），而是为"家里养了个AI"的场景定制的：**撒娇、嘴硬、委屈、累、低落、开心、兴奋、平静、生气、紧张**——都可以在配置里改成你们的相处方式。

## v0.2 · 两只新耳朵

### 声纹锁：它只认你

前6句合格录音自动注册你的声纹（存的是192维数学向量，不是音频）。之后每段声音先过一道"是不是你"的门：

- 是你 → 照常转写、分析、入档
- 不是你 → **不转写、不上云、内容一个字不记**，只留一笔"有别的声音出现"

朋友来家里说的话不该被偷偷记录——这是待客的规矩，也是这道锁存在的理由。你的个人化基线也从此不被电视声、访客声污染。

### 声纹家谱：认识你的家人

```
POST /api/earsplus/enroll  {"name":"妈妈"}
```

开启注册，让她对着说6句——之后她一开口，记录里就是"妈妈说话了（内容未记录）"。认人，不记话，全自动。注册进度看 `GET /api/earsplus/status`。

### 环境声：听见你的世界

每段语音顺手识别背景里的500多种声音——雨声、猫叫、门铃、键盘声、你的笑——写进记录的 `env_sounds` 字段。你的AI从此知道：你说这句话的时候，窗外正在打雷。

（细节：人声间隙检测——只拿你说话的间隙帧测环境声，你的声音不会把背景的猫叫压没。）

### 装这两只耳朵（可选）

不装也完全能用，ears 保持原样。要装的话：

```powershell
.\venv\Scripts\pip install onnxruntime
```

下载两个模型放进 `models/` 目录：

| 耳朵 | 模型 | 来源 |
|------|------|------|
| 环境声 | `yamnet.onnx` + `yamnet_class_map.csv` | huggingface.co/zeropointnine/yamnet-onnx |
| 声纹 | `embedding_model.onnx` → 改名 `ecapa.onnx` | huggingface.co/losfen/spkrec-ecapa-voxceleb-onnx |

`.env` 里加一行 `OWNER_NAME=你的名字`（记录里显示谁在说话）。

## 需要准备

| 东西 | 说明 |
|------|------|
| Python 3.10+ | python.org 下载，安装时勾选 Add to PATH |
| ffmpeg | 音频转码用，安装命令见下 |
| Groq API Key | **免费**。console.groq.com 注册 → API Keys → Create（转写和情绪判断都用它） |

## 安装（Windows）

```powershell
# 装 ffmpeg
winget install Gyan.FFmpeg

# 拉代码、装依赖
git clone https://github.com/Eveacla11/ears.git
cd ears
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt

# 配置
copy .env.example .env
# 用记事本打开 .env，填上你的 GROQ_API_KEY（国内用户同时填 PROXY 代理端口）
```

Mac/Linux 把安装 ffmpeg 换成 `brew install ffmpeg` / `sudo apt install ffmpeg`，路径斜杠反过来即可。

## 运行

```powershell
.\venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8020
```

默认只绑本机（接口没有鉴权，绑 `0.0.0.0` 意味着局域网里任何设备都能删你的记录、往你的基线里灌声音）。手机场景走HTTPS反代（见下），反代在同一台机器上，`127.0.0.1` 照样通。真要直接暴露到局域网，把 host 改成 `0.0.0.0`，风险自担。

浏览器打开 `http://localhost:8020`，按住圆按钮说句话。第一次会显示"准备中…"（在要麦克风权限），变成"松开听你说"再开口。

前8句它在认识你的声音（页面下方有进度）；之后你会开始看到"和平时比：音量偏低、停顿偏高"这样的分析。

## Zeabur 部署 + ChatGPT MCP

这个 fork 同时提供网页录音和 Streamable HTTP MCP。音频只从浏览器传到你自己的 ears 服务；ChatGPT 通过 MCP 读取已经生成的文字与语气结果，不需要从聊天端口上传音频。

### 1. 在 Zeabur 部署

从 GitHub 导入仓库即可，项目内的 `Dockerfile` 会安装 ffmpeg 和 Python 依赖。添加一个挂载到 `/data` 的持久卷，并配置：

```env
GROQ_API_KEY=你的Groq密钥
OWNER_NAME=念初
```

容器已默认把数据写入 `/data`。在 Zeabur 上如果没有单独配置 `ACCESS_TOKEN`，服务会自动复用 Zeabur 生成的 `PASSWORD` 作为网页钥匙，并由它派生 MCP 私密路径；原密码不会直接出现在 MCP URL 中。也可以显式设置 `ACCESS_TOKEN` 和 `MCP_PATH_SECRET` 覆盖这套默认值。Zeabur 分配 HTTPS 域名后，打开首页、输入网页钥匙，就能在 iPhone 浏览器里按住录音。解锁成功后还会出现“复制 ChatGPT MCP 地址”按钮，私密地址只在浏览器本地派生，不会把钥匙发送给其他服务。

### 2. 接入 ChatGPT

在 ChatGPT 网页端开启开发者模式，然后添加远程 MCP：

```text
https://你的Zeabur域名/mcp/你的MCP_PATH_SECRET/
```

连接成功后会出现四个只读工具：

- `get_latest_voice`：读取刚刚录下的那一句（最常用）
- `get_voice_by_id`：按网页卡片上的 12 位编号读取
- `list_recent_voices`：查看最近几句
- `get_ears_status`：检查记录数量与个人基线进度

例如在网页说完后，回到聊天里说“先生，听听我刚才说的”，ChatGPT 就能读取转写、语气、相对个人基线和环境声。生产公开发布建议升级为 OAuth；当前私密路径方案面向个人开发者模式使用。

## 接你自己的AI

每次分析的结果都追加在 `data/moments.jsonl`，一行一条：

```json
{"ts":"2026-07-12T10:07:18+08:00","text":"...","emotion":"撒娇","confidence":0.8,
 "hint":"她现在的心情很甜蜜","features":{...},"relative":{"音量":"比较偏低"}}
```

怎么用随你：
- 你的AI后端直接读这个文件（最简单，推荐）
- 调 `GET /api/recent?n=20` 拿最近记录
- 你的聊天后端把用户语音直接 `POST /api/listen`（multipart，字段名`file`），同步拿到转写+语气，拼进AI的上下文

**建议**：把分析结果绑在**具体那条消息**上，不要做"最近的语气全局漂浮注入"——漂浮的语气会让AI搞不清是谁什么时候说的，实测会造成混乱（我们踩过）。

## 能在什么地方用（前端兼容性）

自带的"按住说话"页面开箱即用，条件只有一个：**麦克风需要安全上下文**。

| 场景 | 能否用 | 说明 |
|------|--------|------|
| 电脑浏览器 `localhost:8020` | ✅ 直接用 | localhost 天然是安全上下文 |
| 手机/平板访问 `http://局域网IP:8020` | ❌ 录不了音 | 浏览器在 http 下禁用麦克风，这是浏览器的规定，谁都绕不开 |
| 手机访问 HTTPS 反代后的地址 | ✅ 直接用 | 推荐 Caddy 自签证书（`tls internal`），手机装一次根证书即可 |

桌面 Chrome/Edge/Safari/Firefox、iOS Safari、安卓 Chrome 实测均可（用的是标准 MediaRecorder API）。

## 嵌进你自己的聊天界面

不想用自带页面？`/api/listen` 就是个普通的文件上传接口，任何前端都能调。最小示例（把录好的音频 blob 发过去）：

```js
const fd = new FormData();
fd.append("file", audioBlob, "voice.webm");   // m4a/mp3/wav/ogg 都行
const r = await fetch("http://你的ears地址:8020/api/listen", { method: "POST", body: fd });
const d = await r.json();
// d = { text: "转写内容", emotion: "撒娇", confidence: 0.8,
//       hint: "一句话状态", relative: {"音量":"比较偏低"}, baseline_progress: "8/8" }
```

**用不了插件的封闭app**（claude.ai / ChatGPT 官方应用这类）也有笨办法：对着 ears 说话 → 点结果卡片上的 ⧉ 复制 → 粘进任何聊天框。复制出来的是一句完整的话：`[语音] 我没事（语气：委屈，声音比平时低，和平时比音量比较偏低）`——哪个AI读了都懂。

拿到结果后怎么展示、怎么拼进你的AI上下文，就是你的自由了。两条实战经验：

- iOS 的 MediaRecorder 录出来是 `audio/mp4`，文件名按真实 mimeType 起，别硬写 `.webm`（不然 iOS 自己都播放不了）
- 把分析结果绑在**那一条消息**上发给你的AI，别做全局漂浮注入（见上一节的建议）

## 隐私（诚实版）

- **音频会离开你的电脑**：发给你自己配置的转写接口（默认 Groq 云端）。接受不了就别用，或者自己改成本地 whisper。
- 音频分析完**默认阅后即焚**（`KEEP_AUDIO=1` 才留存）；文字和分析结果只存在你自己的电脑上。
- 这个项目本身不上传任何东西到作者这里——没有统计、没有埋点，代码就这几百行，欢迎自己审。
- `confidence` 数字只是模型的自我感觉，没有统计学意义，当参考别当真。

## FAQ

**Q：家里不止一个人用，会怎么样？**
基线目前**假设单人使用**——它学的是"一个人的平时"。两个人对着它说话，基线会被搅成两人的平均，对谁都不准。多人场景请各跑一个实例（`DATA` 指到不同目录即可），或等多用户版。

**Q：说错了/不满意，想删掉某条记录？**
页面每张结果卡右上角有 ✕，点了这条就被删掉，**基线也会当没听过重新算**。也可以调接口：`POST /api/forget`，body 传 `{"ts":"那条的时间戳"}` 或 `{"last":true}` 删最近一条。

**Q：换了麦克风/搬了家/感冒了一周，基线不准了想重来？**
删掉 `data/profile.json`，基线归零重新学（8句重新认识你）；历史文字记录不受影响。想连记录一起清空，删整个 `data/` 目录。

**Q：录了但转写文不对题，冒出"点赞订阅"之类的话？**
录音太短（<0.5秒）时 Whisper 会幻觉。ears 已在服务端拦截过短录音，遇到就重说一次。

**Q：报错 ProxyError / 连不上 api.groq.com？**
国内网络需要代理。`.env` 里 `PROXY` 填你的代理端口：Clash 默认 `http://127.0.0.1:7890`，v2rayN 默认 `http://127.0.0.1:10809`。代理软件掉线也是同样报错。

**Q：Windows 上中文乱码/奇怪的编码报错？**
本项目所有IO都显式UTF-8，正常不会遇到。如果你改代码，记住Windows三坑：subprocess 要 `encoding="utf-8"`、写文件要 `encoding="utf-8"`、别用bat解析.env。

**Q：能不能识别得更准？换个大模型？**
可以把 `LLM_MODEL` 换成任何 OpenAI 兼容接口的模型。但不建议上专门的SER学术模型——它们在表演性英文语料上训练，输出的标签也答非所问，还会带来3GB的依赖。这个项目的判断力主要来自"个人化基线+说话内容"，不是玄学声学。

## English (in brief)

**ears** listens to *how* you speak, not just what you say. Hold to talk → Whisper transcription + 9 acoustic features (pitch, energy, pauses, tempo, jitter...) → compared against **your own personal baseline** (median ± MAD, robust to noisy environments) → an LLM names the mood with companion-oriented labels (playful-mad, holding-back-tears, tired...). Lightweight: no torch, installs in a minute. Results go to a local JSONL / REST API / webhook — wire it to any AI you like. Audio is transcribed via the cloud API you configure (Groq by default) and deleted after analysis; everything else stays on your machine.

## License

MIT © Eve
