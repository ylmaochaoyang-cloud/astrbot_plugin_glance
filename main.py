"""余光：在 App 进程内偶尔看屏，按需说话，随 AstrBot 起落。

四档，写在 mode 里：

  silent  只记账：定时采样、写状态与日志，不推送、不插话
  push    记账 + 变化时推一句（旧版行为）
  look    不采样，只给模型挂一个 look_at_screen 工具，它想看才看
  both    采样推送 + 工具，两样都在

默认 silent。装上不读文档，也不会开始截屏或说话。

其他取向：
  - 不起外部进程、不挂计划任务，插件加载即开工，AstrBot 关掉就随之结束；
  - 识图是同步 HTTP，全部丢进线程池（asyncio.to_thread），不阻塞事件循环；
  - 视觉模型输出 JSON：app / doing / texts / confidence，另附一句人话摘要；
  - 找不到 JSON 就退回整段文本当摘要，不崩。
"""
import asyncio
import base64
import io
import json
import os
import random
import time
import urllib.request
from pathlib import Path

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.message.message_event_result import MessageChain
except Exception:
    from astrbot.api.event import MessageChain  # type: ignore

try:
    from astrbot.api.message_components import Plain
except Exception:
    from astrbot.core.message.components import Plain  # type: ignore

DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
SAMPLE_SECONDS = 300
PROB_PERCENT = 20
KEEP_SHOTS = 200
LOG_MAX = 2 * 1024 * 1024
MODES = ("silent", "push", "look", "both")

PROMPT = (
    "这是用户电脑当前屏幕。只输出一个 JSON 对象，不要解释、不要代码块围栏。格式："
    '{"app":"正在使用的程序或页面","doing":"他大概在做什么，一句话",'
    '"texts":["画面里显眼的三到五条文字"],"confidence":0.0}'
    " 看不清或不确定的字段留空字符串或空数组，不要猜测。"
)


def _data_dir() -> Path:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path  # type: ignore
        return Path(get_astrbot_data_path()) / "plugin_data" / "eye_watch"
    except Exception:
        pass
    try:
        from astrbot.core.star.star_tools import StarTools  # type: ignore
        return Path(StarTools.get_data_dir("eye_watch"))
    except Exception:
        return Path.cwd() / "eye_watch_data"


def _parse_vision(text: str) -> dict:
    """从模型输出里抠出 JSON；抠不到就把整段当摘要。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, dict):
                app = str(data.get("app") or "").strip()
                doing = str(data.get("doing") or "").strip()
                texts = data.get("texts")
                texts = [str(t).strip() for t in texts if str(t).strip()][:5] if isinstance(texts, list) else []
                try:
                    conf = float(data.get("confidence") or 0)
                except Exception:
                    conf = 0.0
                summary = "｜".join(x for x in (app, doing) if x) or "看不清"
                return {"app": app, "doing": doing, "texts": texts,
                        "confidence": max(0.0, min(1.0, conf)), "summary": summary}
        except Exception:
            pass
    return {"app": "", "doing": "", "texts": [], "confidence": 0.0,
            "summary": raw[:200] or "（空回答）"}


@register("eye_watch", "余光", "偶尔看一眼屏幕，按档位决定说不说，随 AstrBot 起落", "0.5.0")
class EyeWatch(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        cfg = config or {}
        base = _data_dir()
        # 旧配置只有 enable；enable=true 视为 push，保持原行为。
        mode = str(cfg.get("mode") or "").strip().lower()
        if mode not in MODES:
            mode = "push" if cfg.get("enable", False) else "silent"
        self.mode: str = mode
        self.target: str = str(cfg.get("target_umo") or "").strip()
        self.interval: float = float(cfg.get("sample_seconds", SAMPLE_SECONDS) or SAMPLE_SECONDS)
        self.prob: int = max(0, min(100, int(cfg.get("prob_percent", PROB_PERCENT) or PROB_PERCENT)))
        self.model: str = str(cfg.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)
        self.ollama: str = str(cfg.get("ollama_url", DEFAULT_OLLAMA) or DEFAULT_OLLAMA)
        self.shot_dir: str = str(cfg.get("shot_dir") or (base / "shots"))
        self.log_path: str = str(cfg.get("log_path") or (base / "look_out.jsonl"))
        self.status_path: str = str(cfg.get("status_path") or (base / "eye_status.json"))
        self.state_path: str = str(cfg.get("state_path") or (base / "eye_state.json"))
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_summary = ""
        self._turn = 0

    # ---------- 生命周期 ----------

    async def initialize(self) -> None:
        if self.mode not in ("silent", "push", "both"):
            logger.info("[eye_watch] mode=%s，不采样；只挂 look_at_screen 工具", self.mode)
            return
        if not self.target:
            logger.warning("[eye_watch] 已启用采样但没有配置 target_umo，为免截了没处送，本次不启动")
            return
        os.makedirs(self.shot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._task = asyncio.create_task(self._loop())
        logger.info("[eye_watch] 已启动 mode=%s：每 %ss 掷骰，%s%% 命中",
                    self.mode, int(self.interval), self.prob)

    async def terminate(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("[eye_watch] 已停止")

    # ---------- 采集 ----------

    def _capture(self) -> dict:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        w, h = img.size
        if w > 1280:
            img = img.resize((1280, int(h * 1280 / w)))
        path = os.path.join(self.shot_dir, "shot_" + time.strftime("%Y%m%d-%H%M%S") + ".png")
        img.save(path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        body = json.dumps({
            "model": self.model, "prompt": PROMPT,
            "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False, "think": False, "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 1024},
        }).encode()
        req = urllib.request.Request(self.ollama.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=90).read())
        parsed = _parse_vision(r.get("response") or "")
        parsed["shot"] = path
        return parsed

    def _housekeeping(self) -> None:
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > LOG_MAX:
                os.replace(self.log_path, self.log_path + ".1")
        except Exception:
            pass
        try:
            files = sorted((os.path.join(self.shot_dir, f) for f in os.listdir(self.shot_dir)
                            if f.endswith(".png")), key=os.path.getmtime)
            for p in files[: max(0, len(files) - KEEP_SHOTS)]:
                os.remove(p)
        except Exception:
            pass

    def _write(self, rec: dict) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8", newline="") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        try:
            tmp = self.status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "mode": self.mode, "interval": self.interval,
                           "prob": self.prob, "state": "running", "turn": self._turn,
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "last": rec if rec.get("hit") else None},
                          f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.status_path)
        except Exception:
            pass
        if rec.get("hit") and rec.get("summary"):
            try:
                tmp = self.state_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.state_path)
            except Exception:
                pass

    # ---------- 主循环 ----------

    async def _loop(self) -> None:
        while not self._stop.is_set():
            self._turn += 1
            roll = random.randrange(100)
            hit = roll < self.prob
            rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "turn": self._turn,
                   "roll": roll, "hit": hit}
            if hit:
                t0 = time.time()
                try:
                    rec.update(await asyncio.to_thread(self._capture))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                rec["elapsed"] = round(time.time() - t0, 1)
                if self.mode in ("push", "both"):
                    await self._maybe_push(rec)
            self._write(rec)
            await asyncio.to_thread(self._housekeeping)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    async def _maybe_push(self, rec: dict) -> None:
        summary = (rec.get("summary") or "").strip()
        if rec.get("error") or not summary or summary == self._last_summary:
            return
        if not self.target:
            return
        try:
            await self.context.send_message(self.target, MessageChain([Plain("看了一眼：" + summary)]))
            self._last_summary = summary
        except Exception as e:
            logger.warning("[eye_watch] 推送失败：%s", e)

    # ---------- 给模型的工具 ----------

    @filter.llm_tool(name="look_at_screen")
    async def look_at_screen(self, event: AstrMessageEvent) -> str:
        """看一眼用户当前的屏幕，返回正在用的程序、大概在做什么、画面上的文字。

        当用户问"我在干嘛""这关怎么过""屏幕上是什么""帮我看看这个界面"时调用。
        不要在没有必要时频繁调用；一次调用需要几秒钟。
        """
        if self.mode not in ("look", "both"):
            return json.dumps({"status": "disabled",
                               "message": "看屏已关闭。不要向用户声称看过屏幕。"},
                              ensure_ascii=False)
        try:
            rec = await asyncio.to_thread(self._capture)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"},
                              ensure_ascii=False)
        rec.update({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "turn": self._turn, "hit": True})
        self._write(rec)
        return json.dumps({"status": "ok", "app": rec.get("app"), "doing": rec.get("doing"),
                           "texts": rec.get("texts"), "confidence": rec.get("confidence"),
                           "summary": rec.get("summary")},
                          ensure_ascii=False)

    # ---------- 指令 ----------

    @filter.command("eye")
    async def eye_cmd(self, event: AstrMessageEvent):
        """看看眼睛状态：/eye"""
        if self.mode not in ("silent", "push", "both"):
            yield event.plain_result(f"mode={self.mode}，未采样；需要时让模型调用 look_at_screen。")
            return
        try:
            d = json.load(open(self.status_path, encoding="utf-8"))
            yield event.plain_result(
                f"mode={self.mode} / 第 {d.get('turn')} 轮 / {d.get('state')} / 更新于 {d.get('updated_at')}")
        except Exception as e:
            yield event.plain_result(f"读不到状态：{e}")
