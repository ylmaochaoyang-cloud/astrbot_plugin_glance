"""眼睛：在 App 进程内偶尔看屏、命中才推送，随 AstrBot 起落。

设计取向是"轻"：
  - 总开关默认关闭。装上不读文档也不会开始截屏，先让人觉得体贴，而不是被盯着；
  - 不起外部进程、不挂计划任务，插件加载即开工，AstrBot 关掉就随之结束；
  - 每 sample_seconds 掷一次骰子，只有 prob_percent 的概率真的去看；
  - 识图是同步 HTTP，全部丢进线程池（asyncio.to_thread），不阻塞事件循环；
  - 描述与上次相同就不推送，其余时间安静；
  - 写 look_out.jsonl 与 eye_status.json，供外部查看与自检插件核对。

不做自恢复：坏了就在状态文件里留着旧时间戳，靠人在对话里发现。
"""
from __future__ import annotations

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
PROMPT = ("这是用户电脑当前屏幕。用中文，三句话内说清："
          "1) 正在用什么程序或页面；2) 他大概在做什么；3) 画面上显眼的文字。"
          "看不清就说看不清，不要编。")


def _data_dir() -> Path:
    """插件自己的数据目录；拿不到就退回当前目录下的 eye_watch_data。"""
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


@register("eye_watch", "眼睛", "进程内偶尔看屏、命中才推送，随 AstrBot 起落", "0.4.0")
class EyeWatch(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        cfg = config or {}
        base = _data_dir()
        self.enabled: bool = bool(cfg.get("enable", False))
        self.target: str = str(cfg.get("target_umo") or "").strip()
        self.interval: float = float(cfg.get("sample_seconds", SAMPLE_SECONDS) or SAMPLE_SECONDS)
        self.prob: int = max(0, min(100, int(cfg.get("prob_percent", PROB_PERCENT) or PROB_PERCENT)))
        self.model: str = str(cfg.get("model", DEFAULT_MODEL) or DEFAULT_MODEL)
        self.ollama: str = str(cfg.get("ollama_url", DEFAULT_OLLAMA) or DEFAULT_OLLAMA)
        self.shot_dir: str = str(cfg.get("shot_dir") or (base / "shots"))
        self.log_path: str = str(cfg.get("log_path") or (base / "look_out.jsonl"))
        self.status_path: str = str(cfg.get("status_path") or (base / "eye_status.json"))
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_text = ""
        self._turn = 0

    async def initialize(self) -> None:
        if not self.enabled:
            logger.info("[eye_watch] 未启用（enable=false），本次不截屏也不推送")
            return
        if not self.target:
            logger.warning("[eye_watch] 已启用但没有配置 target_umo，为免截了没处送，本次不启动")
            return
        os.makedirs(self.shot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._task = asyncio.create_task(self._loop())
        logger.info("[eye_watch] 已启动：每 %ss 掷骰，%s%% 命中", int(self.interval), self.prob)

    async def terminate(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("[eye_watch] 已停止")

    # ---------- 采集 ----------

    def _capture(self) -> dict:
        """同步动作，跑在线程池里：截图 + 问本地视觉模型。"""
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
        return {"text": (r.get("response") or "").strip() or "（空回答）", "shot": path}

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
                json.dump({"pid": os.getpid(), "interval": self.interval, "prob": self.prob,
                           "state": "running", "turn": self._turn,
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "last": rec if rec.get("hit") else None},
                          f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.status_path)
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
                await self._maybe_push(rec)
            self._write(rec)
            await asyncio.to_thread(self._housekeeping)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    async def _maybe_push(self, rec: dict) -> None:
        text = (rec.get("text") or "").strip()
        if rec.get("error") or not text or text == self._last_text:
            return
        try:
            await self.context.send_message(self.target, MessageChain([Plain("看了一眼：" + text)]))
            self._last_text = text
        except Exception as e:
            logger.warning("[eye_watch] 推送失败：%s", e)

    @filter.command("eye")
    async def eye_cmd(self, event: AstrMessageEvent):
        """看看眼睛状态：/eye"""
        if not self.enabled:
            yield event.plain_result("眼睛没开（enable=false）。")
            return
        try:
            d = json.load(open(self.status_path, encoding="utf-8"))
            yield event.plain_result(
                f"第 {d.get('turn')} 轮 / {d.get('state')} / 更新于 {d.get('updated_at')}")
        except Exception as e:
            yield event.plain_result(f"读不到状态：{e}")
