"""NATS 行情消费端，API 对齐 QMT xtdata 的 subscribe_whole_quote / unsubscribe_quote。

兼容 sickle 统一 tick quote（TickPayload，见 sickle/data/tick/tick_quote.py）：
- 单 code：{"000001.SZ": {tick...}}
- 多 code 批量：{"code1": {tick...}, "code2": {tick...}, ...}
仍兼容旧 envelope：{"code": "...", "quote": {..., "time": ...}}
"""
import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

from credentials import NATS_CONSUMER_URL
from tools.utils_remote import is_tick_quote

try:
    from credentials import NATS_AM_SUBJECT as _DEFAULT_NATS_SUBJECT
except ImportError:
    from credentials import NATS_CONSUMER_SUBJECT as _DEFAULT_NATS_SUBJECT

Quotes = dict[str, dict[str, Any]]
QuoteCallback = Callable[[Quotes], None]

_SUB_OK = 0
_SUB_FAIL = -1


class AmNatsConsumer:
    def __init__(
        self,
        nats_url: str = NATS_CONSUMER_URL,
        nats_subject: str = _DEFAULT_NATS_SUBJECT,
        interval: float = 1.0,
    ) -> None:
        self.nats_url = nats_url
        self.nats_subject = nats_subject
        self.interval = interval
        self.quote_count = 0

        self._lock = threading.Lock()
        self._callback: QuoteCallback | None = None
        self._code_list: frozenset[str] = frozenset()
        self._quotes: Quotes = {}
        self._callback_running = False

        self._runner_thread: threading.Thread | None = None
        self._nc: NATS | None = None

    def subscribe_whole_quote(self, code_list: list[str], callback: QuoteCallback) -> int:
        with self._lock:
            if self._callback is not None:
                return _SUB_FAIL
            self._callback = callback
            self._code_list = frozenset(code_list)
            self._quotes = {}
            self.quote_count = 0
            need_start = self._runner_thread is None or not self._runner_thread.is_alive()

        if need_start:
            self._runner_thread = threading.Thread(
                target=self._run_loop,
                name='am-nats-consumer',
                daemon=True,
            )
            self._runner_thread.start()
        return _SUB_OK

    def unsubscribe_quote(self) -> int:
        with self._lock:
            if self._callback is None:
                return _SUB_FAIL
            self._callback = None
            self._code_list = frozenset()
            self._quotes = {}
        return _SUB_OK

    def update_code_list(self, code_list: list[str]) -> None:
        with self._lock:
            self._code_list = frozenset(code_list)

    def _run_loop(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        self._nc = await nats.connect(self.nats_url)
        await self._nc.subscribe(self.nats_subject, cb=self.on_message)
        print(f'[NATS] 已订阅 {self.nats_subject} @ {self.nats_url}', end='')
        try:
            await self._dispatch_loop()
        finally:
            await self._nc.drain()
            self._nc = None

    async def on_message(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        parsed = self._parse_quotes(data)
        if not parsed:
            return

        with self._lock:
            if self._callback is None:
                return
            if self._code_list:
                for code, quote in parsed.items():
                    if code in self._code_list:
                        self._quotes[code] = quote
                        self.quote_count += 1
            else:
                self._quotes.update(parsed)
                self.quote_count += len(parsed)

    async def _dispatch_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            with self._lock:
                subscribed = self._callback is not None
            if subscribed:
                await self._dispatch()

    async def _dispatch(self) -> None:
        if self._callback_running:
            return

        with self._lock:
            if self._callback is None or not self._quotes:
                return
            quotes, self._quotes = self._quotes, {}
            callback = self._callback

        self._callback_running = True
        try:
            await asyncio.to_thread(callback, quotes)
        finally:
            self._callback_running = False

    @staticmethod
    def _parse_quotes(data: Any) -> Quotes | None:
        if not isinstance(data, dict):
            return None

        code = data.get('code')
        quote = data.get('quote')
        if isinstance(code, str) and isinstance(quote, dict) and 'time' in quote:
            return {code: quote}

        quotes: Quotes = {}
        for item_code, item_quote in data.items():
            if isinstance(item_code, str) and is_tick_quote(item_quote):
                quotes[item_code] = item_quote

        if quotes:
            return quotes

        if len(data) == 1:
            item_code, item_quote = next(iter(data.items()))
            if isinstance(item_quote, dict) and 'time' in item_quote:
                return {str(item_code): item_quote}

        return None
