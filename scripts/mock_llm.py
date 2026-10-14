#!/usr/bin/env python3
"""Заглушка OpenAI-совместимого chat/completions для тестов.

Запуск: mock_llm.py ПОРТ ФАЙЛ_ЗАПРОСОВ [сценарий]

Сценарии:
  ok     — на любой запрос отдаёт один и тот же годный план;
  repair — первый ответ приходит с циклом в зависимостях (его отвергнет
           проверка плана), второй — исправленный. Так проверяется, что
           клиент показывает модели причину отказа и просит исправить,
           а не молча откатывается на эвристику.

План печатается в markdown-ограждении — так проверяется, что клиент умеет
снимать ```-ограждения и доставать JSON.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PLAN = {
    "goal": "стоимость закупки и доля одной части",
    "steps": [
        {"id": "s1", "role": "compute", "title": "стоимость закупки",
         "payload": "1250*3", "deps": []},
        {"id": "s2", "role": "compute", "title": "доля части",
         "payload": "(1250*3+480)/7", "deps": []},
        {"id": "s3", "role": "verify", "title": "сверка суммы",
         "payload": "3750", "deps": ["s1", "s2"]},
        {"id": "s4", "role": "memorize", "title": "итог",
         "payload": "стоимость закупки и доля части", "deps": ["s3"]},
    ],
}

# тот же план, но шаги зависят сами от себя — проверка обязана его отвергнуть
BROKEN = {
    "goal": "стоимость закупки и доля одной части",
    "steps": [
        {"id": "s1", "role": "compute", "title": "стоимость закупки",
         "payload": "1250*3", "deps": ["s3"]},
        {"id": "s2", "role": "compute", "title": "доля части",
         "payload": "(1250*3+480)/7", "deps": []},
        {"id": "s3", "role": "verify", "title": "сверка суммы",
         "payload": "3750", "deps": ["s1", "s2"]},
        {"id": "s4", "role": "memorize", "title": "итог",
         "payload": "стоимость закупки и доля части", "deps": ["s3"]},
    ],
}

SCENARIO = sys.argv[3] if len(sys.argv) > 3 else "ok"
SEEN = []


def reply(plan):
    content = "Конечно!\n\n```json\n" + json.dumps(plan, ensure_ascii=False) + "\n```\n"
    return json.dumps({
        "id": "chatcmpl-mock",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    }).encode()


MODELS = ["mock-model", "mock-model-mini"]


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        """Список моделей — так отвечает любой OpenAI-совместимый сервер;
        именно его спрашивает `dcctl llm`, чтобы убедиться, что сервер живой."""
        out = json.dumps({"object": "list",
                          "data": [{"id": m, "object": "model"} for m in MODELS]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(body)
        with open(sys.argv[2], "w") as f:
            json.dump({"request": body,
                       "requests": len(SEEN),
                       "auth": self.headers.get("Authorization"),
                       "content_type": self.headers.get("Content-Type")}, f,
                      ensure_ascii=False)
        out = reply(BROKEN if (SCENARIO == "repair" and len(SEEN) == 1) else PLAN)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()