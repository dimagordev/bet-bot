# Project: Bet AI Telegram Bot + WebApp (mini‑app)

This single document contains a full minimal implementation of **bot + webapp** with:
- Telegram main menu (Регистрация, Язык, Инструкция, Поддержка, Получить сигнал)
- Real **Telegram WebApp** (mini‑app) that opens by the button «Получить Сигнал»
- Upload screenshot → analyze with OpenAI Vision → implied probabilities + top outcome
- Language switcher inside the webapp
- **Registration & Deposit verification hooks** (server endpoints). You must provide either:
  - a real bookmaker API integration (1win or another) via the stub `BookmakerVerifier`, or
  - a simple webhook/callback from your affiliate network to our `/api/affiliate/callback` endpoint.

> ⚠️ Внимание: у 1win нет публичной документации по API. В коде ниже — аккуратные **заглушки**, которые ждут либо ваш токен/endpoint от партнёрской сети, либо простую схему подтверждения через реф‑параметр и callback. Я оставил понятные TODO‑места и тесты. Как только дадите конкретику по API (URL/ключи/схема), я включу реальную проверку.

---

## 0) How to run (Windows OK, venv optional)
1. Install Python 3.11+ (with SSL). During installation tick **Add Python to PATH**.
2. Create project folder, add two files from this doc:
   - `main.py` (bot + web server)
   - `webapp/` with `index.html`, `app.js`, `styles.css` (embedded below – save each to the path shown)
   - `requirements.txt` (deps listed below)
3. Install deps:
   ```powershell
   pip install -r requirements.txt
   ```
4. Create `.env` in project root:
   ```dotenv
   TELEGRAM_BOT_TOKEN=123456:ABC...             # BotFather token
   OPENAI_API_KEY=sk-...                         # OpenAI key
   SUPPORT_USERNAME=your_support_username        # without @
   WEBAPP_PUBLIC_URL=https://your-domain.tld     # where the webapp is served (no trailing slash)
   AFFILIATE_REG_URL=https://1win.example/reg?ref=YOUR_REF  # your real ref link (used in buttons)
   # Optional — if you have affiliate callbacks/API:
   AFFILIATE_CALLBACK_SECRET=change_me
   AFFILIATE_API_BASE=https://affiliate.api.example
   AFFILIATE_API_KEY=your_api_key
   ```
5. Run tests (no network required):
   ```powershell
   set RUN_TESTS=1
   python main.py
   ```
   Expect: `All tests passed! ✅`
6. Launch server + bot:
   ```powershell
   python main.py
   ```
   The bot will run and the web server (FastAPI) will serve at `http://127.0.0.1:8000` by default.
   Update `WEBAPP_PUBLIC_URL` to a real domain with HTTPS when you deploy.

`requirements.txt`:
```
aiogram==3.12.0
fastapi==0.115.2
uvicorn==0.30.6
python-dotenv==1.0.1
openai==1.52.2
Pillow==10.4.0
httpx==0.27.2
jinja2==3.1.4
```

---

## 1) File: main.py (bot + web server + analysis + verification)
```python
from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

# Lazy imports inside functions avoid SSL/module issues during tests

load_dotenv()

# ---------------- Env helpers ----------------

def _get_env(name: str, *, required: bool = False, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# ---------------- Core odds utils ----------------

num_re = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        m = num_re.search(x.strip())
        if not m:
            return None
        token = m.group(0).replace(',', '.')
        try:
            return float(token)
        except Exception:
            return None
    return None

@dataclass
class OddsParse:
    teams: Dict[str, Optional[str]]
    market: str
    odds: Dict[str, float]
    confidence: float

    @staticmethod
    def from_json(d: Dict) -> "OddsParse":
        teams = d.get('teams', {}) or {}
        market = d.get('market', '1X2')
        odds = d.get('odds', {}) or {}
        c_raw = d.get('source_confidence', 0)
        try:
            conf = float(c_raw)
        except Exception:
            conf = 0.0
        normalized: Dict[str, float] = {}
        for k, v in odds.items():
            val = _safe_float(v)
            if val is None:
                continue
            key = str(k).lower().strip()
            if key in {'home','red','п1','team1','red_win','home_win','1'}:
                normalized['red_win'] = val
            elif key in {'away','blue','п2','team2','blue_win','away_win','2'}:
                normalized['blue_win'] = val
            elif key in {'draw','ничья','x','х'}:
                normalized['draw'] = val
            else:
                normalized[key] = val
        return OddsParse(teams=teams, market=market, odds=normalized, confidence=conf)


def odds_to_probs(odds: Dict[str, float]) -> Dict[str, float]:
    clean = {k: float(v) for k, v in odds.items() if isinstance(v, (int, float)) and 1.01 < v < 1000}
    if not clean:
        return {}
    inv = {k: 1.0/v for k, v in clean.items()}
    s = sum(inv.values())
    if s <= 0:
        return {}
    return {k: (v/s)*100.0 for k, v in inv.items()}


def pick_best(probs: Dict[str, float]) -> Tuple[str, float]:
    if not probs:
        return ("", 0.0)
    k = max(probs, key=lambda x: probs[x])
    return k, probs[k]


def _strip_code_fences(text: str) -> str:
    t = (text or '').strip()
    if len(t) >= 6 and t.startswith('```') and t.endswith('```'):
        inner = t[3:-3].strip()
        m = re.match(r"^[A-Za-z0-9_+-]+\n(.*)$", inner, flags=re.S)
        return (m.group(1) if m else inner).strip()
    return t

# ---------------- OpenAI (Vision) ----------------
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=_get_env('OPENAI_API_KEY', required=True))
    return _openai_client

async def analyze_image_bytes(img_bytes: bytes) -> Dict[str, object]:
    client = _get_openai_client()
    data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('ascii')}"
    system = (
        "You are a precise OCR and information extraction assistant. "
        "From a sportsbook screenshot, extract 1X2 market odds as DECIMAL odds when possible. "
        "Return STRICT JSON with keys: teams {red, blue}, market, odds {red_win, draw, blue_win}, source_confidence (0..1). "
        "Support Russian labels: П1=home/red, Х or X=draw, П2=away/blue."
    )
    user_prompt = (
        "Extract odds and team names. If colors aren't labeled, map first team to red/home (П1), second to blue/away (П2). "
        "Only output JSON, no extra text."
    )
    resp = client.responses.create(
        model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'),
        temperature=0.1,
        input=[
            {"role":"system","content":[{"type":"text","text":system}]},
            {"role":"user","content":[{"type":"input_text","text":user_prompt},{"type":"input_image","image_url":data_url}]}
        ]
    )
    text = getattr(resp, 'output_text', '') or ''
    text = _strip_code_fences(text)
    try:
        data = json.loads(text)
    except Exception:
        data = {"teams":{"red":None,"blue":None},"market":"1X2","odds":{},"source_confidence":0}
    # Normalize odds numbers
    if isinstance(data, dict) and isinstance(data.get('odds'), dict):
        data['odds'] = {k:_safe_float(v) for k,v in data['odds'].items() if _safe_float(v) is not None}
    parsed = OddsParse.from_json(data)
    probs = odds_to_probs(parsed.odds)
    best_key, best_val = pick_best(probs)
    return {
        "parsed": parsed.__dict__,
        "probs": probs,
        "best": {"key": best_key, "pct": best_val}
    }

# ---------------- Affiliate / Bookmaker verification ----------------
class BookmakerVerifier:
    """Pluggable verification. Replace with your affiliate or bookmaker API details.
    Strategy:
      1) When a user presses «Регистрация», we open your ref link with a unique tag ?uid=<telegram_id>.
      2) Your affiliate platform calls our webhook /api/affiliate/callback with uid and status (registered/deposited).
      3) Or, if you have a polling API, configure AFFILIATE_API_* and we poll by uid.
    """
    def __init__(self):
        self.api_base = os.getenv('AFFILIATE_API_BASE')
        self.api_key = os.getenv('AFFILIATE_API_KEY')
        self.callback_secret = os.getenv('AFFILIATE_CALLBACK_SECRET')

    async def poll_status(self, uid: str) -> Dict[str, bool]:
        """Return {'registered': bool, 'deposited': bool}. If no API configured, fallback to memory store."""
        if not self.api_base or not self.api_key:
            # Fallback in-memory store touched by /api/affiliate/callback
            return _MemDB.get(uid)
        import httpx
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=15) as client:
            # Example only — replace with your endpoint contract
            r = await client.get(f"{self.api_base}/player/status", params={"uid": uid}, headers=headers)
            r.raise_for_status()
            j = r.json()
            return {"registered": bool(j.get('registered')), "deposited": bool(j.get('deposited'))}

    def build_ref_url(self, telegram_id: int) -> str:
        base = _get_env('AFFILIATE_REG_URL', required=True)
        sep = '&' if ('?' in base) else '?'
        return f"{base}{sep}uid={telegram_id}"

# Simple in-memory status if no external API; updated by webhook
class _MemDB:
    store: Dict[str, Dict[str,bool]] = {}
    @classmethod
    def update(cls, uid: str, registered: Optional[bool]=None, deposited: Optional[bool]=None):
        row = cls.store.get(uid, {"registered": False, "deposited": False})
        if registered is not None: row['registered'] = registered
        if deposited is not None: row['deposited'] = deposited
        cls.store[uid] = row
    @classmethod
    def get(cls, uid: str) -> Dict[str,bool]:
        return cls.store.get(uid, {"registered": False, "deposited": False})

# ---------------- FastAPI (web server + webapp) ----------------

def create_web_app():
    from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Serve static webapp (index.html, js, css)
    static_dir = os.path.join(os.path.dirname(__file__), 'webapp')
    app.mount('/static', StaticFiles(directory=static_dir), name='static')

    verifier = BookmakerVerifier()

    @app.get('/', response_class=HTMLResponse)
    async def root():
        # Basic index to confirm server is up
        with open(os.path.join(static_dir, 'index.html'), 'r', encoding='utf-8') as f:
            return f.read()

    @app.post('/api/analyze')
    async def api_analyze(file: UploadFile = File(...)):
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail='Only image files are accepted')
        data = await file.read()
        # Normalize to JPG reasonable size
        img = Image.open(io.BytesIO(data)).convert('RGB')
        max_dim = 1600
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        out = io.BytesIO()
        img.save(out, format='JPEG', quality=90)
        result = await analyze_image_bytes(out.getvalue())
        return JSONResponse(result)

    @app.post('/api/affiliate/callback')
    async def affiliate_callback(request: Request):
        # Expect JSON: {"uid": "<telegram_id>", "status": "registered|deposited", "secret": "..."}
        body = await request.json()
        secret = os.getenv('AFFILIATE_CALLBACK_SECRET')
        if secret and body.get('secret') != secret:
            raise HTTPException(status_code=403, detail='Invalid secret')
        uid = str(body.get('uid'))
        status = str(body.get('status'))
        if not uid or status not in {'registered','deposited'}:
            raise HTTPException(status_code=400, detail='Bad payload')
        if status == 'registered':
            _MemDB.update(uid, registered=True)
        else:
            _MemDB.update(uid, deposited=True)
        return {"ok": True, "state": _MemDB.get(uid)}

    @app.get('/api/status')
    async def api_status(uid: str):
        return _MemDB.get(uid)

    return app

# ---------------- Aiogram bot ----------------

def create_bot():
    # Ensure ssl exists
    import ssl  # noqa: F401
    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import Command
    from aiogram.types import (Message, InlineKeyboardButton, InlineKeyboardMarkup,
                               CallbackQuery, WebAppInfo)

    bot = Bot(_get_env('TELEGRAM_BOT_TOKEN', required=True))
    dp = Dispatcher()
    verifier = BookmakerVerifier()

    def main_menu_kb(lang='ru') -> InlineKeyboardMarkup:
        web_url = _get_env('WEBAPP_PUBLIC_URL', required=True)
        kb = [
            [InlineKeyboardButton(text='🧾 Регистрация', url=verifier.build_ref_url(0))],  # uid will be added at runtime
            [InlineKeyboardButton(text='🌐 Выбрать язык', callback_data='lang')],
            [InlineKeyboardButton(text='📘 Инструкция', callback_data='help')],
            [InlineKeyboardButton(text='🆘 Поддержка', url=f"https://t.me/{_get_env('SUPPORT_USERNAME', required=True)}")],
            [InlineKeyboardButton(text='⚜️ Получить Сигнал', web_app=WebAppInfo(url=f"{web_url}/#tg"))],
        ]
        return InlineKeyboardMarkup(inline_keyboard=kb)

    @dp.message(Command('start'))
    async def cmd_start(m: Message):
        # personalize ref URL with user id
        kb = main_menu_kb()
        # Patch first button url to include uid
        kb.inline_keyboard[0][0].url = verifier.build_ref_url(m.from_user.id)
        await m.answer_photo(photo='https://picsum.photos/1200/600',
                             caption='Главное меню:',
                             reply_markup=kb)

    @dp.callback_query(F.data == 'help')
    async def cb_help(c: CallbackQuery):
        text = (
            "\U0001F916 Бот обучен для анализа скринов из БК.\n\n"
            "1) Зарегистрируйся по кнопке и ВАЖНО: используй промокод (если требуется).\n"
            "2) Внеси депозит.\n"
            "3) Нажми \"Получить сигнал\" и загрузи скрин события.\n"
            "4) Получи имплицитные вероятности и самый вероятный исход.\n\n"
            "⚠️ Это не финансовый совет. 18+."
        )
        await c.message.answer(text)
        await c.answer()

    @dp.callback_query(F.data == 'lang')
    async def cb_lang(c: CallbackQuery):
        # simple list, extend as needed
        langs = ['Русский','English','Español','Türkçe','Português','Oʻzbek','한국어','العربية']
        rows = []
        for i in range(0, len(langs), 2):
            chunk = langs[i:i+2]
            row = [InlineKeyboardButton(text=name, callback_data=f"setlang:{name}") for name in chunk]
            rows.append(row)
        await c.message.answer('Select language', reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
        await c.answer()

    @dp.callback_query(F.data.startswith('setlang:'))
    async def cb_setlang(c: CallbackQuery):
        lang = c.data.split(':',1)[1]
        await c.message.answer(f'Язык установлен: {lang}')
        await c.answer()

    @dp.message(F.text.lower().in_({"проверка", "статус"}))
    async def check_status(m: Message):
        state = await verifier.poll_status(str(m.from_user.id))
        txt = ("\n".join([
            f"Регистрация: {'✅' if state['registered'] else '❌'}",
            f"Депозит: {'✅' if state['deposited'] else '❌'}"
        ]))
        await m.answer(txt)

    return bot, dp

# ---------------- Tests ----------------

def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol

def run_tests() -> None:
    print('Running tests…')
    assert _safe_float('1,85') == 1.85
    assert _safe_float('3.25') == 3.25
    assert _safe_float('x=4,5y') == 4.5
    assert _safe_float('abc') is None
    fenced = """```json\n{\n  \"a\": 1\n}\n```"""
    assert _strip_code_fences(fenced) == '{\n  "a": 1\n}'
    parsed = OddsParse.from_json({"odds": {"П1": 2.0, "Х": 3.5, "П2": 4.0}})
    probs = odds_to_probs(parsed.odds)
    assert _approx(sum(probs.values()), 100.0, 1e-6)
    k, v = pick_best(probs)
    assert k == 'red_win' and v == max(probs.values())
    # two-outcome
    parsed2 = OddsParse.from_json({"odds": {"П1": 1.80, "П2": 2.00}})
    probs2 = odds_to_probs(parsed2.odds)
    assert 'draw' not in parsed2.odds
    assert _approx(sum(probs2.values()), 100.0, 1e-6)
    print('All tests passed! ✅')

# ---------------- Entrypoint ----------------

async def start_everything():
    # Web server
    app = create_web_app()
    import uvicorn
    config = uvicorn.Config(app, host='0.0.0.0', port=8000, log_level='info')
    server = uvicorn.Server(config)

    # Bot
    bot, dp = create_bot()

    async def run_server():
        await server.serve()

    async def run_bot():
        print('Bot is running…')
        await dp.start_polling(bot)

    await asyncio.gather(run_server(), run_bot())


def _entrypoint():
    try:
        # If already inside an event loop (Jupyter), schedule background
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(start_everything())
    else:
        asyncio.create_task(start_everything())
        print('Scheduled bot+server background task')

if __name__ == '__main__':
    if os.getenv('RUN_TESTS') == '1':
        run_tests()
    else:
        _entrypoint()
```

---

## 2) WebApp files (save to `webapp/`)

### `webapp/index.html`
```html
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ставки от AI</title>
  <link rel="stylesheet" href="/static/styles.css" />
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
</head>
<body>
  <div class="container">
    <header class="header">
      <div class="brand">💡 Ставки от AI</div>
      <div class="lang">
        <select id="lang">
          <option value="ru">RU</option>
          <option value="en">EN</option>
        </select>
      </div>
    </header>

    <div class="card">
      <label>Тип игры</label>
      <select id="gameType">
        <option>Футбол</option>
        <option>Хоккей</option>
        <option>Баскетбол</option>
      </select>
    </div>

    <div class="card">
      <label>Загрузите изображение события</label>
      <input id="file" type="file" accept="image/*" />
      <div id="preview"></div>
      <button id="analyze">🔎 Проанализировать ставку</button>
      <div id="progress" class="progress hidden">Анализируем…</div>
      <div id="quota">Осталось попыток на сегодня: <span id="quotaCount">10</span> из 10</div>
    </div>

    <div class="card hidden" id="result">
      <div class="result-row"><span>Уверенность AI:</span> <b id="conf">—</b></div>
      <div class="result-row"><span>Надёжный сигнал:</span> <b id="best"></b></div>
      <div id="bars"></div>
      <div class="note">⚠️ Это не финансовый совет. 18+.</div>
    </div>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

### `webapp/styles.css`
```css
:root { --bg:#0f1420; --panel:#131a2a; --text:#e6eefc; --muted:#9bb0d3; --accent:#3be091; }
* { box-sizing: border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system; background: var(--bg); color: var(--text); }
.container { max-width: 720px; margin: 24px auto; padding: 0 12px; }
.header { display:flex; justify-content: space-between; align-items:center; margin-bottom:12px; }
.brand { font-weight: 700; font-size: 20px; }
.card { background: var(--panel); border-radius: 16px; padding: 16px; margin: 12px 0; box-shadow: 0 6px 20px rgba(0,0,0,.3); }
label { display:block; margin-bottom:8px; color: var(--muted); }
select, input[type=file] { width:100%; padding:10px; border-radius:10px; border:1px solid #22314f; background:#0d1320; color:var(--text); }
button { margin-top:10px; width:100%; padding:12px; border:0; border-radius:12px; background: var(--accent); color:#091117; font-weight:700; cursor:pointer; }
.progress { margin-top:10px; opacity:.9; }
.hidden { display:none; }
#preview img { max-width:100%; margin-top:10px; border-radius: 12px; }
.result-row { display:flex; justify-content: space-between; padding:6px 0; }
.bar { height:16px; border-radius:8px; background: linear-gradient(90deg, #31d58a, #67f3b8); margin:6px 0; }
.bar-wrap { background:#0a0f1b; border-radius:8px; padding:2px; }
.note { color: var(--muted); margin-top: 8px; font-size: 12px; }
```

### `webapp/app.js`
```javascript
const tg = window.Telegram?.WebApp;
if (tg) { tg.expand(); }

const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyze');
const progress = document.getElementById('progress');
const quotaCount = document.getElementById('quotaCount');
const resultCard = document.getElementById('result');
const bestEl = document.getElementById('best');
const confEl = document.getElementById('conf');
const barsEl = document.getElementById('bars');

let quota = 10;

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.innerHTML = `<img src="${e.target.result}"/>`;
  };
  reader.readAsDataURL(f);
});

analyzeBtn.addEventListener('click', async () => {
  if (quota <= 0) { alert('Лимит попыток исчерпан на сегодня'); return; }
  const f = fileInput.files[0];
  if (!f) { alert('Сначала загрузите изображение.'); return; }
  progress.classList.remove('hidden');
  resultCard.classList.add('hidden');

  const form = new FormData();
  form.append('file', f);
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();
    quota--; quotaCount.textContent = String(quota);
    progress.classList.add('hidden');

    const probs = data.probs || {};
    const labels = { red_win: 'Победа Красных (П1)', draw: 'Ничья (Х)', blue_win: 'Победа Синих (П2)' };
    bestEl.textContent = labels[data.best?.key] ? `${labels[data.best.key]} — ${data.best.pct?.toFixed(1)}%` : '—';
    const conf = (data.parsed?.confidence || 0) * 100;
    confEl.textContent = `${Math.round(conf)}%`;

    barsEl.innerHTML = '';
    ['red_win','draw','blue_win'].forEach(k => {
      if (probs[k] !== undefined) {
        const pct = probs[k];
        const row = document.createElement('div');
        row.innerHTML = `<div style="display:flex;justify-content:space-between"><span>${labels[k]}</span><b>${pct.toFixed(1)}%</b></div>`;
        const wrap = document.createElement('div');
        wrap.className = 'bar-wrap';
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = Math.min(100, pct) + '%';
        wrap.appendChild(bar); row.appendChild(wrap); barsEl.appendChild(row);
      }
    });

    resultCard.classList.remove('hidden');
  } catch (e) {
    progress.classList.add('hidden');
    alert('Ошибка анализа. Попробуйте другое изображение.');
    console.error(e);
  }
});
```

---

## 3) What’s left for **real** registration/deposit verification
You have two options:
1) **Affiliate callback**: ask your partner manager to send a POST request to your server when a user registers or deposits:
   - URL: `POST {WEBAPP_PUBLIC_URL}/api/affiliate/callback`
   - Body JSON: `{ "uid": "<telegram_id>", "status": "registered"|"deposited", "secret": "<AFFILIATE_CALLBACK_SECRET>" }`
   - We store the status in memory (for demo). For production, switch `_MemDB` to a DB (Redis/Postgres).
2) **Affiliate API polling**: give me `AFFILIATE_API_BASE` and `AFFILIATE_API_KEY` and the endpoint spec, e.g. `/player/status?uid=...` that returns JSON with `registered` / `deposited`. The stub `BookmakerVerifier.poll_status` is ready; I will wire in the exact path/fields once you share the docs.

The bot command **«проверка» / «статус»** shows current flags so вы можете вручную проверить обновление статусов.

---

## 4) Notes & disclaimers
- This app presents **implied probabilities from odds**. It’s **not betting advice**. Add your own compliance notes.
- Host the webapp under HTTPS so Telegram can open it inside clients.
- Replace the placeholder image in `/start` with your artwork (from your screenshots) and extend keyboards/texts as you like.
- For multi‑language texts: add i18n in both bot and webapp; I can wire an i18n JSON next iteration.

---

## 5) Quick checklist
- [ ] Put your **BotFather token** and **OpenAI key** in `.env`.
- [ ] Set `WEBAPP_PUBLIC_URL` to your server domain (https://...).
- [ ] Put your **affiliate ref link** into `AFFILIATE_REG_URL`.
- [ ] (Option A) Give your affiliate platform the **callback URL** + secret.
- [ ] (Option B) Provide API base/key & endpoint contract for polling.
- [ ] Deploy behind HTTPS (Caddy/NGINX/Cloudflare) and keep the same paths.

```
# Procfile (optional, for Heroku-like):
web: python main.py
```
