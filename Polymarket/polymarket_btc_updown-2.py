"""
Polymarket BTC Up/Down (5m) — real-time probability monitor
=============================================================

Continuously streams live "Up"/"Down" implied-probability data for
Polymarket's rolling 5-minute BTC market, and automatically rolls over to
the next 5-minute window the moment the current one closes.

Two things this version fixes vs. a REST-polling approach:

1. LATENCY — instead of polling the REST order book every N seconds (always
   at least one poll-interval + one round-trip behind), this opens a
   WebSocket to Polymarket's CLOB market channel and receives book/price
   updates pushed the moment they happen — the same feed the Polymarket
   website itself renders from.

2. ROLLOVER — "btc-updown-5m-<timestamp>" markets are NOT discovered by
   searching/filtering the market list (unreliable: the list is huge,
   sorting by end-date is dominated by unrelated markets ending at the same
   instant, and a text filter can't tell a 5m window from a 15m one).
   Instead, the slug is deterministic: Polymarket opens a new window every
   time the clock hits a multiple of 300 seconds (5 minutes), named:

       btc-updown-5m-<window_start_unix_timestamp>

   So we compute window_start = now - (now % 300) directly, synced to
   Polymarket's own server clock (GET /time) to avoid local clock drift.
   When a window's close time passes, we recompute the next window_start
   and reconnect automatically — no searching, and it can't get stuck on
   an expired market.

FIXED: prices stuck at ~50%
-----------------------------
The previous version read `bids[0]` / `asks[0]` as the "best" price. In
Polymarket's real order-book payloads, bids are sorted ASCENDING and asks
DESCENDING, so index [0] is actually the WORST price on each side (near the
0.01/0.99 edges) — averaging those two lands close to 0.50 almost no matter
where the real market price is. Fixed by taking max(bids)/min(asks) instead
of trusting array position. A second bug meant `price_change` events (the
most frequent update type) were silently dropped, because their `asset_id`
is nested inside each item of a `price_changes` array, not on the event
itself. Both were found by inspecting Polymarket's actual captured WS
payloads (github.com/nevuamarkets/poly-websockets).

This version also replicates Polymarket's actual displayed-price rule
(confirmed against their docs): the midpoint of best bid/ask, UNLESS that
spread is wider than $0.10, in which case the last traded price is shown
instead. See: https://docs.polymarket.com/polymarket-learn/trading/how-are-prices-calculated

RISK / LEGAL NOTE
------------------
This is a data/monitoring tool, not a trading edge. Polymarket's 5-minute
BTC markets are extremely fast-moving and close to a random walk at that
horizon. Prediction-market availability/legality depends on your
jurisdiction. Not financial advice.
"""

import asyncio
import json
import os
import time
from typing import Dict, Optional, Tuple

import requests
import websockets  # pip install websockets

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

WINDOW_SECONDS = 300       # markets rotate every 5 minutes
INDEX_LAG_RETRIES = 6      # retries in case Gamma hasn't indexed a brand-new window yet
INDEX_LAG_DELAY = 1.0      # seconds between those retries
WATCHDOG_TIMEOUT = 15.0    # force a reconnect if the socket goes silent this long


# --------------------------------------------------------------------------
# Clock sync — Polymarket's server time, not your machine's
# --------------------------------------------------------------------------

_server_offset = 0.0  # server_time - local_time, seconds


def fetch_server_time() -> int:
    """Hits Polymarket's own clock. Falls back to local time if unreachable."""
    try:
        resp = requests.get(f"{CLOB_BASE}/time", timeout=10)
        resp.raise_for_status()
        return int(resp.text.strip())
    except Exception:
        return int(time.time())


def sync_server_time() -> None:
    global _server_offset
    _server_offset = fetch_server_time() - time.time()


def estimated_server_time() -> int:
    """Cheap, non-blocking estimate of current server time from the last sync."""
    return int(time.time() + _server_offset)


# --------------------------------------------------------------------------
# Deterministic market discovery (this is the rollover fix)
# --------------------------------------------------------------------------

def window_start_for(ts: int) -> int:
    return ts - (ts % WINDOW_SECONDS)


def slug_for_window(window_start: int) -> str:
    return f"btc-updown-5m-{window_start}"


def fetch_market_by_slug(slug: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        if not events or not events[0].get("markets"):
            return None
        return events[0]["markets"][0]
    except Exception as e:
        print(f"  [warn] lookup failed for {slug}: {e}")
        return None


def parse_tokens(market: dict) -> Dict[str, str]:
    """Returns {"Up": token_id, "Down": token_id} (or whatever the two outcome names are)."""
    outcomes = json.loads(market.get("outcomes", "[]"))
    token_ids = json.loads(market.get("clobTokenIds", "[]"))
    return dict(zip(outcomes, token_ids))


def get_current_market() -> Tuple[dict, int]:
    """
    Computes the live window from Polymarket's own clock and fetches it by
    its deterministic slug. Retries briefly in case a window just rolled
    over and Gamma hasn't indexed the new market yet.
    """
    sync_server_time()
    window_start = window_start_for(estimated_server_time())

    for _ in range(INDEX_LAG_RETRIES):
        market = fetch_market_by_slug(slug_for_window(window_start))
        if market is not None:
            return market, window_start
        time.sleep(INDEX_LAG_DELAY)
        sync_server_time()
        window_start = window_start_for(estimated_server_time())

    raise RuntimeError(f"No market found for window starting {window_start}")


# --------------------------------------------------------------------------
# Live state tracking
# --------------------------------------------------------------------------

class MarketState:
    """
    Tracks best bid/ask + last trade price per outcome, and derives the same
    "displayed probability" Polymarket's own UI shows: the midpoint of the
    best bid/ask, UNLESS that spread is wider than $0.10 — in which case the
    last traded price is used instead.
    Source: https://docs.polymarket.com/polymarket-learn/trading/how-are-prices-calculated

    Two real-wire-format details this depends on getting right (verified
    against Polymarket's actual WS payloads):
    - `book` events sort bids ASCENDING and asks DESCENDING, so the best
      price on each side is whichever is highest for bids / lowest for
      asks — found with max()/min() rather than assumed by index position.
    - `price_change` events nest `asset_id` INSIDE each item of a
      `price_changes` array, not on the event itself.
    """

    def __init__(self, tokens: Dict[str, str]):
        self.tokens = tokens
        self.name_by_token = {tid: name for name, tid in tokens.items()}
        self.best = {name: {"bid": None, "ask": None, "last_trade": None} for name in tokens}

    def _update_book(self, name: str, bids: list, asks: list) -> None:
        if bids:
            self.best[name]["bid"] = max(float(b["price"]) for b in bids)
        if asks:
            self.best[name]["ask"] = min(float(a["price"]) for a in asks)

    def apply(self, evt: dict) -> bool:
        etype = evt.get("event_type")
        changed = False
        try:
            if etype == "book":
                name = self.name_by_token.get(evt.get("asset_id"))
                if name:
                    self._update_book(name, evt.get("bids") or [], evt.get("asks") or [])
                    changed = True

            elif etype == "price_change":
                # asset_id lives INSIDE each item here, not on the outer event
                for item in evt.get("price_changes") or []:
                    name = self.name_by_token.get(item.get("asset_id"))
                    if not name:
                        continue
                    if item.get("best_bid") is not None:
                        self.best[name]["bid"] = float(item["best_bid"])
                    if item.get("best_ask") is not None:
                        self.best[name]["ask"] = float(item["best_ask"])
                    changed = True

            elif etype == "last_trade_price":
                name = self.name_by_token.get(evt.get("asset_id"))
                if name and evt.get("price") is not None:
                    self.best[name]["last_trade"] = float(evt["price"])
                    changed = True

            elif etype == "best_bid_ask":  # undocumented in practice, harmless if it ever fires
                name = self.name_by_token.get(evt.get("asset_id"))
                if name:
                    if evt.get("best_bid") is not None:
                        self.best[name]["bid"] = float(evt["best_bid"])
                    if evt.get("best_ask") is not None:
                        self.best[name]["ask"] = float(evt["best_ask"])
                    changed = True
        except (TypeError, ValueError, KeyError):
            return False  # unexpected payload shape — skip this update rather than crash
        return changed

    def displayed_price(self, name: str) -> Optional[float]:
        """Replicates Polymarket's own display rule (see docstring above)."""
        bid, ask, last = self.best[name]["bid"], self.best[name]["ask"], self.best[name]["last_trade"]
        if bid is not None and ask is not None and (ask - bid) <= 0.10:
            return (bid + ask) / 2
        if last is not None:
            return last
        if bid is not None and ask is not None:
            return (bid + ask) / 2  # wide spread but no trade yet — best available
        return None

    def line(self, closes_in: float) -> str:
        cells = []
        for name in self.tokens:
            bid, ask, last = self.best[name]["bid"], self.best[name]["ask"], self.best[name]["last_trade"]
            price = self.displayed_price(name)
            if price is None:
                cells.append(f"{name}: waiting for data...")
                continue
            if bid is not None and ask is not None and (ask - bid) <= 0.10:
                detail = f"mid, bid {bid:.2f}/ask {ask:.2f}"
            elif last is not None:
                detail = f"last trade {last:.2f}"
                if bid is not None and ask is not None:
                    detail += f", wide spread {bid:.2f}/{ask:.2f}"
            else:
                detail = f"bid {bid:.2f}/ask {ask:.2f}, wide & no trades yet"
            cells.append(f"{name}: {price * 100:5.1f}%  ({detail})")
        ts = time.strftime("%H:%M:%S")
        return f"[{ts}] closes in {max(closes_in, 0):4.0f}s | " + "   ".join(cells)


# --------------------------------------------------------------------------
# Streaming one window, then handing back control to roll over
# --------------------------------------------------------------------------

async def prime_from_rest(state: MarketState) -> None:
    """Grabs an initial REST snapshot so real numbers show before the first WS book arrives.
    (Brief blocking calls here are fine — this runs once per 5-minute window, not in a hot loop.)"""
    for token_id in state.tokens.values():
        try:
            resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=10)
            resp.raise_for_status()
            book = resp.json()
            book["event_type"] = "book"
            book["asset_id"] = token_id
            state.apply(book)
        except Exception:
            pass  # the WS book snapshot will fill this in moments later anyway


async def stream_window(market: dict, window_start: int) -> None:
    """Streams one 5-minute window in real time; returns when it closes (for rollover)."""
    tokens = parse_tokens(market)
    if len(tokens) < 2:
        raise RuntimeError(f"Expected 2 outcomes for {market.get('slug')}, got {tokens}")

    state = MarketState(tokens)
    close_time = window_start + WINDOW_SECONDS
    print(f"\n=== {market.get('slug')} — streaming live "
          f"(closes {time.strftime('%H:%M:%S', time.localtime(close_time))} local) ===")

    await prime_from_rest(state)

    async with websockets.connect(WS_URL, ping_interval=10, ping_timeout=10) as ws:
        await ws.send(json.dumps({
            "type": "market",
            "assets_ids": list(tokens.values()),
            "custom_feature_enabled": True,  # needed for best_bid_ask events
        }))

        last_msg_at = time.monotonic()

        while True:
            remaining = close_time - estimated_server_time()
            if remaining <= 0:
                print()
                print("Window closed — rolling over to the next market...")
                return

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                if time.monotonic() - last_msg_at > WATCHDOG_TIMEOUT:
                    print("\n  [warn] no data for a while — forcing a reconnect...")
                    return
                print("\r" + state.line(remaining) + "   ", end="", flush=True)
                continue

            last_msg_at = time.monotonic()

            if raw in ("PING", "ping"):
                await ws.send("PONG")
                continue

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            events = parsed if isinstance(parsed, list) else [parsed]
            for evt in events:
                if isinstance(evt, dict):
                    state.apply(evt)

            print("\r" + state.line(remaining) + "   ", end="", flush=True)


# --------------------------------------------------------------------------
# Main loop — runs forever, rolling from one 5-minute window to the next
# --------------------------------------------------------------------------

async def run_forever() -> None:
    backoff = 1.0
    while True:
        try:
            market, window_start = get_current_market()
        except Exception as e:
            print(f"[error] finding current market: {e}; retrying in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

        try:
            await stream_window(market, window_start)
            backoff = 1.0  # reset after a cleanly-completed window
        except Exception as e:
            print(f"\n[warn] stream dropped ({e}); reconnecting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


# --------------------------------------------------------------------------
# Order placement (unchanged capability — manual, not auto-wired into the
# monitor above). Grab a token_id for the *current* window with:
#   market, window_start = get_current_market()
#   tokens = parse_tokens(market)   # {"Up": "...", "Down": "..."}
# --------------------------------------------------------------------------

def place_order(token_id: str, side: str, price: float, size: float,
                 order_type: str = "GTC", dry_run: bool = True):
    """
    Place a limit order on the CLOB.

    Requires environment variables:
      PK              - your wallet's private key
      FUNDER          - address holding your funds (proxy/Magic wallet address)

    dry_run=True (default) only builds and prints the order — it does NOT
    submit it. Pass dry_run=False to actually send it.
    """
    try:
        from py_clob_client_v2 import ClobClient, OrderArgs, OrderType, Side
        from py_clob_client_v2 import PartialCreateOrderOptions
    except ImportError as e:
        raise RuntimeError(
            "pip install py-clob-client-v2   (the old py-clob-client no "
            "longer works for live orders after the CLOB V2 cutover)"
        ) from e

    private_key = os.environ.get("PK")
    if not private_key:
        raise RuntimeError("Set the PK environment variable to your wallet's private key.")

    client = ClobClient(host=CLOB_BASE, chain_id=137, key=private_key)
    creds = client.create_or_derive_api_key()
    client = ClobClient(host=CLOB_BASE, chain_id=137, key=private_key, creds=creds)

    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        side=Side.BUY if side.upper() == "BUY" else Side.SELL,
        size=size,
    )

    if dry_run:
        print("[DRY RUN] Would place order:")
        print(f"  token_id={token_id} side={side} price={price} size={size} type={order_type}")
        print("  Re-run with dry_run=False to actually submit this order.")
        return None

    resp = client.create_and_post_order(
        order_args=order_args,
        options=PartialCreateOrderOptions(tick_size="0.01"),
        order_type=OrderType.GTC if order_type == "GTC" else OrderType.FOK,
    )
    print("Order response:", resp)
    return resp


if __name__ == "__main__":
    print("Polymarket BTC Up/Down (5m) — live monitor. Ctrl+C to stop.")
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("\nStopped.")
