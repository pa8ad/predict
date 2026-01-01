import asyncio
import os
import socket
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import streamlit as st

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 12060
DEFAULT_SPOT_MAX_AGE_MINUTES = 20
DEFAULT_BANDPLAN = {
    "1.8": (1800, 1845),
    "3.5": (3500, 3560),
    "7": (7000, 7060),
    "14": (14000, 14070),
    "21": (21000, 21070),
    "28": (28000, 28070),
    "50": (50000, 50150),
}

# -----------------------------
# Data models
# -----------------------------


def normalized_band(value: str) -> str:
    """Normalize band strings like "3,5" -> "3.5"."""
    return value.replace(",", ".").strip()


@dataclass
class RadioStatus:
    radio_nr: int
    freq: Optional[int] = None
    tx_freq: Optional[int] = None
    mode: str = ""
    is_running: bool = False
    is_transmitting: bool = False
    focus_radio_nr: Optional[int] = None
    active_radio_nr: Optional[int] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Contact:
    contact_id: str
    timestamp: datetime
    band: str
    rx_freq: Optional[int]
    call: str
    prefix: str
    country_prefix: str
    continent: str
    points: int
    is_multiplier: bool
    radio_nr: Optional[int] = None


@dataclass
class Spot:
    dx_call: str
    frequency: float
    timestamp: datetime
    action: str
    mode: str
    comment: str
    status: str
    statuslist: str
    age_minutes: float
    snr_db: Optional[float] = None
    wpm: Optional[int] = None


class ContestState:
    def __init__(self):
        self.radios: Dict[int, RadioStatus] = {
            1: RadioStatus(radio_nr=1),
            2: RadioStatus(radio_nr=2),
        }
        self.contacts: Dict[str, Contact] = {}
        self.prefixes_by_band: Dict[str, set] = defaultdict(set)
        self.total_prefixes: set = set()
        self.qso_timestamps: Deque[datetime] = deque()
        self.spots: List[Spot] = []
        self.event_log: Deque[str] = deque(maxlen=500)
        self.lock = threading.Lock()

    def log_event(self, message: str):
        with self.lock:
            ts = datetime.utcnow().strftime("%H:%M:%S")
            self.event_log.appendleft(f"[{ts}] {message}")

    def update_radio(self, info: Dict[str, str]):
        radio_nr = int(info.get("RadioNr", 0) or 0)
        if radio_nr not in self.radios:
            return
        status = self.radios[radio_nr]
        status.freq = self._safe_int(info.get("Freq"))
        status.tx_freq = self._safe_int(info.get("TXFreq"))
        status.mode = info.get("Mode", "")
        status.is_running = info.get("IsRunning", "False").lower() == "true"
        status.is_transmitting = info.get("IsTransmitting", "False").lower() == "true"
        status.focus_radio_nr = self._safe_int(info.get("FocusRadioNr"))
        status.active_radio_nr = self._safe_int(info.get("ActiveRadioNr"))
        status.last_updated = datetime.utcnow()
        self.log_event(f"Radio {radio_nr} update: {status.freq} Hz CW={status.mode}")

    def add_contact(self, info: Dict[str, str]):
        contact_id = info.get("ID") or info.get("id") or info.get("uniqueid") or str(len(self.contacts) + 1)
        ts = parse_timestamp(info.get("timestamp"))
        band = normalized_band(info.get("band", ""))
        rx_freq = self._safe_int(info.get("rxfreq"))
        call = info.get("call", "").upper()
        prefix = info.get("wpxprefix", "").upper()
        country_prefix = info.get("countryprefix", "")
        continent = info.get("continent", "")
        points = self._safe_int(info.get("points"), 0)
        ismult = str(info.get("ismultiplierl", info.get("ismult1", info.get("ismult")))).lower() in {"1", "true", "yes"}
        radio_nr = self._safe_int(info.get("radionr"))
        contact = Contact(
            contact_id=contact_id,
            timestamp=ts,
            band=band,
            rx_freq=rx_freq,
            call=call,
            prefix=prefix,
            country_prefix=country_prefix,
            continent=continent,
            points=points,
            is_multiplier=ismult,
            radio_nr=radio_nr,
        )
        with self.lock:
            self.contacts[contact_id] = contact
            if prefix:
                self.prefixes_by_band[band].add(prefix)
                self.total_prefixes.add(prefix)
            self.qso_timestamps.append(ts)
        self.log_event(f"QSO logged: {call} on {band} ({prefix})")

    def delete_contact(self, contact_id: str):
        with self.lock:
            contact = self.contacts.pop(contact_id, None)
            if contact:
                self.recalculate_prefixes()
                self.log_event(f"QSO deleted: {contact.call} ({contact.prefix})")

    def replace_contact(self, old_id: str, info: Dict[str, str]):
        self.delete_contact(old_id)
        self.add_contact(info)

    def add_spot(self, info: Dict[str, str], received_at: datetime):
        dx_call = info.get("dxcall", "").upper()
        freq = safe_float(info.get("frequency", 0.0))
        action = info.get("action", "add")
        mode = info.get("mode", "")
        comment = info.get("comment", "")
        status = info.get("status", "")
        statuslist = info.get("statuslist", "")
        ts = parse_timestamp(info.get("timestamp")) or received_at
        snr_db, wpm = parse_comment_metrics(comment)
        age_minutes = max(0.0, (received_at - ts).total_seconds() / 60.0)
        spot = Spot(
            dx_call=dx_call,
            frequency=freq,
            timestamp=ts,
            action=action,
            mode=mode,
            comment=comment,
            status=status,
            statuslist=statuslist,
            age_minutes=age_minutes,
            snr_db=snr_db,
            wpm=wpm,
        )
        with self.lock:
            if action.lower() == "remove":
                self.spots = [s for s in self.spots if not (s.dx_call == dx_call and abs(s.frequency - freq) < 0.01)]
            else:
                self.spots.append(spot)
        self.log_event(f"Spot: {dx_call} {freq} kHz ({comment})")

    def recalculate_prefixes(self):
        self.prefixes_by_band = defaultdict(set)
        self.total_prefixes = set()
        for c in self.contacts.values():
            if c.prefix:
                self.prefixes_by_band[c.band].add(c.prefix)
                self.total_prefixes.add(c.prefix)

    def totals(self):
        with self.lock:
            qsos = len(self.contacts)
            points = sum(c.points for c in self.contacts.values())
            mults = len(self.total_prefixes)
        return qsos, points, mults

    def band_mults(self):
        with self.lock:
            return {band: len(prefixes) for band, prefixes in self.prefixes_by_band.items()}

    def qso_rate(self, minutes: int) -> float:
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        with self.lock:
            while self.qso_timestamps and self.qso_timestamps[0] < cutoff:
                self.qso_timestamps.popleft()
            return len([t for t in self.qso_timestamps if t >= cutoff]) / minutes if minutes > 0 else 0.0

    def get_spots(self) -> List[Spot]:
        cutoff = datetime.utcnow() - timedelta(minutes=DEFAULT_SPOT_MAX_AGE_MINUTES)
        with self.lock:
            self.spots = [s for s in self.spots if s.timestamp >= cutoff]
            return list(self.spots)

    @staticmethod
    def _safe_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
        try:
            return int(str(value)) if value is not None and str(value).strip() != "" else default
        except (TypeError, ValueError):
            return default


# -----------------------------
# Parsing helpers
# -----------------------------


def parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(value.replace("O", "0"), fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def parse_xml(xml_text: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        return None


def parse_to_dict(element: ET.Element) -> Dict[str, str]:
    return {child.tag: child.text or "" for child in element}


def parse_comment_metrics(comment: str) -> Tuple[Optional[float], Optional[int]]:
    snr = None
    wpm = None
    parts = comment.upper().replace("DB", " DB").split()
    for i, token in enumerate(parts):
        if token == "DB" and i > 0:
            try:
                snr = float(parts[i - 1])
            except ValueError:
                pass
        if token.endswith("DB"):
            try:
                snr = float(token[:-2])
            except ValueError:
                pass
        if token.endswith("WPM"):
            try:
                wpm = int(token.replace("WPM", ""))
            except ValueError:
                pass
    return snr, wpm


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# -----------------------------
# Advisor heuristics
# -----------------------------


def score_spot(spot: Spot, state: ContestState, weights: Dict[str, float], max_age_min: int, bandplan: Dict[str, Tuple[int, int]]):
    band = infer_band_from_freq(spot.frequency)
    band_key = normalized_band(band)
    is_new_prefix = spot.dx_call[:3] not in state.total_prefixes  # rough estimate
    is_new_band_prefix = False
    with state.lock:
        for prefix in state.prefixes_by_band.get(band_key, set()):
            if spot.dx_call.startswith(prefix):
                is_new_band_prefix = False
                break
        else:
            is_new_band_prefix = True
    mult_value = weights.get("mult", 10.0) if is_new_prefix else (weights.get("band_mult", 5.0) if is_new_band_prefix else 0.0)
    freshness = max(0.0, 1 - spot.age_minutes / max_age_min)
    freshness_score = freshness * weights.get("fresh", 3.0)
    snr_score = 0.0
    if spot.snr_db is not None:
        snr_score = min(1.0, max(-1.0, spot.snr_db / 10.0)) * weights.get("snr", 2.0)
    band_range = bandplan.get(band_key)
    band_penalty = 0.0
    if band_range:
        low, high = band_range
        freq10 = spot.frequency * 10  # freq in kHz to 10Hz units approximate
        if not (low <= freq10 <= high * 10):
            band_penalty = weights.get("band_penalty", 5.0)
    dupe_penalty = weights.get("dupe_penalty", 4.0) if is_dupe(spot, state, band_key) else 0.0
    score = mult_value + freshness_score + snr_score - band_penalty - dupe_penalty
    return score, band_key, is_new_prefix or is_new_band_prefix


def infer_band_from_freq(freq_khz: float) -> str:
    if freq_khz <= 0:
        return ""
    if freq_khz < 2000:
        return "1.8"
    if freq_khz < 4000:
        return "3.5"
    if freq_khz < 8000:
        return "7"
    if freq_khz < 11000:
        return "10"
    if freq_khz < 15000:
        return "14"
    if freq_khz < 19000:
        return "18"
    if freq_khz < 22000:
        return "21"
    if freq_khz < 25000:
        return "24"
    if freq_khz < 30000:
        return "28"
    return "50"


def is_dupe(spot: Spot, state: ContestState, band: str) -> bool:
    with state.lock:
        for contact in state.contacts.values():
            if contact.call == spot.dx_call and contact.band == band:
                return True
    return False


def best_actions(state: ContestState, weights: Dict[str, float], max_age_min: int, bandplan: Dict[str, Tuple[int, int]], mycall: str) -> List[str]:
    spots = state.get_spots()
    scored = []
    for spot in spots:
        score, band_key, is_mult = score_spot(spot, state, weights, max_age_min, bandplan)
        scored.append((score, spot, band_key, is_mult))
    scored.sort(key=lambda x: x[0], reverse=True)
    actions = []
    for rank, entry in enumerate(scored[:3]):
        score, spot, band_key, is_mult = entry
        if score < 0:
            continue
        why = []
        if is_mult:
            why.append("new prefix")
        if spot.snr_db is not None:
            why.append(f"{spot.snr_db} dB")
        if spot.wpm:
            why.append(f"{spot.wpm} WPM")
        why_str = ", ".join(why) if why else "good spot"
        actions.append(f"{rank+1}. Work {spot.dx_call} on {band_key}m @ {spot.frequency:.1f} kHz ({why_str}, score {score:.1f})")
    if not actions:
        actions.append("Keep running; no high-value spots available.")
    return actions


# -----------------------------
# UDP Listener
# -----------------------------


class UDPListener:
    def __init__(self, host: str, port: int, handler: Callable[[str], None]):
        self.host = host
        self.port = port
        self.handler = handler
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.transport = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=1)

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        listen = self.loop.create_datagram_endpoint(lambda: UDPProtocol(self.handler), local_addr=(self.host, self.port))
        self.transport, _ = self.loop.run_until_complete(listen)
        try:
            self.loop.run_forever()
        finally:
            if self.transport:
                self.transport.close()
            self.loop.close()


class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler: Callable[[str], None]):
        super().__init__()
        self.handler = handler

    def datagram_received(self, data: bytes, addr):
        text = data.decode(errors="ignore")
        self.handler(text)


# -----------------------------
# Streamlit UI helpers
# -----------------------------


def process_packet(text: str, state: ContestState):
    element = parse_xml(text)
    if element is None:
        state.log_event("Malformed XML ignored")
        return
    tag = element.tag.lower()
    info = parse_to_dict(element)
    if tag == "radioinfo":
        state.update_radio(info)
    elif tag in {"contactinfo", "qsoinfo"}:
        state.add_contact(info)
    elif tag == "contactdelete":
        target_id = info.get("ID") or info.get("id") or ""
        if target_id:
            state.delete_contact(target_id)
    elif tag == "contactreplace":
        old_id = info.get("OldID") or info.get("oldid") or info.get("oldId") or ""
        if old_id:
            state.replace_contact(old_id, info)
    elif tag == "spot":
        state.add_spot(info, datetime.utcnow())
    else:
        state.log_event(f"Unhandled packet type: {tag}")


def render_dashboard(state: ContestState, config: Dict[str, any]):
    st.title("AI Contest Assistant - CQ WPX CW")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radio 1")
        render_radio(state.radios[1])
    with col2:
        st.subheader("Radio 2")
        render_radio(state.radios[2])

    qsos, points, mults = state.totals()
    band_mults = state.band_mults()
    rates = {w: state.qso_rate(w) for w in (1, 5, 15)}

    st.subheader("Scoreboard")
    score_cols = st.columns(4)
    score_cols[0].metric("QSOs", qsos)
    score_cols[1].metric("Points", points)
    score_cols[2].metric("Prefixes", mults)
    score_cols[3].metric("1/5/15m rate", f"{rates[1]:.1f}/{rates[5]:.1f}/{rates[15]:.1f}")
    st.write("Band mults", band_mults)

    spots = state.get_spots()
    weights = config.get("weights", {})
    bandplan = config.get("bandplan", DEFAULT_BANDPLAN)
    max_age = config.get("max_age", DEFAULT_SPOT_MAX_AGE_MINUTES)
    sorted_spots = []
    for spot in spots:
        score, band_key, is_mult = score_spot(spot, state, weights, max_age, bandplan)
        sorted_spots.append((score, spot, band_key, is_mult))
    sorted_spots.sort(key=lambda x: x[0], reverse=True)

    st.subheader("High value spots")
    rows = []
    for score, spot, band_key, is_mult in sorted_spots[:20]:
        rows.append({
            "dxcall": spot.dx_call,
            "freq": f"{spot.frequency:.1f}",
            "band": band_key,
            "age": f"{spot.age_minutes:.1f}m",
            "SNR/WPM": f"{spot.snr_db or ''} {spot.wpm or ''}",
            "status": spot.status,
            "why": "new mult" if is_mult else "",
            "score": f"{score:.1f}",
        })
    st.dataframe(rows, use_container_width=True)

    st.subheader("Next best action")
    actions = best_actions(state, weights, max_age, bandplan, config.get("mycall", ""))
    st.write("\n".join(actions))

    st.subheader("Event log")
    log_lines = list(state.event_log)
    max_lines = st.slider("Max log lines", 10, 200, 50)
    st.text("\n".join(log_lines[:max_lines]))


def render_radio(radio: RadioStatus):
    st.write(f"Freq: {radio.freq} Hz")
    st.write(f"TX Freq: {radio.tx_freq} Hz")
    st.write(f"Mode: {radio.mode}")
    st.write(f"Running: {radio.is_running}")
    st.write(f"Transmitting: {radio.is_transmitting}")
    st.write(f"Focus/Active: {radio.focus_radio_nr}/{radio.active_radio_nr}")
    st.write(f"Updated: {radio.last_updated.strftime('%H:%M:%S')}")


# -----------------------------
# Streamlit entry point
# -----------------------------


def init_state():
    if "contest_state" not in st.session_state:
        st.session_state.contest_state = ContestState()
    if "listener" not in st.session_state:
        st.session_state.listener = None


def main():
    st.set_page_config(page_title="AI Contest Assistant", layout="wide")
    init_state()

    st.sidebar.header("Configuration")
    host = st.sidebar.text_input("Listen host", value=DEFAULT_HOST)
    port = st.sidebar.number_input("Listen port", value=DEFAULT_PORT, step=1)
    mycall = st.sidebar.text_input("My call", value="N0CALL")
    max_age = st.sidebar.number_input("Max spot age (min)", value=DEFAULT_SPOT_MAX_AGE_MINUTES, step=1)

    weights = {
        "mult": st.sidebar.number_input("Weight: new mult", value=10.0),
        "band_mult": st.sidebar.number_input("Weight: new band mult", value=5.0),
        "fresh": st.sidebar.number_input("Weight: freshness", value=3.0),
        "snr": st.sidebar.number_input("Weight: SNR", value=2.0),
        "band_penalty": st.sidebar.number_input("Penalty: out of bandplan", value=5.0),
        "dupe_penalty": st.sidebar.number_input("Penalty: dupe", value=4.0),
    }

    config = {"weights": weights, "bandplan": DEFAULT_BANDPLAN, "max_age": max_age, "mycall": mycall}

    listener_running = st.session_state.listener is not None
    if st.sidebar.button("Start UDP listener", disabled=listener_running):
        st.session_state.listener = UDPListener(host, int(port), lambda t: process_packet(t, st.session_state.contest_state))
        st.session_state.listener.start()
        st.session_state.contest_state.log_event(f"UDP listener started on {host}:{port}")

    if st.sidebar.button("Stop UDP listener", disabled=not listener_running):
        if st.session_state.listener:
            st.session_state.listener.stop()
            st.session_state.listener = None
            st.session_state.contest_state.log_event("UDP listener stopped")

    render_dashboard(st.session_state.contest_state, config)


if __name__ == "__main__":
    main()
