import argparse
import asyncio
import re
import socket
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from PySide6 import QtCore, QtGui, QtWidgets

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
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


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
            ts = datetime.now(UTC).strftime("%H:%M:%S")
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
        status.last_updated = datetime.now(UTC)
        self.log_event(f"Radio {radio_nr} update: {status.freq} Hz CW={status.mode}")

    def add_contact(self, info: Dict[str, str]):
        contact_id = (
            info.get("ID")
            or info.get("id")
            or info.get("uniqueid")
            or str(len(self.contacts) + 1)
        )
        ts = parse_timestamp(info.get("timestamp"))
        band = normalized_band(info.get("band", ""))
        rx_freq = self._safe_int(info.get("rxfreq"))
        call = info.get("call", "").upper()
        prefix = info.get("wpxprefix", "").upper()
        country_prefix = info.get("countryprefix", "")
        continent = info.get("continent", "")
        points = self._safe_int(info.get("points"), 0)
        ismult = str(
            info.get("ismultiplierl", info.get("ismult1", info.get("ismult")))
        ).lower() in {"1", "true", "yes"}
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
        if freq > 100000:  # handle Hz values by converting to kHz
            freq = freq / 1000.0
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
                self.spots = [
                    s
                    for s in self.spots
                    if not (s.dx_call == dx_call and abs(s.frequency - freq) < 0.01)
                ]
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
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
        with self.lock:
            while self.qso_timestamps and self.qso_timestamps[0] < cutoff:
                self.qso_timestamps.popleft()
            return (
                len([t for t in self.qso_timestamps if t >= cutoff]) / minutes
                if minutes > 0
                else 0.0
            )

    def get_spots(self) -> List[Spot]:
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=DEFAULT_SPOT_MAX_AGE_MINUTES)
        with self.lock:
            fresh_spots: List[Spot] = []
            for spot in self.spots:
                spot.age_minutes = max(0.0, (now - spot.timestamp).total_seconds() / 60.0)
                if spot.timestamp >= cutoff:
                    fresh_spots.append(spot)
            self.spots = fresh_spots
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
        return datetime.now(UTC)
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(value.replace("O", "0"), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return datetime.now(UTC)


def parse_xml(xml_text: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        return None


def parse_to_dict(element: ET.Element) -> Dict[str, str]:
    return {child.tag: child.text or "" for child in element}


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


# -----------------------------
# Advisor heuristics
# -----------------------------


def score_spot(
    spot: Spot,
    state: ContestState,
    weights: Dict[str, float],
    max_age_min: int,
    bandplan: Dict[str, Tuple[int, int]],
):
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
    mult_value = (
        weights.get("mult", 10.0)
        if is_new_prefix
        else (weights.get("band_mult", 5.0) if is_new_band_prefix else 0.0)
    )
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


def best_actions(
    state: ContestState,
    weights: Dict[str, float],
    max_age_min: int,
    bandplan: Dict[str, Tuple[int, int]],
    mycall: str,
) -> List[str]:
    del mycall  # placeholder for future LLM integration
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
        actions.append(
            f"{rank+1}. Work {spot.dx_call} on {band_key}m @ {spot.frequency:.1f} kHz ({why_str}, score {score:.1f})"
        )
    if not actions:
        actions.append("Keep running; no high-value spots available.")
    return actions


def parse_comment_metrics(comment: str) -> Tuple[Optional[float], Optional[int]]:
    snr = None
    wpm = None
    text = comment.upper().replace(",", " ")
    db_match = re.search(r"(-?\d+(?:\.\d+)?)\s*DB", text)
    if db_match:
        try:
            snr = float(db_match.group(1))
        except ValueError:
            snr = None
    wpm_match = re.search(r"(\d{2,3})\s*WPM", text)
    if wpm_match:
        try:
            wpm = int(wpm_match.group(1))
        except ValueError:
            wpm = None
    return snr, wpm


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
        listen = self.loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.handler), local_addr=(self.host, self.port)
        )
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
        del addr
        text = data.decode(errors="ignore")
        self.handler(text)


# -----------------------------
# Packet processing
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
        state.add_spot(info, datetime.now(UTC))
    else:
        state.log_event(f"Unhandled packet type: {tag}")


# -----------------------------
# Qt UI
# -----------------------------


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, state: ContestState, config: Dict[str, any]):
        super().__init__()
        self.state = state
        self.config = config
        self.listener: Optional[UDPListener] = None
        self.setWindowTitle("AI Contest Assistant - CQ WPX CW")
        self.resize(1100, 750)
        self._build_ui()
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_views)
        self.refresh_timer.start(1000)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        self.host_input = QtWidgets.QLineEdit(self.config.get("host", DEFAULT_HOST))
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setMaximum(65535)
        self.port_input.setValue(int(self.config.get("port", DEFAULT_PORT)))
        self.call_input = QtWidgets.QLineEdit(self.config.get("mycall", "N0CALL"))
        self.max_age_input = QtWidgets.QSpinBox()
        self.max_age_input.setRange(1, 120)
        self.max_age_input.setValue(int(self.config.get("max_age", DEFAULT_SPOT_MAX_AGE_MINUTES)))
        self.contest_select = QtWidgets.QComboBox()
        self.contest_select.addItems(["CQ WPX CW"])
        self.start_button = QtWidgets.QPushButton("Start listener")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)

        controls.addWidget(QtWidgets.QLabel("Host"))
        controls.addWidget(self.host_input)
        controls.addWidget(QtWidgets.QLabel("Port"))
        controls.addWidget(self.port_input)
        controls.addWidget(QtWidgets.QLabel("My call"))
        controls.addWidget(self.call_input)
        controls.addWidget(QtWidgets.QLabel("Contest"))
        controls.addWidget(self.contest_select)
        controls.addWidget(QtWidgets.QLabel("Max spot age (min)"))
        controls.addWidget(self.max_age_input)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        self.start_button.clicked.connect(self.start_listener)
        self.stop_button.clicked.connect(self.stop_listener)

        # Radio panels
        radio_layout = QtWidgets.QHBoxLayout()
        self.radio1_widget = self._radio_panel("Radio 1")
        self.radio2_widget = self._radio_panel("Radio 2")
        radio_layout.addWidget(self.radio1_widget)
        radio_layout.addWidget(self.radio2_widget)
        layout.addLayout(radio_layout)

        # Scoreboard and actions
        self.score_group = QtWidgets.QGroupBox("Scoreboard")
        score_layout = QtWidgets.QGridLayout()
        self.score_group.setLayout(score_layout)
        self.qso_label = QtWidgets.QLabel("QSOs: 0")
        self.points_label = QtWidgets.QLabel("Points: 0")
        self.mults_label = QtWidgets.QLabel("Prefixes: 0")
        self.rate_label = QtWidgets.QLabel("Rates 1/5/15m: 0/0/0")
        self.band_mults_label = QtWidgets.QLabel("Band mults: {}")
        score_layout.addWidget(self.qso_label, 0, 0)
        score_layout.addWidget(self.points_label, 0, 1)
        score_layout.addWidget(self.mults_label, 0, 2)
        score_layout.addWidget(self.rate_label, 1, 0, 1, 2)
        score_layout.addWidget(self.band_mults_label, 1, 2)

        self.actions_group = QtWidgets.QGroupBox("Next best actions")
        actions_layout = QtWidgets.QVBoxLayout()
        self.actions_group.setLayout(actions_layout)
        self.actions_text = QtWidgets.QTextEdit()
        self.actions_text.setReadOnly(True)
        self.actions_text.setFixedHeight(90)
        actions_layout.addWidget(self.actions_text)

        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addWidget(self.score_group)
        stats_layout.addWidget(self.actions_group)
        layout.addLayout(stats_layout)

        # Contest rules and weights
        rules_group = QtWidgets.QGroupBox("Rules & Heuristics")
        rules_layout = QtWidgets.QFormLayout()
        rules_group.setLayout(rules_layout)
        self.weight_inputs: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, default in default_weights().items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-20.0, 50.0)
            spin.setSingleStep(0.5)
            spin.setValue(float(self.config.get("weights", {}).get(key, default)))
            self.weight_inputs[key] = spin
            rules_layout.addRow(f"{key} weight", spin)
        self.rules_hint = QtWidgets.QLabel(
            "CQ WPX CW: Prefix multipliers per band, heuristic scoring for spots."
        )
        rules_layout.addRow(self.rules_hint)
        layout.addWidget(rules_group)

        # Spots table
        self.spot_table = QtWidgets.QTableWidget()
        self.spot_table.setColumnCount(8)
        self.spot_table.setHorizontalHeaderLabels(
            ["Call", "Freq", "Band", "Age", "SNR", "WPM", "Status", "Score"]
        )
        self.spot_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(QtWidgets.QLabel("High value spots"))
        layout.addWidget(self.spot_table)

        # Event log
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        layout.addWidget(QtWidgets.QLabel("Event log"))
        layout.addWidget(self.log_view)

        self.setCentralWidget(central)

    def _radio_panel(self, title: str) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QFormLayout()
        group.setLayout(layout)
        labels = {
            "freq": QtWidgets.QLabel("-"),
            "tx_freq": QtWidgets.QLabel("-"),
            "mode": QtWidgets.QLabel("-"),
            "running": QtWidgets.QLabel("-"),
            "transmitting": QtWidgets.QLabel("-"),
            "focus": QtWidgets.QLabel("-"),
            "active": QtWidgets.QLabel("-"),
            "updated": QtWidgets.QLabel("-"),
        }
        layout.addRow("Freq", labels["freq"])
        layout.addRow("TX Freq", labels["tx_freq"])
        layout.addRow("Mode", labels["mode"])
        layout.addRow("Running", labels["running"])
        layout.addRow("Transmitting", labels["transmitting"])
        layout.addRow("Focus", labels["focus"])
        layout.addRow("Active", labels["active"])
        layout.addRow("Updated", labels["updated"])
        group.labels = labels  # type: ignore[attr-defined]
        return group

    def refresh_views(self):
        qsos, points, mults = self.state.totals()
        band_mults = self.state.band_mults()
        rates = {w: self.state.qso_rate(w) for w in (1, 5, 15)}
        self.qso_label.setText(f"QSOs: {qsos}")
        self.points_label.setText(f"Points: {points}")
        self.mults_label.setText(f"Prefixes: {mults}")
        self.rate_label.setText(
            f"Rates 1/5/15m: {rates[1]:.1f}/{rates[5]:.1f}/{rates[15]:.1f}"
        )
        self.band_mults_label.setText(f"Band mults: {band_mults}")

        self._update_radio_panel(self.radio1_widget, self.state.radios[1])
        self._update_radio_panel(self.radio2_widget, self.state.radios[2])

        # Spots
        weights = self._current_weights()
        bandplan = self.config.get("bandplan", DEFAULT_BANDPLAN)
        max_age = int(self.max_age_input.value())
        spots = self.state.get_spots()
        rows: List[Tuple[float, Spot, str, bool]] = []
        for spot in spots:
            score, band_key, is_mult = score_spot(spot, self.state, weights, max_age, bandplan)
            rows.append((score, spot, band_key, is_mult))
        rows.sort(key=lambda x: x[0], reverse=True)
        self.spot_table.setRowCount(min(len(rows), 30))
        for row_idx, (score, spot, band_key, is_mult) in enumerate(rows[:30]):
            values = [
                spot.dx_call,
                f"{spot.frequency:.1f}",
                band_key,
                f"{spot.age_minutes:.1f}m",
                str(spot.snr_db or ""),
                str(spot.wpm or ""),
                "new mult" if is_mult else spot.status,
                f"{score:.1f}",
            ]
            for col, val in enumerate(values):
                item = QtWidgets.QTableWidgetItem(val)
                self.spot_table.setItem(row_idx, col, item)

        # Actions
        actions = best_actions(self.state, weights, max_age, bandplan, self.call_input.text())
        self.actions_text.setPlainText("\n".join(actions))

        # Event log
        log_lines = list(self.state.event_log)[:80]
        self.log_view.setPlainText("\n".join(log_lines))

    def _update_radio_panel(self, panel: QtWidgets.QGroupBox, radio: RadioStatus):
        labels = panel.labels  # type: ignore[attr-defined]
        labels["freq"].setText(f"{radio.freq or '-'} Hz")
        labels["tx_freq"].setText(f"{radio.tx_freq or '-'} Hz")
        labels["mode"].setText(radio.mode or "-")
        labels["running"].setText("Yes" if radio.is_running else "No")
        labels["transmitting"].setText("Yes" if radio.is_transmitting else "No")
        labels["focus"].setText(str(radio.focus_radio_nr or "-"))
        labels["active"].setText(str(radio.active_radio_nr or "-"))
        labels["updated"].setText(radio.last_updated.strftime("%H:%M:%S"))

    def start_listener(self):
        host = self.host_input.text() or DEFAULT_HOST
        port = int(self.port_input.value())
        self.config["host"] = host
        self.config["port"] = port
        self.config["mycall"] = self.call_input.text()
        self.config["max_age"] = int(self.max_age_input.value())
        self.config["contest"] = self.contest_select.currentText()
        self.config["weights"] = self._current_weights()
        if self.listener:
            return

        try:
            socket.getaddrinfo(host, port)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid address", str(exc))
            return

        self.listener = UDPListener(host, port, lambda t: process_packet(t, self.state))
        self.listener.start()
        self.state.log_event(f"UDP listener started on {host}:{port}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
            self.state.log_event("UDP listener stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _current_weights(self) -> Dict[str, float]:
        if not hasattr(self, "weight_inputs"):
            return self.config.get("weights", default_weights())
        weights = {}
        for key, spin in self.weight_inputs.items():
            weights[key] = float(spin.value())
        return weights

    def closeEvent(self, event: QtGui.QCloseEvent):  # noqa: N802
        self.stop_listener()
        super().closeEvent(event)


def default_weights() -> Dict[str, float]:
    return {
        "mult": 10.0,
        "band_mult": 5.0,
        "fresh": 3.0,
        "snr": 2.0,
        "band_penalty": 5.0,
        "dupe_penalty": 4.0,
    }


# -----------------------------
# Console mode
# -----------------------------


def run_console_listener(host: str, port: int, mycall: str, max_age: int):
    state = ContestState()
    weights = default_weights()
    listener = UDPListener(host, port, lambda t: process_packet(t, state))
    listener.start()
    state.log_event(f"Console UDP listener started on {host}:{port}")

    try:
        while True:
            qsos, points, mults = state.totals()
            rates = {w: state.qso_rate(w) for w in (1, 5, 15)}
            spots = state.get_spots()
            ranked = []
            for spot in spots:
                score, band_key, is_mult = score_spot(
                    spot, state, weights, max_age, DEFAULT_BANDPLAN
                )
                ranked.append((score, spot, band_key, is_mult))
            ranked.sort(key=lambda x: x[0], reverse=True)

            print("--- Contest snapshot ---")
            print(f"QSOs: {qsos}  Points: {points}  Prefixes: {mults}")
            print(
                f"Rates 1/5/15m: {rates[1]:.1f}/{rates[5]:.1f}/{rates[15]:.1f}"
            )
            print("Top spots:")
            for score, spot, band_key, is_mult in ranked[:5]:
                why = "new mult" if is_mult else ""
                print(
                    f"  {spot.dx_call} {spot.frequency:.1f}kHz {band_key} age {spot.age_minutes:.1f}m "
                    f"SNR {spot.snr_db or '-'} WPM {spot.wpm or '-'} score {score:.1f} {why}"
                )
            print("Event log (latest 5):")
            for line in list(state.event_log)[:5]:
                print("  ", line)
            print("-----------------------\n")
            threading.Event().wait(5)
    except KeyboardInterrupt:
        print("Stopping listener...")
    finally:
        listener.stop()


# -----------------------------
# Entry point
# -----------------------------


def run_gui(host: str, port: int, mycall: str, max_age: int):
    app = QtWidgets.QApplication(sys.argv)
    config = {
        "host": host,
        "port": port,
        "mycall": mycall,
        "max_age": max_age,
        "weights": default_weights(),
        "bandplan": DEFAULT_BANDPLAN,
    }
    state = ContestState()
    window = MainWindow(state, config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Contest Assistant")
    parser.add_argument("mode", choices=["gui", "console"], nargs="?", default="gui")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-age", type=int, default=DEFAULT_SPOT_MAX_AGE_MINUTES)
    parser.add_argument("--mycall", default="N0CALL")
    args = parser.parse_args()

    if args.mode == "console":
        run_console_listener(args.host, args.port, args.mycall, args.max_age)
    else:
        run_gui(args.host, args.port, args.mycall, args.max_age)
