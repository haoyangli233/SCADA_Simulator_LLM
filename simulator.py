#!/usr/bin/env python3
import csv
import math
import os
import random
import time
from typing import List, Optional, Tuple, Union
import requests

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Device model
# ---------------------------------------------------------------------------

class Device:
    def __init__(self, name: str, device_type: str, normal_reading: Optional[float] = None):
        """
        device_type:
          'dummy', 'wind_speed', 'wind_dir', 'inside_temp', 'outside_temp'
        """
        assert device_type in (
            "dummy",
            "wind_speed",
            "wind_dir",
            "inside_temp",
            "outside_temp",
        )
        self.name = name
        self.device_type = device_type
        self.normal_reading = normal_reading

        self.signal_strength: float = 1.0  # 0.0 - 1.0
        self.hacked: bool = False

        # Value column in UI
        self.value: Union[float, str, None] = None

        # LLM decision shown in "LLM Des" column (last decision)
        self.llm_decision: Optional[str] = None

        # History of recent LLM decisions and final aggregated label
        self.llm_decision_history: List[str] = []
        self.llm_final: Optional[str] = None

        # Optional comment from secondary LLM when llm_final == "FAULT"
        self.comment: Optional[str] = None

        # History: list of (timestamp, value) for the last seconds
        self.history: List[Tuple[float, Union[float, str, None]]] = []

    # ------------------------------------------------------------------
    # Fault status rule (local rule-based logic)
    # ------------------------------------------------------------------

    def fault_status(self) -> str:
        """
        GOOD / ATTENTION / FAULT based on signal_strength and hacked flag.

        - 0.3 <= signal <= 0.8  -> ATTENTION
        - signal < 0.3 or hacked -> FAULT
        - otherwise             -> GOOD
        """
        s = max(0.0, min(1.0, self.signal_strength))
        if self.hacked or s < 0.3:
            return "FAULT"
        elif 0.3 <= s <= 0.8:
            return "ATTENTION"
        else:
            return "GOOD"

    # ------------------------------------------------------------------
    # LLM decision aggregation
    # ------------------------------------------------------------------

    def update_llm_decision(self, decision: str, window: int = 6) -> None:
        """
        Update the latest LLM decision and aggregate the last `window`
        decisions into a final majority label.

        LLM Final is only available once we have at least `window` decisions.
        Tie-break is conservative: FAULT > ATTENTION > GOOD.
        """
        self.llm_decision = decision

        if decision not in ("GOOD", "ATTENTION", "FAULT"):
            return

        # Append and keep only last `window` decisions
        self.llm_decision_history.append(decision)
        if len(self.llm_decision_history) > window:
            self.llm_decision_history = self.llm_decision_history[-window:]

        # If we have fewer than `window` decisions, no LLM Final yet
        if len(self.llm_decision_history) < window:
            self.llm_final = None
            return

        # Now we have exactly `window` most recent decisions (previous 6)
        counts = {"GOOD": 0, "ATTENTION": 0, "FAULT": 0}
        for lab in self.llm_decision_history:
            if lab in counts:
                counts[lab] += 1

        max_count = max(counts.values())
        # Conservative tie-break: FAULT > ATTENTION > GOOD
        for lab in ("FAULT", "ATTENTION", "GOOD"):
            if counts[lab] == max_count:
                self.llm_final = lab
                break


    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _record_history(self):
        """Record current value with timestamp, keeping only the last ~5 seconds."""
        now = time.time()
        self.history.append((now, self.value))
        cutoff = now - 5.0
        # Drop old entries
        while self.history and self.history[0][0] < cutoff:
            self.history.pop(0)

    def recent_values(self, window_sec: float = 5.0) -> List[Union[float, str, None]]:
        """Return values within the last window_sec seconds (oldest -> newest)."""
        now = time.time()
        vals: List[Union[float, str, None]] = []
        for ts, v in self.history:
            if ts >= now - window_sec:
                if isinstance(v, float):
                    vals.append(round(v, 4))
                else:
                    vals.append(v)
        return vals

    # ------------------------------------------------------------------
    # Reading update for dummy devices
    # ------------------------------------------------------------------

    def _update_dummy_reading(self) -> None:
        """Generate a dummy numeric value, degraded by signal and hacked."""
        s = max(0.0, min(1.0, self.signal_strength))
        base = self.normal_reading if self.normal_reading is not None else 50.0

        # Hacked dummy: very unstable, jumping between too-low and too-high.
        if self.hacked:
            t = time.time()
            # Alternate every second between ~0.1x and ~3x of normal with noise
            if int(t) % 2 == 0:
                # much higher than normal
                self.value = base * 3.0 + random.gauss(0.0, base * 0.2)
            else:
                # much lower than normal (but not exactly zero)
                self.value = base * 0.1 + random.gauss(0.0, base * 0.05)
            return

        # Signal completely gone -> NA
        if s == 0.0:
            self.value = "NA"
            return

        # Noise & dropout pattern depending on signal strength
        if s >= 0.8:
            # Good signal: low noise, no NA
            p_na = 0.0
            noise_mult = 0.05  # ±5% noise
        elif s >= 0.3:
            # Weak signal: some NA, more noise
            p_na = 0.25
            noise_mult = 0.15  # ±15%
        else:
            # Very weak (already FAULT): frequent NA, large noise
            p_na = 0.6
            noise_mult = 0.3  # ±30%

        r = random.random()
        if r < p_na:
            self.value = "NA"
            return

        noise = random.gauss(0.0, noise_mult * base)
        self.value = base + noise

    # ------------------------------------------------------------------
    # Reading update for wind devices
    # ------------------------------------------------------------------

    def _update_wind_reading(
        self,
        base_speed: float,
        base_direction: float,
        ws_std: float,
        wd_std: float,
    ) -> None:
        """
        Use a real SCADA wind sample (base_speed, base_direction) and degrade it
        depending on signal strength and hacked flag.

        ws_std, wd_std: std-dev of speed/direction in the dataset.
        """
        s = max(0.0, min(1.0, self.signal_strength))

        # Hacked wind device: clearly abnormal behaviour.
        if self.hacked:
            t = time.time()
            if self.device_type == "wind_speed":
                mode = int(t) % 3
                if mode == 0:
                    # Occasionally negative or huge values
                    self.value = random.choice([
                        -abs(base_speed + ws_std),                       # negative (impossible)
                        base_speed + 40.0 + 10.0 * random.random(),     # very high
                    ])
                elif mode == 1:
                    # Violent oscillation around a high speed
                    self.value = base_speed + 25.0 * math.sin(t)
                else:
                    # Sudden spikes
                    self.value = base_speed + random.choice([30.0, -10.0])
            else:  # wind_dir
                # Direction jumping erratically, sometimes out of range
                mode = int(t) % 3
                if mode == 0:
                    self.value = base_direction + random.choice([-400.0, 400.0])  # outside [0, 360]
                elif mode == 1:
                    self.value = (base_direction + 180.0 * math.sin(t * 2.0))
                else:
                    self.value = base_direction + random.choice([720.0, -90.0])
            return

        # Signal completely gone -> NA
        if s == 0.0:
            self.value = "NA"
            return

        # Bands: good / weak / very weak (faulty)
        if s >= 0.8:
            # Good signal
            p_na = 0.0
            p_big_error = 0.01
            noise_mult = 0.1
        elif s >= 0.3:
            # Weak signal
            p_na = 0.3
            p_big_error = 0.02
            noise_mult = 0.3
        else:
            # Very weak (FAULT)
            p_na = 0.7
            p_big_error = 0.05
            noise_mult = 0.5

        r = random.random()

        # Dropout
        if r < p_na:
            self.value = "NA"
            return

        # Non-hacked anomaly: out-of-range spikes
        if r < p_na + p_big_error:
            if self.device_type == "wind_speed":
                self.value = max(
                    0.0,
                    base_speed
                    + random.choice([-1.0, 1.0]) * (2.0 * ws_std + 5.0),
                )
            else:  # wind_dir
                self.value = (
                    base_direction
                    + random.choice([-1.0, 1.0]) * (3.0 * wd_std)
                ) % 360.0
            return

        # Normal-ish reading with Gaussian noise
        if self.device_type == "wind_speed":
            self.value = max(
                0.0,
                random.gauss(base_speed, noise_mult * ws_std),
            )
        else:  # wind_dir
            wd = random.gauss(base_direction, noise_mult * wd_std)
            self.value = wd % 360.0

    # ------------------------------------------------------------------
    # Reading update for temperature devices
    # ------------------------------------------------------------------

    def _update_temp_reading(self, base_temp: float, t_std: float) -> None:
        """
        Use real temperature sample and degrade it depending on signal strength
        and hacked flag.
        """
        s = max(0.0, min(1.0, self.signal_strength))

        # Hacked temp device: obvious false data injection.
        if self.hacked:
            t = time.time()
            # Alternate between very cold and very hot every second
            if int(t) % 2 == 0:
                # unrealistically low for inside, or extreme for outside
                self.value = base_temp - random.choice([20.0, 30.0])
            else:
                # unrealistically high
                self.value = base_temp + random.choice([20.0, 30.0])
            return

        # Signal completely gone -> NA
        if s == 0.0:
            self.value = "NA"
            return

        # Bands: good / weak / very weak (faulty)
        if s >= 0.8:
            p_na = 0.0
            p_big_error = 0.01
            noise_mult = 0.1
        elif s >= 0.3:
            p_na = 0.3
            p_big_error = 0.02
            noise_mult = 0.3
        else:
            p_na = 0.7
            p_big_error = 0.05
            noise_mult = 0.5

        r = random.random()

        # Dropout
        if r < p_na:
            self.value = "NA"
            return

        # Non-hacked anomaly: occasional unrealistic spike
        if r < p_na + p_big_error:
            self.value = base_temp + random.choice([-1.0, 1.0]) * (3.0 * t_std + 5.0)
            return

        # Normal-ish reading with Gaussian noise
        self.value = random.gauss(base_temp, noise_mult * t_std)

    # ------------------------------------------------------------------
    # Public update entry
    # ------------------------------------------------------------------

    def update(
        self,
        wind_sample: Optional[Tuple[float, float]],
        ws_std: float,
        wd_std: float,
        temp_sample: Optional[Tuple[float, float]],
        tin_std: float,
        tout_std: float,
    ) -> None:
        """Update this device's readings and history."""
        if self.device_type == "dummy":
            self._update_dummy_reading()
        else:
            # Unpack / fallback for wind
            if wind_sample is None:
                base_speed = random.uniform(0.0, 25.0)
                base_dir = random.uniform(0.0, 360.0)
            else:
                base_speed, base_dir = wind_sample

            # Unpack / fallback for temp
            if temp_sample is None:
                temp_inside = random.uniform(18.0, 24.0)
                temp_outside = random.uniform(0.0, 30.0)
            else:
                temp_inside, temp_outside = temp_sample

            if self.device_type in ("wind_speed", "wind_dir"):
                self._update_wind_reading(base_speed, base_dir, ws_std, wd_std)
            elif self.device_type == "inside_temp":
                self._update_temp_reading(temp_inside, tin_std)
            elif self.device_type == "outside_temp":
                self._update_temp_reading(temp_outside, tout_std)

        # Record history after generating value
        self._record_history()


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SCADASimulator:
    """Manages devices and provides wind & temperature samples."""

    def __init__(self):
        self.devices: List[Device] = []

        # Wind data
        self.wind_samples: List[Tuple[float, float]] = []
        self.ws_std: float = 4.0   # defaults if CSV not found
        self.wd_std: float = 90.0
        self._wind_index: int = 0

        # Temperature data
        self.temp_samples: List[Tuple[float, float]] = []
        self.tin_std: float = 1.0
        self.tout_std: float = 1.0
        self._temp_index: int = 0

        self._load_wind_data()
        self._load_temp_data()

    # ---------------- Wind data loading ----------------

    def _load_wind_data(self):
        csv_path = os.path.join(os.path.dirname(__file__), "T1.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: T1.csv not found at {csv_path}. Using synthetic wind data.")
            return

        try:
            speeds: List[float] = []
            dirs: List[float] = []
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ws = float(row["Wind Speed (m/s)"])
                        wd = float(row["Wind Direction (°)"])
                    except Exception:
                        continue
                    speeds.append(ws)
                    dirs.append(wd)
                    self.wind_samples.append((ws, wd))

            if speeds:
                ws_mean = sum(speeds) / len(speeds)
                self.ws_std = math.sqrt(
                    sum((x - ws_mean) ** 2 for x in speeds) / len(speeds)
                )
            if dirs:
                wd_mean = sum(dirs) / len(dirs)
                self.wd_std = math.sqrt(
                    sum((x - wd_mean) ** 2 for x in dirs) / len(dirs)
                )
            print(
                f"Loaded {len(self.wind_samples)} wind samples, "
                f"ws_std={self.ws_std:.2f}, wd_std={self.wd_std:.2f}"
            )
        except Exception as e:
            print(f"Warning: failed to load T1.csv: {e}")
            self.wind_samples = []

    def _next_wind_sample(self) -> Optional[Tuple[float, float]]:
        if not self.wind_samples:
            return None
        sample = self.wind_samples[self._wind_index]
        self._wind_index = (self._wind_index + 1) % len(self.wind_samples)
        return sample

    # ---------------- Temperature data loading ----------------

    def _load_temp_data(self):
        csv_path = os.path.join(os.path.dirname(__file__), "temp2.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: temp2.csv not found at {csv_path}. Using synthetic temperature data.")
            return

        try:
            tins: List[float] = []
            touts: List[float] = []
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tin = float(row["tInside"])
                        tout = float(row["tOutside"])
                    except Exception:
                        continue
                    tins.append(tin)
                    touts.append(tout)
                    self.temp_samples.append((tin, tout))

            if tins:
                tin_mean = sum(tins) / len(tins)
                self.tin_std = math.sqrt(
                    sum((x - tin_mean) ** 2 for x in tins) / len(tins)
                )
            if touts:
                tout_mean = sum(touts) / len(touts)
                self.tout_std = math.sqrt(
                    sum((x - tout_mean) ** 2 for x in touts) / len(touts)
                )
            print(
                f"Loaded {len(self.temp_samples)} temperature samples, "
                f"tin_std={self.tin_std:.2f}, tout_std={self.tout_std:.2f}"
            )
        except Exception as e:
            print(f"Warning: failed to load temp2.csv: {e}")
            self.temp_samples = []

    def _next_temp_sample(self) -> Optional[Tuple[float, float]]:
        if not self.temp_samples:
            return None
        sample = self.temp_samples[self._temp_index]
        self._temp_index = (self._temp_index + 1) % len(self.temp_samples)
        return sample

    # ---------------- Device management ----------------

    def add_dummy_device(self, name: str, normal_reading: float):
        self.devices.append(Device(name, "dummy", normal_reading))

    def add_wind_speed_device(self, name: str):
        self.devices.append(Device(name, "wind_speed", None))

    def add_wind_dir_device(self, name: str):
        self.devices.append(Device(name, "wind_dir", None))

    def add_inside_temp_device(self, name: str):
        self.devices.append(Device(name, "inside_temp", None))

    def add_outside_temp_device(self, name: str):
        self.devices.append(Device(name, "outside_temp", None))

    def set_signal_strength(self, index: int, value: float):
        if 0 <= index < len(self.devices):
            self.devices[index].signal_strength = max(0.0, min(1.0, value))

    def set_hacked(self, index: int, hacked: bool):
        if 0 <= index < len(self.devices):
            self.devices[index].hacked = hacked

    # ---------------- Simulation step ----------------

    def update_readings(self):
        wind_sample = self._next_wind_sample()
        temp_sample = self._next_temp_sample()
        for d in self.devices:
            d.update(
                wind_sample,
                self.ws_std,
                self.wd_std,
                temp_sample,
                self.tin_std,
                self.tout_std,
            )

    # ---------------- Snapshot for UI ----------------

    def get_snapshot(self):
        snap = []
        for idx, d in enumerate(self.devices):
            if d.device_type == "dummy":
                type_str = "Dummy"
            elif d.device_type == "wind_speed":
                type_str = "windspeed (m/s)"
            elif d.device_type == "wind_dir":
                type_str = "winddir()"
            elif d.device_type == "inside_temp":
                type_str = "inside temp (°C)"
            else:
                type_str = "outside temp (°C)"

            snap.append(
                {
                    "idx": idx,
                    "name": d.name,
                    "type": type_str,
                    "normal": d.normal_reading,
                    "signal": d.signal_strength,
                    "hacked": d.hacked,
                    "value": d.value,
                    "fault": d.fault_status(),
                    "llm_des": d.llm_decision,
                    "llm_final": d.llm_final,
                    "comment": d.comment,
                }
            )
        return snap


# ---------------------------------------------------------------------------
# Dify clients
# ---------------------------------------------------------------------------

class DifyClient:
    """
    Minimal client for calling a Dify Chat App API (classification app).

    We send ONLY (in `inputs`):
      - device_name
      - device_type
      - normal
      - value   (current value)

    And we send the last ~5 seconds of values in the `query` text.

    The app should respond with a single word in `answer`:
      GOOD, ATTENTION, or FAULT.
    """

    def __init__(
        self,
        base_url: str = "http://localhost/v1",
        api_key: str = "YOUR_API_KEY_HERE",
        enable: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enable = enable and requests is not None

    def classify_device(self, device: Device) -> Optional[str]:
        """Send one device's recent history to Dify and return GOOD / ATTENTION / FAULT."""
        if not self.enable:
            return None

        try:
            url = f"{self.base_url}/chat-messages"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Current value for inputs
            value = device.value
            if isinstance(value, float):
                val_for_llm = round(value, 4)
            else:
                val_for_llm = value

            # Last ~5 seconds of values (oldest -> newest)
            recent_vals = device.recent_values(window_sec=5.0)
            recent_str = ", ".join(str(v) for v in recent_vals) if recent_vals else "none"

            query_text = (
                "You are a SCADA sensor fault classifier.\n"
                "Classify the state of this device as GOOD, ATTENTION or FAULT, "
                "using the recent time series and any retrieved knowledge.\n\n"
                f"device_name: {device.name}\n"
                f"device_type: {device.device_type}\n"
                f"normal: {device.normal_reading}\n"
                f"current_value: {val_for_llm}\n"
                f"recent_values_last_5s (oldest->newest): [{recent_str}]\n\n"
                "Return only one word: GOOD, ATTENTION, or FAULT."
            )

            payload = {
                "inputs": {
                    "device_name": device.name,
                    "device_type": device.device_type,
                    "normal": device.normal_reading,
                    "value": val_for_llm,
                },
                "query": query_text,
                "response_mode": "blocking",
                "conversation_id": "",
                "user": "scada-simulator",
            }

            resp = requests.post(  # type: ignore[arg-type]
                url,
                headers=headers,
                json=payload,
                timeout=3,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "")
            if not isinstance(answer, str):
                return None

            decision = answer.strip().upper()
            if decision not in ("GOOD", "ATTENTION", "FAULT"):
                return None
            return decision

        except Exception as e:
            print(f"Dify classify error for {device.name}: {e}")
            return None


class DifyCommentClient:
    """
    Second Dify app for generating concise comments when LLM Final == FAULT.

    It returns a short natural-language comment in data['answer'].
    """

    def __init__(
        self,
        base_url: str = "http://localhost/v1",
        api_key: str = "YOUR_API_KEY_HERE",
        enable: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enable = enable and requests is not None

    def comment_on_fault(self, device: Device) -> Optional[str]:
        """Ask Dify for a short comment explaining a FAULT state."""
        if not self.enable:
            return None

        try:
            url = f"{self.base_url}/chat-messages"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            value = device.value
            if isinstance(value, float):
                val_for_llm = round(value, 4)
            else:
                val_for_llm = value

            recent_vals = device.recent_values(window_sec=5.0)
            recent_str = ", ".join(str(v) for v in recent_vals) if recent_vals else "none"

            query_text = (
                "You are a SCADA assistant.\n"
                "The aggregated LLM label for this device is FAULT.\n"
                "Provide a concise comment (max 20 words) to help the operator\n"
                "understand the probable issue, based on the data.\n\n"
                f"device_name: {device.name}\n"
                f"device_type: {device.device_type}\n"
                f"normal: {device.normal_reading}\n"
                f"current_value: {val_for_llm}\n"
                f"recent_values_last_5s (oldest->newest): [{recent_str}]\n\n"
                "Return ONLY the comment text, no label and no extra explanation."
            )

            payload = {
                "inputs": {
                    "device_name": device.name,
                    "device_type": device.device_type,
                    "normal": device.normal_reading,
                    "value": val_for_llm,
                    "llm_final_descision": "FAULT",
                },
                "query": query_text,
                "response_mode": "blocking",
                "conversation_id": "",
                "user": "scada-simulator-comment",
            }

            resp = requests.post(  # type: ignore[arg-type]
                url,
                headers=headers,
                json=payload,
                timeout=3,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "")
            if not isinstance(answer, str):
                return None

            return answer.strip()

        except Exception as e:
            print(f"Dify comment error for {device.name}: {e}")
            return None


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class SCADAApp:
    UPDATE_INTERVAL_MS = 1000   # simulate every 1s
    LLM_INTERVAL_MS = 5000      # call Dify every 5s

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SCADA Simulator (Large Font UI)")
        self.root.geometry("1700x850")

        # Global bigger font
        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=18)
        self.root.option_add("*Font", base_font)

        style = ttk.Style()
        style.configure("Treeview", rowheight=32, font=base_font)
        style.configure("Treeview.Heading", font=(base_font.actual("family"), 18, "bold"))

        self.sim = SCADASimulator()
        self.current_selected_index: Optional[int] = None

        # Update the api keys below with your own Dify app keys and urls
        self.dify = DifyClient(
            base_url="http://localhost/v1",
            api_key="app-xxx",  # classifier
            enable=True,
        )
        self.comment_llm = DifyCommentClient(
            base_url="http://localhost/v1",
            api_key="app-xxx",  # comment generator
            enable=True,
        )

        # Devices
        self.sim.add_dummy_device("Dummy_1", 10.0)
        self.sim.add_dummy_device("Dummy_2", 20.0)
        self.sim.add_dummy_device("Dummy_3", 30.0)
        self.sim.add_wind_speed_device("WindSpeed_1")
        self.sim.add_wind_dir_device("WindDir_1")
        self.sim.add_inside_temp_device("InsideTemp_1")
        self.sim.add_outside_temp_device("OutsideTemp_1")

        self._build_ui()
        self._refresh_table()
        self._schedule_update()
        self._schedule_llm()

    # ---------------- UI Construction ----------------

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Table (Treeview)
        table_frame = ttk.Frame(left_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = (
            "idx",
            "name",
            "type",
            "normal",
            "signal",
            "hacked",
            "value",
            "fault",
            "llm_des",
            "llm_final",
            "comment",
        )
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=15,
        )

        headings = {
            "idx": "Idx",
            "name": "Name",
            "type": "Type",
            "normal": "Normal",
            "signal": "Signal",
            "hacked": "Hacked",
            "value": "Value",
            "fault": "Fault",
            "llm_des": "LLM Des",
            "llm_final": "LLM Final",
            "comment": "Comment",
        }
        widths = {
            "idx": 50,
            "name": 180,
            "type": 240,
            "normal": 120,
            "signal": 120,
            "hacked": 120,
            "value": 180,
            "fault": 120,
            "llm_des": 140,
            "llm_final": 140,
            "comment": 350,
        }

        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.CENTER)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right-side controls
        info_frame = ttk.LabelFrame(right_frame, text="Selected Device")
        info_frame.pack(fill=tk.X, pady=10)

        self.label_selected_name = ttk.Label(info_frame, text="Name: -")
        self.label_selected_name.pack(anchor="w", pady=2)

        self.label_selected_type = ttk.Label(info_frame, text="Type: -")
        self.label_selected_type.pack(anchor="w", pady=2)

        self.label_selected_fault = ttk.Label(info_frame, text="Fault: -")
        self.label_selected_fault.pack(anchor="w", pady=2)

        self.label_selected_llm = ttk.Label(info_frame, text="LLM Des: -")
        self.label_selected_llm.pack(anchor="w", pady=2)

        self.label_selected_llm_final = ttk.Label(info_frame, text="LLM Final: -")
        self.label_selected_llm_final.pack(anchor="w", pady=2)

        self.label_selected_comment = ttk.Label(info_frame, text="Comment: -", wraplength=400, justify=tk.LEFT)
        self.label_selected_comment.pack(anchor="w", pady=4)

        # Signal strength slider + hack toggle
        signal_frame = ttk.LabelFrame(right_frame, text="Signal & Hacking")
        signal_frame.pack(fill=tk.X, pady=10)

        ttk.Label(signal_frame, text="Signal (0.0 - 1.0):").pack(anchor="w", pady=2)

        self.signal_scale = tk.Scale(
            signal_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=400,
            showvalue=True,
            command=self._on_signal_change,
        )
        self.signal_scale.set(100)  # default 1.0
        self.signal_scale.pack(fill=tk.X, pady=5)

        self.hacked_var = tk.BooleanVar(value=False)
        hacked_check = ttk.Checkbutton(
            signal_frame,
            text="Device hacked",
            variable=self.hacked_var,
            command=self._on_hacked_change,
        )
        hacked_check.pack(anchor="w", pady=5)

        # Rules info
        rules_frame = ttk.LabelFrame(right_frame, text="Fault Rules")
        rules_frame.pack(fill=tk.X, pady=10)

        rules_text = (
            "Local FAULT rule:\n"
            "  - FAULT when signal < 0.3 or device is hacked\n"
            "  - ATTENTION when 0.3 <= signal <= 0.8 and not hacked\n"
            "  - GOOD when signal > 0.8 and not hacked\n\n"
            "Sensors:\n"
            "  - 'windspeed (m/s)': Value = wind speed\n"
            "  - 'winddir()': Value = wind direction (deg)\n"
            "  - 'inside temp (°C)': inside temperature\n"
            "  - 'outside temp (°C)': outside temperature\n"
            "  - Normal is N/A for wind & temperature.\n\n"
            "LLM Des (via Dify classifier):\n"
            "  - Uses device_name, device_type, normal, current value\n"
            "  - And recent values from the last 5 seconds.\n"
            "LLM Final:\n"
            "  - Aggregated majority of the last 6 LLM decisions\n"
            "    (ties broken conservatively: FAULT > ATTENTION > GOOD).\n\n"
            "Comment (via second Dify workflow):\n"
            "  - Generated only when LLM Final becomes FAULT.\n"
            "  - Short explanation to help the operator."
        )
        ttk.Label(rules_frame, text=rules_text, justify=tk.LEFT, wraplength=430).pack(anchor="w")

    # ---------------- UI Helpers ----------------

    def _schedule_update(self):
        self.root.after(self.UPDATE_INTERVAL_MS, self._update_loop)

    def _schedule_llm(self):
        self.root.after(self.LLM_INTERVAL_MS, self._llm_loop)

    def _update_loop(self):
        self.sim.update_readings()
        self._refresh_table()
        self._schedule_update()

    def _llm_loop(self):

        # Every 5s, ask Dify for a decision for each device in parallel
        if self.dify and self.dify.enable:
            devices = list(self.sim.devices)
            prev_finals = [d.llm_final for d in devices]

            with ThreadPoolExecutor(max_workers=len(devices) or 1) as ex:
                future_to_info = {
                    ex.submit(self.dify.classify_device, d): (i, prev_finals[i])
                    for i, d in enumerate(devices)
                }
                for fut in as_completed(future_to_info):
                    idx, prev_final = future_to_info[fut]
                    try:
                        decision = fut.result()
                    except Exception as e:
                        print(f"LLM classification error for device index {idx}: {e}")
                        decision = None

                    if decision is not None:
                        devices[idx].update_llm_decision(decision)
                        curr_final = devices[idx].llm_final

                        # If LLM Final just became FAULT, ask comment-LLM
                        if (
                            curr_final == "FAULT"
                            and prev_final != "FAULT"
                            and self.comment_llm
                            and self.comment_llm.enable
                        ):
                            comment = self.comment_llm.comment_on_fault(devices[idx])
                            if comment:
                                devices[idx].comment = comment

                        # If LLM Final returned to GOOD from FAULT, clear comment
                        elif prev_final == "FAULT" and curr_final == "GOOD":
                            devices[idx].comment = None
                    # If decision is None, leave llm_final/comment unchanged

        self._refresh_table()
        self._schedule_llm()


    def _refresh_table(self):
        selected = self.current_selected_index

        for row in self.tree.get_children():
            self.tree.delete(row)

        for row in self.sim.get_snapshot():
            idx = row["idx"]
            name = row["name"]
            dev_type = row["type"]
            normal = row["normal"]
            signal = row["signal"]
            hacked = row["hacked"]
            value = row["value"]
            fault = row["fault"]
            llm_des = row["llm_des"]
            llm_final = row["llm_final"]
            comment = row["comment"]

            normal_str = "NA" if normal is None else f"{normal:.2f}"
            signal_str = f"{signal:.2f}"
            hacked_str = "Yes" if hacked else "No"

            if isinstance(value, str):
                value_str = value
            elif value is None:
                value_str = "NA"
            else:
                value_str = f"{value:.2f}"

            llm_str = llm_des if llm_des is not None else ""
            llm_final_str = llm_final if llm_final is not None else ""
            comment_str = comment if comment is not None else ""

            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    idx,
                    name,
                    dev_type,
                    normal_str,
                    signal_str,
                    hacked_str,
                    value_str,
                    fault,
                    llm_str,
                    llm_final_str,
                    comment_str,
                ),
            )

        # Reselect previous selection
        if selected is not None and str(selected) in self.tree.get_children(""):
            self.tree.selection_set(str(selected))
            self.tree.focus(str(selected))
        else:
            selected = None
        self.current_selected_index = selected
        self._update_selected_labels()

    def _on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            self.current_selected_index = None
        else:
            self.current_selected_index = int(sel[0])
        self._update_controls_from_device()
        self._update_selected_labels()

    def _update_selected_labels(self):
        if self.current_selected_index is None:
            self.label_selected_name.config(text="Name: -")
            self.label_selected_type.config(text="Type: -")
            self.label_selected_fault.config(text="Fault: -")
            self.label_selected_llm.config(text="LLM Des: -")
            self.label_selected_llm_final.config(text="LLM Final: -")
            self.label_selected_comment.config(text="Comment: -")
            return

        d = self.sim.devices[self.current_selected_index]
        if d.device_type == "dummy":
            type_str = "Dummy"
        elif d.device_type == "wind_speed":
            type_str = "windspeed (m/s)"
        elif d.device_type == "wind_dir":
            type_str = "winddir()"
        elif d.device_type == "inside_temp":
            type_str = "inside temp (°C)"
        else:
            type_str = "outside temp (°C)"

        self.label_selected_name.config(text=f"Name: {d.name}")
        self.label_selected_type.config(text=f"Type: {type_str}")
        self.label_selected_fault.config(text=f"Fault: {d.fault_status()}")
        self.label_selected_llm.config(
            text=f"LLM Des: {d.llm_decision if d.llm_decision else '-'}"
        )
        self.label_selected_llm_final.config(
            text=f"LLM Final: {d.llm_final if d.llm_final else '-'}"
        )
        self.label_selected_comment.config(
            text=f"Comment: {d.comment if d.comment else '-'}"
        )

    def _update_controls_from_device(self):
        if self.current_selected_index is None:
            return
        d = self.sim.devices[self.current_selected_index]
        self.signal_scale.set(int(d.signal_strength * 100))
        self.hacked_var.set(d.hacked)

    def _on_signal_change(self, _event=None):
        if self.current_selected_index is None:
            return
        value = self.signal_scale.get() / 100.0
        self.sim.set_signal_strength(self.current_selected_index, value)
        self._update_selected_labels()
        self._refresh_table()

    def _on_hacked_change(self):
        if self.current_selected_index is None:
            return
        hacked = self.hacked_var.get()
        self.sim.set_hacked(self.current_selected_index, hacked)
        self._update_selected_labels()
        self._refresh_table()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = SCADAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
