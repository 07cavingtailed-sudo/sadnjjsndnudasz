"""Synthetic key input, with a backend per platform.

The important detail is *scancodes*.  Geometry Dash reads the keyboard through
raw input, and a synthetic event carrying only a virtual-key code is frequently
ignored by games that do this -- the classic symptom of "my bot works in Notepad
but the game does nothing".  The Windows backend therefore sends scancodes via
``SendInput``, and the Linux backend writes to ``/dev/uinput``, which produces
events indistinguishable from a real keyboard at the kernel level.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from abc import ABC, abstractmethod

from gdbot.config import InputConfig

#: Scancodes (set 1) for the keys worth binding to jump.
SCANCODES = {"space": 0x39, "up": 0xC8, "w": 0x11, "enter": 0x1C}
#: X11 keysym names for the xdotool fallback.
XDO_KEYS = {"space": "space", "up": "Up", "w": "w", "enter": "Return"}
#: Linux input event codes (``linux/input-event-codes.h``).
EV_KEYS = {"space": 57, "up": 103, "w": 17, "enter": 28}


class KeyPresser(ABC):
    """Holds and releases one key."""

    def __init__(self, key: str = "space") -> None:
        self.key = key
        self.held = False

    @abstractmethod
    def _down(self) -> None: ...

    @abstractmethod
    def _up(self) -> None: ...

    def set(self, hold: bool) -> None:
        """Make the key's state match ``hold``, sending nothing if it already does."""
        if hold and not self.held:
            self._down()
            self.held = True
        elif not hold and self.held:
            self._up()
            self.held = False

    def release(self) -> None:
        self.set(False)

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "KeyPresser":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullPresser(KeyPresser):
    """Records presses without sending them.  Used by tests and dry runs."""

    def __init__(self, key: str = "space") -> None:
        super().__init__(key)
        self.events: list[bool] = []

    def _down(self) -> None:
        self.events.append(True)

    def _up(self) -> None:
        self.events.append(False)


class SendInputPresser(KeyPresser):  # pragma: no cover - Windows only
    """Windows ``SendInput`` with scancodes."""

    def __init__(self, key: str = "space") -> None:
        super().__init__(key)
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        if key not in SCANCODES:
            raise ValueError(f"unsupported key {key!r}; choose from {sorted(SCANCODES)}")
        self.scan = SCANCODES[key]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]

            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT = INPUT
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)

    def _send(self, flags: int) -> None:
        # 0x0008 = KEYEVENTF_SCANCODE, 0x0002 = KEYEVENTF_KEYUP, 0x0001 = EXTENDEDKEY
        extended = 0x0001 if self.scan > 0x7F else 0
        inp = self._INPUT(type=1)
        inp.ki.wVk = 0
        inp.ki.wScan = self.scan & 0x7F if extended else self.scan
        inp.ki.dwFlags = 0x0008 | extended | flags
        inp.ki.time = 0
        inp.ki.dwExtraInfo = None
        sent = self._user32.SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(inp))
        if sent != 1:
            raise OSError(f"SendInput failed: {self._ctypes.get_last_error()}")

    def _down(self) -> None:
        self._send(0)

    def _up(self) -> None:
        self._send(0x0002)


class UinputPresser(KeyPresser):  # pragma: no cover - needs /dev/uinput
    """Linux ``/dev/uinput`` virtual keyboard."""

    def __init__(self, key: str = "space") -> None:
        super().__init__(key)
        try:
            from evdev import UInput, ecodes
        except ImportError as exc:
            raise RuntimeError("the uinput backend needs 'evdev' (pip install evdev)") from exc
        if key not in EV_KEYS:
            raise ValueError(f"unsupported key {key!r}; choose from {sorted(EV_KEYS)}")
        self._ecodes = ecodes
        self._code = EV_KEYS[key]
        self._ui = UInput({ecodes.EV_KEY: [self._code]}, name="gdbot-keyboard")

    def _emit(self, value: int) -> None:
        self._ui.write(self._ecodes.EV_KEY, self._code, value)
        self._ui.syn()

    def _down(self) -> None:
        self._emit(1)

    def _up(self) -> None:
        self._emit(0)

    def close(self) -> None:
        super().close()
        self._ui.close()


class XdotoolPresser(KeyPresser):  # pragma: no cover - needs X11 + xdotool
    """X11 fallback.  Correct but slow: each press spawns a process."""

    def __init__(self, key: str = "space") -> None:
        super().__init__(key)
        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool is not installed")
        self.name = XDO_KEYS.get(key, key)

    def _run(self, verb: str) -> None:
        subprocess.run(["xdotool", verb, self.name], check=False)

    def _down(self) -> None:
        self._run("keydown")

    def _up(self) -> None:
        self._run("keyup")


class PynputPresser(KeyPresser):  # pragma: no cover - needs a desktop session
    """Cross-platform fallback.  Works for menus; may be ignored by the game."""

    def __init__(self, key: str = "space") -> None:
        super().__init__(key)
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError("the pynput backend needs 'pynput'") from exc
        self._kb = Controller()
        self._key = {"space": Key.space, "up": Key.up, "enter": Key.enter}.get(key, key)

    def _down(self) -> None:
        self._kb.press(self._key)

    def _up(self) -> None:
        self._kb.release(self._key)


#: Backends in the order ``auto`` tries them, per platform.
_ORDER = {
    "Windows": ("sendinput", "pynput"),
    "Linux": ("uinput", "xdotool", "pynput"),
    "Darwin": ("pynput",),
}
_BACKENDS = {
    "sendinput": SendInputPresser,
    "uinput": UinputPresser,
    "xdotool": XdotoolPresser,
    "pynput": PynputPresser,
    "null": NullPresser,
}


def make_presser(cfg: InputConfig) -> KeyPresser:
    """Build the best available presser for this machine.

    ``auto`` walks the platform's preference order and returns the first backend
    that constructs, so a missing optional dependency degrades instead of
    crashing.  The reason each attempt failed is kept and reported together if
    none works, because "no input backend" with no explanation is the single most
    annoying way for this to fail.
    """
    if cfg.backend != "auto":
        if cfg.backend not in _BACKENDS:
            raise ValueError(f"unknown input backend {cfg.backend!r}; have {sorted(_BACKENDS)}")
        return _BACKENDS[cfg.backend](cfg.key)

    problems: list[str] = []
    for name in _ORDER.get(platform.system(), ("pynput",)):
        try:
            return _BACKENDS[name](cfg.key)
        except Exception as exc:  # noqa: BLE001 - report every backend's reason
            problems.append(f"  {name}: {exc}")
    raise RuntimeError(
        "no working input backend on this machine:\n" + "\n".join(problems)
    )
