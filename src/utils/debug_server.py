import socket
import threading
import json
import subprocess
import time
import sys
import os

PORTA = 9999


class DebugServer:
    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("localhost", PORTA))
        self._server.listen(1)
        self._server.settimeout(20)

        projeto_dir = os.getcwd()
        subprocess.Popen(
            f'start "FNAF IA Debug" cmd /k "{sys.executable} -m src.utils.debug_viewer"',
            shell=True,
            cwd=projeto_dir,
        )

        print("[Debug] Aguardando viewer conectar...")
        try:
            conn, _ = self._server.accept()
            self._conn = conn
            print("[Debug] Viewer conectado!")
        except socket.timeout:
            print("[Debug] Viewer nao conectou. Modo debug desativado.")

    def _enviar(self, dados: dict):
        if self._conn is None:
            return
        try:
            with self._lock:
                msg = json.dumps(dados, ensure_ascii=False) + "\n"
                self._conn.sendall(msg.encode("utf-8"))
        except Exception:
            self._conn = None

    def log_step(self, info: dict, reward: float):
        self._enviar({
            "tipo": "step",
            "acao": info.get("acao_nome", "?"),
            "valida": bool(info.get("acao_valida", True)),
            "reward": round(float(reward), 2),
            "energia": round(float(info.get("energia", 0)), 1),
            "passos": int(info.get("passos", 0)),
            "porta_esq": int(bool(info.get("porta_esq", False))),
            "porta_dir": int(bool(info.get("porta_dir", False))),
            "luz_esq": int(bool(info.get("luz_esq", False))),
            "luz_dir": int(bool(info.get("luz_dir", False))),
            "camera_aberta": int(bool(info.get("camera_aberta", False))),
            "camera_ativa": int(info.get("camera_ativa", 0)),
            "tempo": round(float(info.get("tempo", 0)), 1),
        })

    def log_episodio(self, ep: int, resultado: str, passos: int, tempo_s: float, recompensa: float, taxa_vitoria: float):
        self._enviar({
            "tipo": "episodio",
            "ep": ep,
            "resultado": resultado,
            "passos": passos,
            "tempo_min": round(tempo_s / 60.0, 2),
            "recompensa": round(recompensa, 1),
            "taxa_vitoria": round(taxa_vitoria, 1),
        })

    def fechar(self):
        self._enviar({"tipo": "fim"})
        time.sleep(0.3)
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        try:
            self._server.close()
        except Exception:
            pass
