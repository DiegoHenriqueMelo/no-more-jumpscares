import socket
import json
import time
import os

# Habilita ANSI no cmd do Windows
os.system("")

PORTA = 9999

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"

W = 52  # largura alvo em chars

ACAO_CORES = {
    "nada":               DIM,
    "porta_esquerda":     CYAN,
    "porta_direita":      CYAN,
    "luz_esquerda":       YELLOW,
    "luz_direita":        YELLOW,
    "abrir_fechar_camera": MAGENTA,
}

ACAO_ABREV = {
    "nada":               "nada",
    "porta_esquerda":     "porta_esq",
    "porta_direita":      "porta_dir",
    "luz_esquerda":       "luz_esq",
    "luz_direita":        "luz_dir",
    "abrir_fechar_camera": "cam_tab",
}


def _cor_acao(acao: str) -> str:
    if acao.startswith("camera_"):
        return BLUE
    return ACAO_CORES.get(acao, WHITE)


def formatar_step(d: dict) -> str:
    acao    = d.get("acao", "?")
    valida  = d.get("valida", True)
    reward  = d.get("reward", 0.0)
    energia = d.get("energia", 0.0)
    passos  = d.get("passos", 0)
    pe      = d.get("porta_esq", 0)
    pd      = d.get("porta_dir", 0)
    le      = d.get("luz_esq", 0)
    ld      = d.get("luz_dir", 0)
    ca      = d.get("camera_aberta", 0)
    cv      = d.get("camera_ativa", 0)
    tempo   = d.get("tempo", 0.0)

    abrev      = ACAO_ABREV.get(acao, acao)
    cor_acao   = _cor_acao(acao)
    cor_reward = GREEN if reward > 0 else (RED if reward < 0 else DIM)
    str_valida = f"{GREEN}v{RESET}" if valida else f"{RED}x{RESET}"

    return (
        f"{DIM}#{passos:4d}{RESET} "
        f"{cor_acao}{abrev:<10}{RESET}"
        f"{str_valida} "
        f"R:{cor_reward}{reward:+5.1f}{RESET} "
        f"E:{energia:3.0f}% "
        f"D:{pe}{pd}L:{le}{ld} "
        f"C:{ca}/{cv} "
        f"T:{tempo:3.0f}s"
    )


def formatar_episodio(d: dict) -> str:
    ep         = d.get("ep", 0)
    resultado  = d.get("resultado", "?")
    passos     = d.get("passos", 0)
    tempo_min  = d.get("tempo_min", 0.0)
    recompensa = d.get("recompensa", 0.0)
    taxa       = d.get("taxa_vitoria", 0.0)

    if resultado == "VITORIA":
        cor_res = GREEN + BOLD
    elif resultado == "MORTE":
        cor_res = RED + BOLD
    else:
        cor_res = YELLOW + BOLD

    sep = f"{CYAN}{'─' * W}{RESET}"
    return (
        f"\n{sep}\n"
        f"{BOLD}Ep {ep:3d}{RESET} {cor_res}{resultado:12s}{RESET} p:{passos}\n"
        f"R:{recompensa:7.1f}  t:{tempo_min:.2f}m  vit:{taxa:.1f}%\n"
        f"{sep}"
    )


def main():
    print(f"\n{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"{BOLD}{CYAN}  FNAF IA — Monitor de Treino{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}\n")
    print(f"{DIM}Aguardando conexao com o treino...{RESET}\n")

    s = None
    for _ in range(30):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("localhost", PORTA))
            break
        except ConnectionRefusedError:
            s.close()
            s = None
            time.sleep(0.5)

    if s is None:
        print(f"{RED}Nao foi possivel conectar. Execute o treino com --debug.{RESET}")
        input("Enter para fechar...")
        return

    print(f"{GREEN}Conectado! Monitorando acoes da IA...{RESET}\n")
    print(f"{DIM}{'#':5} {'Acao':<10} {'V':1} {'Reward':>6} {'E':>4} {'DL':>4} {'C':>3} {'T':>5}{RESET}")
    print(f"{DIM}{'─' * W}{RESET}")

    buffer = ""
    rodando = True
    try:
        while rodando:
            chunk = s.recv(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                linha, buffer = buffer.split("\n", 1)
                if not linha.strip():
                    continue
                try:
                    dados = json.loads(linha)
                    tipo  = dados.get("tipo")
                    if tipo == "step":
                        print(formatar_step(dados))
                    elif tipo == "episodio":
                        print(formatar_episodio(dados))
                    elif tipo == "fim":
                        rodando = False
                        break
                except json.JSONDecodeError:
                    pass
    except (ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    finally:
        if s:
            try:
                s.close()
            except Exception:
                pass

    print(f"\n{YELLOW}Treino encerrado.{RESET}")
    input("Enter para fechar...")


if __name__ == "__main__":
    main()
