"""
Penalidade por cliques repetidos na mesma ação.

Portas e luzes têm tolerância maior (cliques situacionais são comuns).
Demais ações penalizam mais cedo.
"""

# Quantos cliques seguidos na mesma ação antes de começar a penalizar
TOLERANCIA = {
    "porta_esquerda": 6,
    "porta_direita":  6,
    "luz_esquerda":   5,
    "luz_direita":    5,
}
TOLERANCIA_PADRAO = 3  # câmeras, abrir_fechar_camera, etc.

PENALIDADE_POR_EXCESSO = 0.5  # por clique acima da tolerância


class PenalidadeRepeticao:
    def __init__(self):
        self._acao_atual: str | None = None
        self._contador: int = 0

    def reset(self):
        self._acao_atual = None
        self._contador = 0

    def calcular(self, nome_acao: str) -> float:
        """Retorna penalidade (>= 0) para a ação informada."""
        if nome_acao == "nada":
            return 0.0

        if nome_acao == self._acao_atual:
            self._contador += 1
        else:
            self._acao_atual = nome_acao
            self._contador = 1

        tolerancia = TOLERANCIA.get(nome_acao, TOLERANCIA_PADRAO)
        excesso = max(0, self._contador - tolerancia)
        return excesso * PENALIDADE_POR_EXCESSO
