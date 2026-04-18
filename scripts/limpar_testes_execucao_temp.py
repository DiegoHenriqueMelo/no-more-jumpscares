#!/usr/bin/env python3
"""Script temporario para limpar dados de testes de execucao no MongoDB.

Uso comum:
  python scripts/limpar_testes_execucao_temp.py --all-console
  python scripts/limpar_testes_execucao_temp.py --all-console --confirm
  python scripts/limpar_testes_execucao_temp.py --sessao Console-20260416T184553Z-cab51bd5 --confirm

Comportamento:
- Por padrao NAO deleta, apenas mostra o que seria removido.
- Para deletar de fato, passe --confirm.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pymongo import MongoClient


def carregar_env(caminho: Path = Path(".env")) -> None:
    if not caminho.exists():
        return

    for linha in caminho.read_text(encoding="utf-8").splitlines():
        texto = linha.strip()
        if not texto or texto.startswith("#") or "=" not in texto:
            continue
        chave, valor = texto.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        if chave and chave not in os.environ:
            os.environ[chave] = valor


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Limpa documentos de teste da collection de logs por "
            "sessao_treino_id/sessao_execucao_terminal."
        )
    )
    parser.add_argument(
        "--sessao",
        action="append",
        default=[],
        help=(
            "Sessao especifica para remover (novo ou legado). "
            "Pode repetir o argumento."
        ),
    )
    parser.add_argument(
        "--all-console",
        action="store_true",
        help="Seleciona todas as sessoes iniciadas por 'Console-'.",
    )
    parser.add_argument(
        "--pc",
        help="Filtra por PC (campo pc) em conjunto com --sessao/--all-console.",
    )
    parser.add_argument(
        "--uri",
        help="Mongo URI. Fallback: MONGO_URI no .env",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("MONGO_DATABASE", "no_more_jumpscares"),
        help="Database de destino no MongoDB",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MONGO_COLLECTION", "treino_logs"),
        help="Collection de destino no MongoDB",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Executa a exclusao de fato. Sem isso, apenas simula.",
    )
    return parser


def main() -> None:
    carregar_env()
    args = criar_parser().parse_args()

    uri = (args.uri or os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise ValueError("Mongo URI ausente. Passe --uri ou configure MONGO_URI no .env")

    filtro: dict[str, object] = {}
    condicoes_sessao: list[dict[str, object]] = []

    sessoes = [s.strip() for s in args.sessao if s and s.strip()]
    if sessoes:
        condicoes_sessao.extend(
            [
                {"sessao_treino_id": {"$in": sessoes}},
                {"sessao_execucao_terminal": {"$in": sessoes}},
            ]
        )
    if args.all_console:
        condicoes_sessao.extend(
            [
                {"sessao_treino_id": {"$regex": r"^Console-"}},
                {"sessao_execucao_terminal": {"$regex": r"^Console-"}},
            ]
        )

    if condicoes_sessao:
        if len(condicoes_sessao) == 1:
            filtro.update(condicoes_sessao[0])
        else:
            filtro["$or"] = condicoes_sessao

    if args.pc and args.pc.strip():
        filtro["pc"] = args.pc.strip()

    if not filtro:
        filtro = {
            "$or": [
                {"sessao_treino_id": {"$regex": r"^Console-"}},
                {"sessao_execucao_terminal": {"$regex": r"^Console-"}},
            ]
        }
        print(
            "Nenhum filtro informado. Usando padrao seguro: "
            "sessao_treino_id/sessao_execucao_terminal iniciados por 'Console-'."
        )

    cliente = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        cliente.admin.command("ping")
        coll = cliente[args.database][args.collection]

        total = coll.count_documents(filtro)
        print(f"Filtro: {filtro}")
        print(f"Documentos encontrados: {total}")

        if not args.confirm:
            print("Simulacao concluida. Nada foi removido (use --confirm para apagar).")
            return

        resultado = coll.delete_many(filtro)
        print(f"Exclusao concluida. Documentos removidos: {resultado.deleted_count}")
    finally:
        cliente.close()


if __name__ == "__main__":
    main()
