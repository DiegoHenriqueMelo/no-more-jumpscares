# Contribuindo

Obrigado por considerar contribuir para o `no-more-jumpscares`!

Guia rápido:

- Use o padrão de mensagens **Conventional Commits**. Tipos aceitáveis:
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `build`, `ci`, `revert`, `release`.

  Exemplo: `feat(calibrar): adicionar modo de captura de janela`

- Para validar mensagens localmente, habilite os hooks do Git:

```bash
git init                # se ainda não inicializou o repositório
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg  # em sistemas POSIX
```

- Para atualizar a versão semântico (semver) do projeto use o script:

```bash
python scripts/bump_version.py patch   # patch, minor, major
```

- Fluxo recomendado de branches:
  - `main`: código pronto/produção
  - `develop`: integração
  - feature branches: `feat/<descrição>`

- Ao criar PRs, descreva o objetivo, passos para reproduzir (se houver) e notas sobre testes.

Automação de releases:
- Quando um tag `vX.Y.Z` for criado, considere criar uma Release no GitHub com notas.
