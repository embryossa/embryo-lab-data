---
description: Поднять self-hosted Teable (Docker) и синхронизировать processed-файлы обеих клиник одной командой
---
Отдельная от `/run-analysis` задача — только подъём self-hosted Teable и заливка уже
готовых `data/processed` файлов в него. Ничего не пересчитывает и не запускает анализ.

1. Проверь контейнеры: `docker ps --format "{{.Names}}"` в
   `C:\Users\User\Desktop\IVF\AI\embryo-lab-data\teable\self-hosted`. Если
   `self-hosted-teable-1` / `-db-1` / `-cache-1` не в списке или не `Up` — подними:
   `cd teable/self-hosted && docker compose up -d`. Если у самого Docker Desktop ещё
   не поднят движок (`docker ps` падает с ошибкой pipe/daemon), сначала запусти
   `Docker Desktop.exe` и подожди, пока `docker ps` не перестанет падать (проверяй
   раз в 5-10 сек, не дольше ~2 минут).
2. Дождись готовности API: опроси `curl -s http://127.0.0.1:3000/health` пока не
   вернётся `{"status":"ok",...}` (обычно 5-15 сек после `docker compose up -d`,
   дольше — если образы ещё не скачаны).
3. Синхронизируй обе клиники (рабочая директория `teable/`), пробрасывая
   `$ARGUMENTS` в обе команды как есть (например `--force`, `--dry-run`):
   - Two-Line: `python teable_sync.py --config teable_config.local.json $ARGUMENTS`
   - Fertimed: `python teable_sync.py --config teable_config.local.json --base bsef6WTgVoVeJeJD1N7 --clinic "Fertimed" --processed "C:\Users\User\Desktop\IVF\AI\embryo-lab-data\data\processed\fertimed" $ARGUMENTS`
   (base_id Fertimed должен совпадать с тем, что в `teable/sync_all_clinics.bat` —
   если базы когда-то пересоздавались через `teable_bootstrap.py`, обнови id сразу
   в обоих местах.)
4. Ретранслируй пользователю ровно то, что напечатали оба вызова (строка вида
   `Parsed: N patients, M with outcome (+K)` и финальная строка `Done.`/`[skip]`/
   ошибку) — не переформулируй и не досчитывай числа сам.
