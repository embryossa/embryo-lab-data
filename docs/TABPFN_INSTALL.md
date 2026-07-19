# Установка batch-адаптера TabPFN

Скопируйте два файла из `external/tabpfn/` в папку существующего проекта TabPFN
на этом ПК (её путь задаётся в `config/machine.local.yaml:
external_tools.tabpfn.project_dir`, см. `config/machine.local.yaml.example`):

- `run_tabpfn_batch.py`
- `run_tabpfn_batch.bat`

BAT использует окружение TabPFN в этой же папке и принимает явные аргументы:

```bat
run_tabpfn_batch.bat --input "C:\path\OPU_table.xlsx" --output "C:\path\tabpfn_predictions.csv" --run-id 20260716T120000
```

Скрипт импортирует существующий backend приложения. Если имена модулей отличаются,
он завершится с диагностикой импорта; менять медицинскую логику или подменять результат
не будет. После копирования выполните `python scripts/check_setup.py` из корня проекта.
