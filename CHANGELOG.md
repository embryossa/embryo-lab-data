# Реализованная реорганизация

Версия ready-1:

- MII сделан необязательным; MIIR считается только по наблюдаемому MII, ноль сохранён.
- Разделены OCCR, MIIR, лабораторный FR, Model_FR и Model_KPIScore.
- Добавлены агрегаты sum/sum, Wilson CI, direction, competency, benchmark, class/source.
- Исправлен верхний контроль DegR/3pNR и срезы времени/метода/эмбриолога/врача.
- Реализован строгий DT + TabPFN + OPU join, enriched OPU и review union после CI.
- DNN требует scaler обучения, проверяет признаки и запрещает fit на inference.
- TabPFN получил headless batch Python + BAT с explicit input/output/run-id.
- Все результаты пишутся атомарно и имеют schema versions/hashes в manifest.
- Агентные пакеты используют только HMAC cycle_uid; crosswalk остаётся локально.
- MIRI анализирует непрерывные episodes, gaps, duration, max deviation, severity/recovery.
- DataLore notebook сохранён как reference, выполняемая логика перенесена в Python.
- Добавлены Claude Code skill `/run-analysis`, Sonnet-роли и итоговый Opus-агент.
- Добавлены синтетические unit/replay fixtures и проверки контрактов.
