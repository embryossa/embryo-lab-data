---
name: miri-forensics
description: MIRI 6391 incubator CSV log schema, setpoints, alarm columns, and Western Electric rules for cultivation-window forensics. Preload into agents that analyze incubator logs against embryo cultivation windows.
---
# Форензик логов инкубатора MIRI 6391

Формат лога (data/miri_logs/*.csv): разделитель `;`, дата `%Y.%m.%d %H:%M:%S`.
Колонки: Temp1..Temp12 (6 камер × 2 датчика), Co2Concentration, O2Concentration,
Co2Flow, N2Flow, Co2Pressure, N2Pressure, TempSetPoint, Co2ConcentrationSetPoint,
O2ConcentrationSetPoint, Chamber1..6TemperatureAlarm, Co2ConcentrationAlarm,
O2ConcentrationAlarm, Co2PressureAlarm, N2PressureAlarm, IsLidOpen.

Уставки по умолчанию: T=37, CO2=6.5, O2=5 (правятся в settings.miri_logs).
Допуски: T ±0.5, CO2 ±0.3, O2 ±0.5.

Окно культивирования: Date OPU .. +6 дней (settings.thresholds.cultivation_window_days).
Сначала выделяем реальные непрерывные эпизоды: начало/конец, продолжительность,
максимальное отклонение, severity, recovery time после закрытия крышки и gaps в ряду.
Warning разрешён только при существующем эпизоде severity=warning/critical; само наличие
строк лога не является предупреждением.

Ограничение: лог не содержит привязки «пациент -> камера» — форензик на уровне
инкубатора (все камеры) за окно, пока не появится маппинг камер. Связь отклонений с
неудачным исходом формулировать как ГИПОТЕЗУ, не как факт.

## Интерпретация пропусков (gap) — правило 8 часов

Каждый `event_type=gap` несёт поле `gap_reason`:
- `power_or_logging_outage` (<8ч, порог `gap_reason_threshold_minutes=480`) — это НЕ
  «нет данных»/потеря наблюдения. Это потеря возможности ПИСАТЬ логи (например,
  отключение электричества) — реальное неблагоприятное событие, которое надо явно
  называть таковым при разборе окна культивирования.
- `incubator_or_logging_stopped` (≥8ч) — явная остановка инкубатора или самого
  логирования, более серьёзный по природе сигнал (не просто перебой).

Не описывай короткий gap как «отсутствие данных» — называй его перебоем
питания/логирования. Не путай этот признак с severity (warning/critical, отдельный
10-минутный порог) — оба поля читаются вместе, `gap_reason` не заменяет severity.
