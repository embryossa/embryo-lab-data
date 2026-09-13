"""dt_api.py — IVF Digital Twin 7.1 через локальный REST API (подключение «Аналитика»).

Единственный вход Digital Twin для всех клиник (батч 7.0 с RUN_BATCH.bat снят 2026-09-13).
Пайплайн отправляет пакет циклов в службу Windows `IVFClinicIntegration71`, служба
считает их тем же кодом, что и настольное приложение, и отдаёт строки в формате
CSV-аналитики «Истории» 7.1 (первый блок — имена колонок dt_predictions 7.0).

Что уходит в службу: только cycle_uid (HMAC, без ID/ФИО/дат) и числовые признаки.
Сопоставление с `_key` делается здесь, локально.

Две временные точки на цикл (владелец, 2026-09-13: модели передаётся всё, что есть
в выгрузке):
  * `transfer`  — все наблюдения (ОКК, MII, 2PN, бластоцисты, хорошие) -> вероятности
                  (DIGITAL TWIN, p_*, bayes_*, BEFE);
  * `retrieval` — только ОКК и MII, как батч 7.0 -> медианы воронки med_*/p025/p975.
Иначе прогноз бластоцист совпал бы с фактическим числом, и сравнение воронки
«прогноз против факта» в отчёте выродилось бы.

Пропуски обязательных признаков API сам не заполняет. Подстановки делает пайплайн,
явно и с флагом в строке: amh_imputed / bmi_imputed (normalize.py), afc_source,
attempt_imputed, sperm_source_inferred.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

UNITS = {"age": "years", "amh": "ng/mL", "afc": "count", "bmi": "kg/m2",
         "height_cm": "cm", "weight_kg": "kg"}
MAX_CYCLES_PER_BATCH = 500
TIMEPOINTS = ("transfer", "retrieval")

# Наблюдения этапов: канонические колонки пайплайна -> поля API.
STAGE_FIELDS = {"known_okk": "occ", "known_mii": "mii", "known_pn2": "two_pn",
                "known_blasts": "bl_total", "known_good": "bl_good", "known_euploid": "euploid"}
RETRIEVAL_FIELDS = ("known_okk", "known_mii")

# Esteves 2019 strata в 7.1. SMF/TESE без диагноза OA/NOA сопоставляются с OA и
# помечаются флагом sperm_source_inferred; клиника может переопределить карту.
DEFAULT_SPERM_SOURCE_MAP = {
    "n": "ejaculate", "ejaculate": "ejaculate", "эякулят": "ejaculate",
    "donor": "donor", "донор": "donor", "донорская": "donor",
    "mesa": "epididymal", "pesa": "epididymal",
    "tese": "testicular_OA", "smf": "testicular_OA", "micro-tese": "testicular_OA",
}
UNAMBIGUOUS_SPERM_TOKENS = {"n", "ejaculate", "эякулят", "donor", "донор", "донорская", "mesa", "pesa"}

# 7.0-имена, которых нет в CSV аналитики 7.1, но которые читают report/consult/teable.
ALIASES_70 = {"p_gnn_raw": "gnn_prob", "p_gnn_ens": "gnn_ensemble_prob", "w_gnn": "gnn_weight",
              "csdi_available": "layer_L5", "gnn_available": "layer_L6",
              "med_bl": "med_blasts", "med_goodbl": "med_good"}
# Эхо входа, как в dt_analytics 7.0 (analytics._restore_true_counts затем
# перезаписывает их истинными до-модельными значениями по cycle_uid).
ECHO_70 = {"Age": "age", "Attempt": "attempt_no", "N folicules OPU": "follicles", "OCC": "occ",
           "MII": "mii", "Inseminated": "inseminated", "2pN": "two_pn", "Bl": "bl_total",
           "Good Bl": "bl_good", "real_pn2": "two_pn", "real_cleav": "cleavage_d3",
           "real_bl": "bl_total", "real_goodbl": "bl_good", "real_cryo": "frozen"}
FUNNEL_COLUMNS = ("med_okk", "med_mii", "med_pn2", "med_blasts", "med_good", "med_euploid",
                  "med_warmed", "p025_okk", "p975_okk", "p025_blasts", "p975_blasts",
                  "p025_good", "p975_good", "p025_mii", "p975_mii", "p025_pn2", "p975_pn2",
                  "p025_euploid", "p975_euploid", "p_no_blast", "p_no_good_blast")
# Вероятности ретроспективной точки «после пункции» — для прямого сравнения с 7.0.
RETRIEVAL_SUMMARY = ("clinical_probability", "befe_per_transfer", "cycle_probability",
                     "p_per_transfer", "bayes_mean", "p_kat_raw", "p_csdi", "gnn_prob",
                     "befe_ci_low", "befe_ci_high", "befe_reliability")


class DTApiError(RuntimeError):
    pass


# ------------------------------------------------------------------ key storage

class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, decrypt: bool) -> bytes:
    """DPAPI текущего пользователя Windows: ключ читается только под этой учётной записью."""
    if os.name != "nt":
        raise DTApiError("хранение ключа API требует Windows DPAPI")
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    buf = ctypes.create_string_buffer(data, len(data))
    source = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    target = _Blob()
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    fn.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    # CRYPTPROTECT_UI_FORBIDDEN = 1
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel.LocalFree(target.pbData)


def default_key_file() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "embryo-lab-data" / "dt-api.key"


def save_key(key: str, path: str | Path | None = None) -> Path:
    key = key.strip()
    if not key.startswith("ivf_api_"):
        raise DTApiError("это не ключ API IVF Digital Twin (ожидается ivf_api_…)")
    target = Path(path) if path else default_key_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(_dpapi(key.encode("utf-8"), decrypt=False))
    os.replace(tmp, target)
    return target


def load_key(api_cfg: dict) -> str:
    env = os.environ.get("EMBRYO_DT_API_KEY")
    if env:
        return env.strip()
    path = Path(api_cfg.get("key_file") or default_key_file())
    if not path.is_file():
        raise DTApiError(
            f"нет ключа API Digital Twin ({path}). В 7.1: Настройка → Подключение МИС / ЛИС → "
            "тип «Аналитика» → Создать ключ; затем: python -m embryo.dt_api store-key")
    return _dpapi(path.read_bytes(), decrypt=True).decode("utf-8")


# ------------------------------------------------------------------ HTTP client

class DTApiClient:
    def __init__(self, url: str, key: str, ca_cert: str | Path, timeout: float = 60.0, session=None):
        import requests
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        # Службе нельзя ходить через прокси: она отклоняет заголовки переадресации.
        self.session.trust_env = False
        self.session.verify = str(ca_cert)
        self.session.headers["Authorization"] = "Bearer " + key

    @classmethod
    def from_config(cls, api_cfg: dict) -> "DTApiClient":
        ca = api_cfg.get("ca_cert") or ""
        if not Path(ca).is_file():
            raise DTApiError(f"нет сертификата API ({ca!r}): скачайте его в 7.1, Настройка → Подключение")
        return cls(api_cfg.get("url") or "https://127.0.0.1:8843/api/v1", load_key(api_cfg), ca,
                   timeout=float(api_cfg.get("request_timeout_seconds", 120)))

    def _call(self, method: str, path: str, **kwargs):
        response = self.session.request(method, self.url + path, timeout=self.timeout, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            raise DTApiError(f"{method} {path}: HTTP {response.status_code} "
                             f"{error.get('code', '')} {error.get('fields', '')}".strip())
        return response.status_code, body

    def health(self) -> dict:
        return self._call("GET", "/health")[1]

    def validate(self, body: dict) -> dict:
        """Проверка пакета службой без сохранения и без расчёта."""
        return self._call("POST", "/analytics/validate", json=body)[1]

    def submit(self, body: dict) -> dict:
        return self._call("POST", "/analytics/batches", json=body)[1]

    def status(self, batch_id: str) -> dict:
        return self._call("GET", f"/analytics/batches/{batch_id}")[1]

    def results(self, batch_id: str) -> dict | None:
        code, body = self._call("GET", f"/analytics/batches/{batch_id}/results")
        return body if code == 200 else None

    def delete(self, batch_id: str) -> None:
        self._call("DELETE", f"/analytics/batches/{batch_id}")

    def wait(self, batch_id: str, poll_seconds: float = 15.0, timeout_seconds: float = 6 * 3600,
             sleep=time.sleep, clock=time.monotonic) -> dict:
        started = clock()
        last = None
        while True:
            out = self.results(batch_id)
            if out is not None:
                return out
            state = self.status(batch_id)
            done = state["complete"] + state["failed"] + state["invalid"]
            if done != last:
                print(f"[dt-api] пакет {batch_id[:8]}: {done}/{state['total']} "
                      f"(ошибок {state['failed']}, отклонено {state['invalid']})", file=sys.stderr)
                last = done
            if clock() - started > timeout_seconds:
                raise DTApiError(f"пакет {batch_id} не завершился за {timeout_seconds / 60:.0f} мин "
                                 "(проверьте службу IVFClinicIntegration71 и лицензию)")
            sleep(poll_seconds)


# ------------------------------------------------------------------ input mapping

def _value(row: pd.Series, column: str):
    if column not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _count(row: pd.Series, column: str):
    value = _value(row, column)
    if value is None:
        return None
    # Нецелое число этапа — ошибка выгрузки; отдаём как есть, API отклонит строку явно.
    return int(value) if float(value).is_integer() else value


def _flag(row: pd.Series, column: str) -> bool:
    if column not in row.index:
        return False
    value = row[column]
    return bool(value) if not pd.isna(value) else False


def build_patient(row: pd.Series, timepoint: str, api_cfg: dict) -> tuple[dict, dict]:
    """Признаки одного цикла для API и флаги подстановок (флаги остаются локально)."""
    impute = api_cfg.get("impute") or {}
    flags: dict = {"amh_imputed": _flag(row, "amh_imputed"), "bmi_imputed": _flag(row, "bmi_imputed")}

    afc, source = None, None
    for column in impute.get("afc_columns") or ("afc", "clin_afc"):
        afc = _count(row, column)
        if afc is not None:
            source = column
            break
    if afc is None and impute.get("afc_from_follicles", True):
        afc, source = _count(row, "follicles"), "follicles"
    if afc is None and impute.get("afc_constant") is not None:
        afc, source = int(impute["afc_constant"]), "constant"
    flags["afc_source"] = source

    attempt = _count(row, "attempt_no")
    flags["attempt_imputed"] = attempt is None
    if attempt is None:
        attempt = int(impute.get("attempt", 1))

    sperm_map = {**DEFAULT_SPERM_SOURCE_MAP,
                 **{str(k).strip().lower(): v for k, v in (api_cfg.get("sperm_source_map") or {}).items()}}
    raw = row.get("sperm_source") if "sperm_source" in row.index else None
    token = "" if raw is None or pd.isna(raw) else str(raw).strip().lower()
    sperm = sperm_map.get(token)
    flags["sperm_source_inferred"] = sperm is None or token not in UNAMBIGUOUS_SPERM_TOKENS
    if sperm is None:
        sperm = impute.get("sperm_source", "ejaculate")

    patient = {"age": _value(row, "age"), "amh": _value(row, "amh"), "afc": afc,
               "bmi": _value(row, "bmi"), "attempt": attempt, "sperm_source": sperm,
               "follicles": _count(row, "follicles")}
    fields = RETRIEVAL_FIELDS if timepoint == "retrieval" else tuple(STAGE_FIELDS)
    for field in fields:
        patient[field] = _count(row, STAGE_FIELDS[field])
    # MII > ОКК — в программу добавлены размороженные ооциты (CLAUDE.md), отдельного
    # счётчика размороженных в выгрузках нет. Модели передаётся нижняя граница числа
    # ооцитов в программе (= MII), иначе 7.1 отклоняет цикл по каскаду, а без ОКК
    # симуляция дала бы зрелость >100% в признаках KAT/GAT. Лабораторные ОКК/KPI и эхо
    # OCC в отчёте остаются фактическими (владелец, 2026-09-13).
    okk, mii = patient.get("known_okk"), patient.get("known_mii")
    flags["okk_raised_for_thawed"] = bool(
        impute.get("okk_for_thawed_oocytes", True) and okk is not None and mii is not None and mii > okk)
    if flags["okk_raised_for_thawed"]:
        patient["known_okk"] = mii
    return {k: v for k, v in patient.items() if v is not None}, flags


def build_batches(model_source: pd.DataFrame, api_cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    if "cycle_uid" not in model_source.columns or "_key" not in model_source.columns:
        raise DTApiError("для API нужны колонки _key и cycle_uid")
    timepoints = tuple(api_cfg.get("timepoints") or TIMEPOINTS)
    options = {"n_sim": int(api_cfg.get("n_sim", 2000))}
    if api_cfg.get("clinic_prior"):
        options["clinic_prior"] = [list(map(int, v)) for v in api_cfg["clinic_prior"]]
    cycles, flags = [], []
    for _, row in model_source.iterrows():
        for timepoint in timepoints:
            patient, row_flags = build_patient(row, timepoint, api_cfg)
            cycles.append({"cycle_id": f"{row['cycle_uid']}.{timepoint}", "patient": patient})
        flags.append({"cycle_uid": row["cycle_uid"], **row_flags})
    chunks = [cycles[i:i + MAX_CYCLES_PER_BATCH] for i in range(0, len(cycles), MAX_CYCLES_PER_BATCH)]
    bodies = [{"schema_version": "1", "units": UNITS, "options": options, "cycles": chunk} for chunk in chunks]
    return bodies, pd.DataFrame(flags)


# ------------------------------------------------------------------ output mapping

def _rows_frame(results: list[dict]) -> pd.DataFrame:
    records = []
    for batch in results:
        for row in batch["rows"]:
            uid, _, timepoint = row["cycle_id"].rpartition(".")
            values = row.get("values") or {}
            records.append({"cycle_uid": uid, "timepoint": timepoint, "status": row["status"],
                            "error": json.dumps(row["error"], ensure_ascii=False) if row.get("error") else None,
                            "seed": row.get("seed"), "model_fingerprint": row.get("model_fingerprint"),
                            "batch_id": batch["id"], **{f"v::{k}": v for k, v in values.items()}})
    return pd.DataFrame(records)


def to_dt_output(model_source: pd.DataFrame, results: list[dict], flags: pd.DataFrame,
                 timepoints: tuple[str, ...] = TIMEPOINTS, fallback: bool = True) -> pd.DataFrame:
    """Строки API -> кадр в контракте DT_OUTPUT (как dt_analytics_ready 7.0 + колонки 7.1).

    fallback: если точка `transfer` отклонена (противоречие стадий в выгрузке), а точка
    `retrieval` посчитана, весь блок значений строки берётся из `retrieval` — одна точка
    на строку, без смешивания колонок — и строка помечается
    dt_timepoint_used=retrieval_fallback (владелец, 2026-09-13).
    """
    rows = _rows_frame(results)
    columns = results[0]["columns"] if results else []
    primary = "transfer" if "transfer" in timepoints else timepoints[0]
    base = model_source[["_key", "cycle_uid"]].copy()

    def block(timepoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        part = rows.loc[rows["timepoint"] == timepoint].drop(columns="timepoint").set_index("cycle_uid")
        values = pd.DataFrame({name: part[f"v::{name}"] if f"v::{name}" in part.columns
                               else pd.Series(np.nan, index=part.index) for name in columns}, index=part.index)
        return values, part[["status", "error", "seed", "model_fingerprint", "batch_id"]]

    values, meta = block(primary)
    has_retrieval = "retrieval" in timepoints and primary != "retrieval"
    used = pd.Series(np.where(meta["status"] == "complete", primary, "none"), index=meta.index, dtype=object)
    if has_retrieval:
        r_values, r_meta = block("retrieval")
        if fallback:
            swap = meta.index[(meta["status"] != "complete")
                              & (r_meta["status"].reindex(meta.index) == "complete")]
            if len(swap):
                values = values.astype(object)
                values.loc[swap] = r_values.loc[swap, values.columns].astype(object).to_numpy()
                values = values.infer_objects()
                used.loc[swap] = "retrieval_fallback"
    out = base.join(values, on="cycle_uid")
    # Колонка record_id/patient_id 7.1 — это cycle_uid с суффиксом точки; reconcile сам
    # выставляет patient_id из _key, поэтому здесь они не нужны.
    out = out.drop(columns=[c for c in ("patient_id", "record_id") if c in out.columns])
    uid = out["cycle_uid"]
    extra: dict[str, object] = {"DIGITAL TWIN": pd.to_numeric(out.get("clinical_probability"), errors="coerce")}
    for alias, source in ALIASES_70.items():
        extra[alias] = out[source] if source in out.columns else np.nan
    extra.update(dt_api_status=uid.map(meta["status"]), dt_api_error=uid.map(meta["error"]),
                 dt_api_seed=uid.map(meta["seed"]), dt_api_model_fingerprint=uid.map(meta["model_fingerprint"]),
                 dt_api_batch_id=uid.map(meta["batch_id"]), dt_api_observations=primary,
                 dt_timepoint_used=uid.map(used))

    if has_retrieval:
        # Воронка «прогноз против факта» — из точки после пункции (как 7.0), иначе
        # переданные модели бластоцисты возвращались бы как «прогноз».
        for name in FUNNEL_COLUMNS:
            if name in r_values.columns:
                extra[f"transfer_{name}"] = out[name] if name in out.columns else np.nan
                extra[name] = uid.map(r_values[name])
        extra["med_bl"] = extra.get("med_blasts", np.nan)
        extra["med_goodbl"] = extra.get("med_good", np.nan)
        for name in RETRIEVAL_SUMMARY:
            if name in r_values.columns:
                extra[f"retrieval_{name}"] = uid.map(r_values[name])
        extra["DIGITAL TWIN (retrieval)"] = pd.to_numeric(extra.get("retrieval_clinical_probability"), errors="coerce")
        extra.update(dt_api_retrieval_status=uid.map(r_meta["status"]),
                     dt_api_retrieval_error=uid.map(r_meta["error"]), dt_funnel_timepoint="retrieval")
    else:
        extra["dt_funnel_timepoint"] = primary
    # «Источник NN» (Teable, карточки): в 7.0 — текстовое описание ансамбля из CSV; 7.1 его
    # не отдаёт, поэтому пишется происхождение строки — служба и отпечаток модели.
    fingerprint = meta["model_fingerprint"]
    if has_retrieval:
        fingerprint = fingerprint.where(fingerprint.notna(), r_meta["model_fingerprint"].reindex(meta.index))
    extra["nn_source"] = uid.map(fingerprint).map(
        lambda fp: f"IVF Digital Twin 7.1 REST API | model_fingerprint={fp}" if isinstance(fp, str) and fp else None)

    for alias, source in ECHO_70.items():
        extra[alias] = model_source[source].to_numpy() if source in model_source.columns else np.nan
    # Подписи узлов «Похожие случаи»/«KAT vs PRAI» и журнал берут patient_name, который
    # батч 7.0 возвращал эхом из ФИО. В службу имя не уходит — подставляется здесь,
    # локально (результат пишется только в local_private и локальный отчёт).
    for source in ("patient_name", "Patient Full Name"):
        if source in model_source.columns:
            extra["patient_name"] = model_source[source].to_numpy()
            break
    extra["notes"] = "api7.1"
    out = out.drop(columns=[c for c in extra if c in out.columns])
    out = pd.concat([out, pd.DataFrame({k: (v.to_numpy() if isinstance(v, pd.Series) else v)
                                        for k, v in extra.items()}, index=out.index)], axis=1)
    return out.merge(flags, on="cycle_uid", how="left", validate="one_to_one")


# ------------------------------------------------------------------ orchestration

LEGACY_70_KEYS = ("run_cmd", "project_dir", "results_dir", "input_target", "launch", "mode")


def api_settings(dt_cfg: dict | None) -> dict:
    """Блок `api` секции digital_twin — единственный вход DT для всех клиник.

    Батч 7.0 (RUN_BATCH.bat) снят 2026-09-13: явный `source` другой версии — ошибка,
    а не тихий переход на API; оставшиеся ключи 7.0 только игнорируются с предупреждением.
    """
    dt_cfg = dt_cfg or {}
    source = dt_cfg.get("source", "api")
    if source != "api":
        raise DTApiError(f"digital_twin.source={source!r}: батч 7.0 снят, Digital Twin работает "
                         "только через API 7.1 — удалите source из config/machine.local.yaml")
    stale = [k for k in LEGACY_70_KEYS if dt_cfg.get(k) not in (None, "")]
    if stale:
        print(f"[dt-api] игнорируются ключи батча 7.0 в digital_twin: {', '.join(stale)}", file=sys.stderr)
    return dt_cfg.get("api") or {}


def run_digital_twin_api(model_source: pd.DataFrame, dt_cfg: dict, client: DTApiClient | None = None,
                         sleep=time.sleep) -> pd.DataFrame:
    api_cfg = api_settings(dt_cfg)
    client = client or DTApiClient.from_config(api_cfg)
    health = client.health()
    if health.get("status") != "ready":
        raise DTApiError(f"служба Digital Twin не готова: {health}")
    bodies, flags = build_batches(model_source, api_cfg)
    timepoints = tuple(api_cfg.get("timepoints") or TIMEPOINTS)
    results = []
    for body in bodies:
        created = client.submit(body)
        print(f"[dt-api] принят пакет {created['id'][:8]}: {created['total']} расчётов, "
              f"отклонено при проверке {created['invalid']}", file=sys.stderr)
        results.append(client.wait(created["id"], poll_seconds=float(api_cfg.get("poll_seconds", 15)),
                                   timeout_seconds=float(api_cfg.get("timeout_minutes", 360)) * 60,
                                   sleep=sleep))
    frame = to_dt_output(model_source, results, flags, timepoints,
                         fallback=bool(api_cfg.get("fallback_to_retrieval", True)))
    if not api_cfg.get("keep_batches", False):
        for batch in results:
            client.delete(batch["id"])
    bad = frame.loc[frame["dt_api_status"] != "complete", ["cycle_uid", "dt_api_status", "dt_api_error"]]
    for _, row in bad.iterrows():
        print(f"[dt-api] {row['cycle_uid']}: {row['dt_api_status']} {row['dt_api_error'] or ''}", file=sys.stderr)
    return frame


def check_digital_twin_api(model_source: pd.DataFrame, dt_cfg: dict, client: DTApiClient | None = None) -> dict:
    """Сухой прогон: те же пакеты, что ушли бы на расчёт, проверяются службой; ничего не считается.

    Возвращает сводку без идентификаторов, кроме cycle_uid отклонённых строк.
    """
    api_cfg = api_settings(dt_cfg)
    client = client or DTApiClient.from_config(api_cfg)
    health = client.health()
    if health.get("status") != "ready":
        raise DTApiError(f"служба Digital Twin не готова: {health}")
    bodies, flags = build_batches(model_source, api_cfg)
    by_timepoint: dict[str, dict] = {}
    invalid: list[dict] = []
    for body in bodies:
        out = client.validate(body)
        for row in out["rows"]:
            uid, _, timepoint = row["cycle_id"].rpartition(".")
            bucket = by_timepoint.setdefault(timepoint, {"valid": 0, "invalid": 0, "errors": {}})
            bucket[row["status"]] += 1
            if row["status"] == "invalid":
                code = row["error"]["code"] + (":" + ",".join(row["error"].get("fields") or []) if row["error"].get("fields") else "")
                bucket["errors"][code] = bucket["errors"].get(code, 0) + 1
                invalid.append({"cycle_uid": uid, "timepoint": timepoint, "error": row["error"]})
    n_calc = sum(b["valid"] for b in by_timepoint.values())
    seconds = float(api_cfg.get("seconds_per_calculation", 25))
    frame = flags.assign(dt_api_status="n/a")
    summary = run_summary(frame)
    # Покрытие DIGITAL TWIN с учётом запасной точки retrieval.
    bad = {}
    for item in invalid:
        bad.setdefault(item["cycle_uid"], set()).add(item["timepoint"])
    timepoints = tuple(api_cfg.get("timepoints") or TIMEPOINTS)
    fallback = bool(api_cfg.get("fallback_to_retrieval", True)) and "retrieval" in timepoints
    coverage = {"transfer": 0, "retrieval_fallback": 0, "none": 0}
    for uid in model_source["cycle_uid"]:
        failed = bad.get(uid, set())
        if timepoints[0] not in failed:
            coverage["transfer"] += 1
        elif fallback and "retrieval" not in failed:
            coverage["retrieval_fallback"] += 1
        else:
            coverage["none"] += 1
    return {"service": health, "cycles": int(len(model_source)), "batches": len(bodies),
            "timepoints": by_timepoint, "dt_coverage": coverage, "calculations": n_calc,
            "estimated_hours": round(n_calc * seconds / 3600, 1),
            "imputed": summary["imputed"], "afc_source": summary["afc_source"], "invalid": invalid}


def run_summary(frame: pd.DataFrame) -> dict:
    """Короткая сводка для run_manifest.json (без идентификаторов)."""
    def counts(column):
        return frame[column].value_counts(dropna=False).to_dict() if column in frame.columns else {}
    return {
        "source": "IVF Digital Twin 7.1 REST API (analytics connection)",
        "status": {str(k): int(v) for k, v in counts("dt_api_status").items()},
        "retrieval_status": {str(k): int(v) for k, v in counts("dt_api_retrieval_status").items()},
        # Какая точка дала DIGITAL TWIN строки: transfer / retrieval_fallback / none.
        "timepoint_used": {str(k): int(v) for k, v in counts("dt_timepoint_used").items()},
        "model_fingerprints": sorted({str(v) for v in frame.get("dt_api_model_fingerprint", pd.Series(dtype=object)).dropna()}),
        "imputed": {name: int(frame[name].fillna(False).astype(bool).sum())
                    for name in ("amh_imputed", "bmi_imputed", "attempt_imputed", "sperm_source_inferred",
                                 "okk_raised_for_thawed")
                    if name in frame.columns},
        "afc_source": {str(k): int(v) for k, v in counts("afc_source").items()},
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import getpass
    import yaml
    if sys.platform == "win32":
        # Консоль/перенаправленный вывод Windows иначе печатает кириллицу в cp1251.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Подключение embryo-lab-data к API IVF Digital Twin 7.1")
    sub = parser.add_subparsers(dest="command", required=True)
    store = sub.add_parser("store-key", help="сохранить ключ «Аналитика» под DPAPI этого пользователя")
    store.add_argument("--stdin", action="store_true", help="прочитать ключ из stdin, не спрашивая")
    sub.add_parser("health", help="проверить службу, сертификат и ключ")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    cfg = {}
    machine = root / "config" / "machine.local.yaml"
    if machine.is_file():
        cfg = (yaml.safe_load(machine.read_text(encoding="utf-8")) or {})
    api_cfg = ((cfg.get("external_tools") or {}).get("digital_twin") or {}).get("api") or {}
    if args.command == "store-key":
        key = sys.stdin.readline() if args.stdin else getpass.getpass("Ключ API (не отображается): ")
        print(f"ключ сохранён: {save_key(key, api_cfg.get('key_file'))}")
        return 0
    client = DTApiClient.from_config(api_cfg)
    print(json.dumps(client.health(), ensure_ascii=False))
    # Авторизованный запрос: проверяет ключ и тип подключения, данных пациентов не содержит.
    listing = client._call("GET", "/analytics/batches")[1]
    print(f"ключ принят, пакетов в службе: {len(listing['batches'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
