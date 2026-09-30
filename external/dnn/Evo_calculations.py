"""
Evo_calculations.py — независимая DNN-проверка (обычная DNN на TensorFlow) для
кросс-контроля Digital Twin и TabPFN. АДАПТИРОВАНО под пайплайн embryo-lab-data:
  * вход берётся из НАШЕГО источника (аргумент {input} — таблица из data/inbox),
    поддержаны и русские, и английские (OPU) имена колонок;
  * ПРИЗНАКИ считаются так же, как при обучении модели (FR = 2pN/инсеминированные,
    ЧОКК = ОКК/фолликулы) — НЕ по Вене; менять нельзя, иначе рассинхрон с обучением;
  * модель грузится по абсолютному пути (аргумент 2 / env DNN_MODEL_PATH);
  * scaler обучения загружается из сохранённого артефакта; fit на inference-батче запрещён;
  * выход атомарно пишется CSV туда, откуда читает наш пайплайн (аргумент 3).

Запуск (питоном СВОЕГО окружения с tensorflow):
  python Evo_calculations.py <input.xlsx|csv> [model_dir] [output.csv] <training_scaler.joblib>

Работает автономно — не зависит от пакета embryo (другое окружение/torch/tf).
"""
import os
import sys
import argparse
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Английские имена OPU -> русские имена, которых ждёт модель.
_ALIAS = {
    "Age": "Возраст", "# Attempt": "№ попытки", "N folicules OPU": "Количество фолликулов",
    "OCC": "Число ОКК", "Inseminated": "Число инсеминированных", "2pN": "2 pN",
    "Cleavage": "Число дробящихся на 3 день", "Bl": "Число Bl", "Good Bl": "Число Bl хор.кач-ва",
    "D5 embryos": "Число эмбрионов 5 дня", "Cryo": "Заморожено эмбрионов",
    "ET": "Перенесено эмбрионов", "Day of ET": "День переноса",
}
SELECTED = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов", "KPIScore",
]


def _kpi_score(row):
    s = 0
    s += 1 if row["Возраст"] >= 40 else (5 if row["Возраст"] <= 36 else 3)
    f = row["Количество фолликулов"]
    s += 5 if f > 15 else (3 if 8 <= f <= 15 else 1)
    ins = row["Число инсеминированных"]
    s += 1 if ins <= 3 else (3 if 4 <= ins <= 7 else 5)
    fr = row["Частота оплодотворения"]
    s += 1 if fr < 0.5 else (3 if fr <= 0.65 else 5)
    g = row["Число Bl хор.кач-ва"]
    s += 1 if g == 0 else (3 if g <= 2 else 5)
    return s


def prepare_features(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает вход, маппит колонки, считает признаки Evo. Возвращает (df, X[SELECTED])."""
    if str(input_path).lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        try:
            df = pd.read_excel(input_path, engine="calamine")
        except Exception:
            df = pd.read_excel(input_path, engine="openpyxl")

    # Only rename when the Russian target isn't already present: model_input.xlsx (this
    # project's own pipeline input) already carries both English and Russian column names
    # side by side, so an unconditional rename would collide into a duplicate-labeled
    # column and turn df.get(...) into a DataFrame instead of a Series.
    df = df.rename(columns={k: v for k, v in _ALIAS.items()
                            if k in df.columns and v not in df.columns})

    # признаки — ИМЕННО по формулам обучения Evo (не Вена)
    df["День переноса"] = pd.to_numeric(df.get("День переноса"), errors="coerce")
    for c in ["2 pN", "Число инсеминированных", "Число дробящихся на 3 день", "Число Bl",
              "Число Bl хор.кач-ва", "Число ОКК", "Количество фолликулов", "№ попытки"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Owner-directed: пустой номер попытки -> попытка №1 (не отмечено, значит первая;
    # повторные попытки, как правило, отмечаются явно). Только для этой необязательной
    # кросс-проверки, не меняет config/columns.yaml/основной пайплайн.
    if "№ попытки" in df.columns:
        df["№ попытки"] = df["№ попытки"].fillna(1)
    df["Частота оплодотворения"] = df["2 pN"] / df["Число инсеминированных"]
    df["Частота дробления"] = df["Число дробящихся на 3 день"] / df["2 pN"]
    df["Частота формирования бластоцист"] = df["Число Bl"] / df["2 pN"]
    df["Частота формирования бластоцист хорошего качества"] = df["Число Bl хор.кач-ва"] / df["2 pN"]
    df["Частота получения ОКК"] = df["Число ОКК"] / df["Количество фолликулов"]

    # Owner-directed approximation for this optional cross-check model only (не Вена,
    # не трогает основной пайплайн/config/columns.yaml): день переноса неизвестен ->
    # считаем D5-протокол; число эмбрионов 5 дня не отслежено отдельно -> берём число
    # дробящихся на 3 день как прокси.
    df["День переноса"] = df["День переноса"].fillna(5)
    if "Число эмбрионов 5 дня" not in df.columns and "Число дробящихся на 3 день" in df.columns:
        df["Число эмбрионов 5 дня"] = df["Число дробящихся на 3 день"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat = [c for c in SELECTED if c != "KPIScore"]
    # Stable feature contract: an absent column is not an observed zero.
    missing = [c for c in feat if c not in df.columns]
    if missing:
        raise ValueError(f"[dnn] отсутствуют признаки обучения: {missing}")
    for c in feat:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Owner-directed (2026-07-18, Fertimed source data): this MIS never writes an
    # explicit 0 for "Заморожено эмбрионов" — blank means nothing was frozen, not
    # unknown. Only for this optional cross-check's feature input (mirrors the
    # "№ попытки" fillna above); the main pipeline's frozen/analytics field keeps
    # NaN as a true missing value.
    if "Заморожено эмбрионов" in df.columns:
        df["Заморожено эмбрионов"] = df["Заморожено эмбрионов"].fillna(0)
    df["KPIScore"] = df.apply(_kpi_score, axis=1)
    X = df[SELECTED].copy()
    bad = X.isna()
    bad_rows = bad.any(axis=1)
    if bad_rows.any():
        # Пропуск строки, а не всего батча: ноль по-прежнему не используется как замена
        # пропуска (та же гарантия), но одна незаполненная строка больше не должна
        # блокировать проверку остальных. Отброшенные строки идентифицируются по
        # id_col ниже; здесь — по индексу и признаку, для диагностики в логе.
        details = {c: bad.index[bad[c]].tolist()[:10] for c in X.columns if bad[c].any()}
        print(f"[dnn] пропуск {int(bad_rows.sum())}/{len(df)} строк с пропусками в признаках "
              f"(ноль не используется как замена пропуска): {details}", file=sys.stderr)
        df = df.loc[~bad_rows].reset_index(drop=True)
        X = X.loc[~bad_rows].reset_index(drop=True)
    if df.empty:
        raise ValueError("[dnn] ни одной строки с полным набором признаков — проверка невозможна")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("[dnn] признаки содержат бесконечные значения")
    return df, X


def _load_scaler(path: str, feature_names: list[str]):
    if not path:
        raise ValueError("[dnn] требуется путь к scaler обучения (DNN_SCALER_PATH/аргумент 4)")
    scaler_path = Path(path)
    if not scaler_path.is_file():
        raise FileNotFoundError(f"[dnn] scaler обучения не найден: {scaler_path}")
    import joblib
    scaler = joblib.load(scaler_path)
    if not callable(getattr(scaler, "transform", None)):
        raise TypeError("[dnn] артефакт scaler не имеет метода transform")
    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(feature_names):
        raise ValueError(
            f"[dnn] scaler ожидает {n_features} признаков, контракт содержит {len(feature_names)}"
        )
    trained_names = getattr(scaler, "feature_names_in_", None)
    if trained_names is not None and list(map(str, trained_names)) != feature_names:
        raise ValueError("[dnn] порядок/имена признаков не совпадают с scaler обучения")
    return scaler


def _atomic_csv(frame: pd.DataFrame, target: str) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Independent IVF DNN verifier")
    # Named flags are the stable project contract; positionals retain old manual usage.
    parser.add_argument("legacy_input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_model", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_output", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_scaler", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--model", dest="model_dir")
    parser.add_argument("--output", dest="out_path")
    parser.add_argument("--scaler", dest="scaler_path")
    args = parser.parse_args()
    args.input_path = args.input_path or args.legacy_input
    args.model_dir = (args.model_dir or args.legacy_model
                      or os.environ.get("DNN_MODEL_PATH", ""))
    args.out_path = args.out_path or args.legacy_output or "evo_predictions.csv"
    args.scaler_path = (args.scaler_path or args.legacy_scaler
                        or os.environ.get("DNN_SCALER_PATH", ""))
    if not args.input_path:
        parser.error("--input is required")
    if not args.model_dir:
        parser.error("--model (or DNN_MODEL_PATH) is required")
    return args


def main():
    args = _parse_args()
    input_path = args.input_path
    model_dir = args.model_dir
    out_path = args.out_path
    scaler_path = args.scaler_path

    df, X = prepare_features(input_path)
    scaler = _load_scaler(scaler_path, SELECTED)
    # Transform only. Calling fit/fit_transform here would leak and invalidate verification.
    Xn = scaler.transform(X)
    if np.asarray(Xn).shape != X.shape:
        raise ValueError(f"[dnn] scaler вернул неверную форму {np.asarray(Xn).shape}, ожидается {X.shape}")

    import tensorflow as tf
    model = tf.keras.models.load_model(model_dir)
    input_shape = getattr(model, "input_shape", None)
    if input_shape is not None and int(input_shape[-1]) != X.shape[1]:
        raise ValueError(f"[dnn] модель ожидает {input_shape[-1]} признаков, получено {X.shape[1]}")
    proba = np.asarray(model.predict(Xn)).reshape(-1)
    if len(proba) != len(df):
        raise ValueError("[dnn] число предсказаний не совпадает с числом строк")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("[dnn] модель вернула невалидную вероятность вне [0, 1]")
    classes = (proba > 0.5).astype(int)

    # "_key" (ID+Date OPU composite) приоритетнее сырого ID/patient_id — так же, как в
    # tabpfn-адаптере: он различает повторные циклы одной пациентки, а join в
    # run_analysis.py (_join_dnn) ключуется именно по этому составному patient_id.
    id_col = ("_key" if "_key" in df.columns
              else "ID" if "ID" in df.columns
              else "patient_id" if "patient_id" in df.columns else None)
    out = pd.DataFrame({
        "patient_id": df[id_col].astype(str).values if id_col else range(len(df)),
        "Исход переноса EVO": classes,
        "Вероятность положительного исхода": proba,
        "dnn_validation_status": "valid_saved_scaler",
    })
    _atomic_csv(out, out_path)
    print(f"[dnn] предсказания -> {out_path}")
    print(f"ЧНБ_прогноз (EVO): {(classes == 1).mean():.2%}")


if __name__ == "__main__":
    main()
