"""Pipeline reproduzível para predição de atributos do solo por espectroscopia.

Exemplo:
    python pipeline_predicao_solo.py --spectra visnir --output-dir resultados_visnir

O pipeline usa os CSVs locais, cruza espectros e atributos por ID_Unique,
aplica transformações somente dentro do treino e salva resultados auditáveis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == "__main__":
    sys.modules.setdefault("pipeline_predicao_solo", sys.modules[__name__])

DEFAULT_TARGETS = [
    "Sand_gkg",
    "Clay_gkg",
    "SOM_gkg",
    "pH_H2O",
    "Ca_mmolkg",
    "CEC_Ph7_mmolkg",
]
SPECTRA_FILES = {
    "visnir": "VIS.NIR_DataSet_SP.csv",
    "nir": "NIR_DataSet_SP.csv",
    "mir": "MIR_DataSet_SP.csv",
}


class SpectralTransform(BaseEstimator, TransformerMixin):
    """Aplica SNV, primeira derivada e Savitzky-Golay sem alterar a largura."""

    def __init__(
        self,
        snv: bool = True,
        derivative: int = 0,
        window_length: int = 11,
        polyorder: int = 2,
    ) -> None:
        self.snv = snv
        self.derivative = derivative
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray, y: object = None) -> "SpectralTransform":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("A matriz espectral deve ser bidimensional.")
        if self.snv:
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True)
            values = (values - mean) / np.where(std == 0, 1, std)
        if self.derivative:
            window = min(self.window_length, values.shape[1])
            if window % 2 == 0:
                window -= 1
            minimum = self.polyorder + 2
            if window < minimum:
                raise ValueError(
                    "window_length precisa comportar o polyorder e ter tamanho ímpar."
                )
            values = savgol_filter(
                values,
                window_length=window,
                polyorder=self.polyorder,
                deriv=self.derivative,
                axis=1,
            )
        return values


SpectralTransform.__module__ = "pipeline_predicao_solo"


def _numeric_wavelength(column: object) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(column).strip())
    return float(match.group()) if match else None


def load_data(
    data_dir: Path, spectra_kind: str, targets: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    attributes_path = data_dir / "Atributos_SP.csv"
    spectra_path = data_dir / SPECTRA_FILES[spectra_kind]
    if not attributes_path.exists():
        raise FileNotFoundError(f"Arquivo de atributos não encontrado: {attributes_path}")
    if not spectra_path.exists():
        raise FileNotFoundError(f"Arquivo espectral não encontrado: {spectra_path}")

    attributes = pd.read_csv(attributes_path, low_memory=False)
    spectra = pd.read_csv(spectra_path, low_memory=False)
    for frame, name in ((attributes, "atributos"), (spectra, "espectros")):
        if "ID_Unique" not in frame.columns:
            raise ValueError(f"{name} precisa conter a coluna ID_Unique.")
        if frame["ID_Unique"].duplicated().any():
            raise ValueError(f"{name} contém IDs duplicados; a junção seria ambígua.")

    requested = list(targets)
    missing_targets = [target for target in requested if target not in attributes.columns]
    if missing_targets:
        raise ValueError(f"Alvos ausentes em Atributos_SP.csv: {missing_targets}")

    wavelength_columns = [
        column
        for column in spectra.columns
        if column != "ID_Unique"
        and _numeric_wavelength(column) is not None
    ]
    wavelength_columns.sort(key=lambda column: _numeric_wavelength(column) or 0)
    if len(wavelength_columns) < 10:
        raise ValueError("Menos de 10 colunas espectrais numéricas foram encontradas.")

    merged = attributes[["ID_Unique", *requested]].merge(
        spectra[["ID_Unique", *wavelength_columns]],
        on="ID_Unique",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Nenhum ID_Unique em comum entre atributos e espectros.")
    return merged[["ID_Unique", *requested]], merged[wavelength_columns], wavelength_columns


def build_model(
    n_components: int, derivative: int, snv: bool, window_length: int
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "spectral",
                SpectralTransform(
                    snv=snv,
                    derivative=derivative,
                    window_length=window_length,
                ),
            ),
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components, scale=False, max_iter=1000)),
        ]
    )


def choose_components(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: list[int],
    cv_splits: int,
    derivative: int,
    snv: bool,
    window_length: int,
    random_state: int,
) -> tuple[int, pd.DataFrame]:
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    records: list[dict[str, float | int]] = []
    values = y.to_numpy(dtype=float)
    for components in candidates:
        fold_rmse: list[float] = []
        for train_idx, valid_idx in kfold.split(X):
            model = build_model(components, derivative, snv, window_length)
            model.fit(X.iloc[train_idx], values[train_idx])
            prediction = model.predict(X.iloc[valid_idx]).ravel()
            fold_rmse.append(float(np.sqrt(mean_squared_error(values[valid_idx], prediction))))
        records.append(
            {
                "n_components": components,
                "cv_rmse_mean": float(np.mean(fold_rmse)),
                "cv_rmse_std": float(np.std(fold_rmse)),
            }
        )
    table = pd.DataFrame(records).sort_values("cv_rmse_mean")
    return int(table.iloc[0]["n_components"]), table


def calculate_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    return {
        "n_test": int(len(y_true)),
        "r2": float(r2_score(y_true, prediction)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "bias": float(np.mean(prediction - y_true)),
        "rpd": float(std / rmse) if rmse else float("inf"),
    }


def plot_spectral_behavior(
    X: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: Path,
    random_state: int,
) -> None:
    rng = np.random.default_rng(random_state)
    sample_size = min(100, len(X))
    sample = X.iloc[rng.choice(len(X), size=sample_size, replace=False)].to_numpy(float)
    mean = np.nanmean(sample, axis=0)
    std = np.nanstd(sample, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(wavelengths, sample.T, color="tab:blue", alpha=0.08)
    axes[0].plot(wavelengths, mean, color="black", linewidth=2, label="média")
    axes[0].fill_between(wavelengths, mean - std, mean + std, color="tab:blue", alpha=0.2)
    axes[0].set_title("Comportamentos espectrais: amostras, média e ±1 desvio-padrão")
    axes[0].set_ylabel("Reflectância / resposta")
    axes[0].legend()
    transformed = SpectralTransform(snv=True, derivative=1).fit_transform(
        SimpleImputer(strategy="median").fit_transform(sample)
    )
    transformed_mean = transformed.mean(axis=0)
    transformed_std = transformed.std(axis=0)
    axes[1].plot(wavelengths, transformed.T, color="tab:orange", alpha=0.08)
    axes[1].plot(wavelengths, transformed_mean, color="black", linewidth=2, label="média")
    axes[1].fill_between(
        wavelengths,
        transformed_mean - transformed_std,
        transformed_mean + transformed_std,
        color="tab:orange",
        alpha=0.2,
    )
    axes[1].set_title("SNV + primeira derivada")
    axes[1].set_xlabel("Comprimento de onda (nm ou cm⁻¹)")
    axes[1].set_ylabel("Resposta transformada")
    axes[1].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets or DEFAULT_TARGETS
    metadata, spectra, wavelength_columns = load_data(data_dir, args.spectra, targets)
    wavelength_values = np.array([_numeric_wavelength(c) for c in wavelength_columns], dtype=float)
    plot_spectral_behavior(spectra, wavelength_values, output_dir / "comportamentos_espectrais.png", args.seed)
    pd.DataFrame({"wavelength": wavelength_values}).to_csv(
        output_dir / "wavelengths.csv", index=False
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(metadata)), test_size=args.test_size, random_state=args.seed
    )
    metrics: list[dict[str, object]] = []
    predictions = metadata.iloc[test_idx][["ID_Unique"]].copy()
    for target in targets:
        valid = metadata[target].notna().to_numpy()
        train_valid = train_idx[valid[train_idx]]
        test_valid = test_idx[valid[test_idx]]
        if len(train_valid) < max(args.cv_splits + 1, 20) or len(test_valid) < 2:
            print(f"[AVISO] {target}: amostras insuficientes; alvo ignorado.")
            continue
        X_train = spectra.iloc[train_valid]
        y_train = pd.to_numeric(metadata.iloc[train_valid][target], errors="coerce")
        finite = y_train.notna().to_numpy()
        X_train, y_train = X_train.iloc[finite], y_train.iloc[finite]
        candidates = list(range(2, min(args.max_components, len(X_train) - 1, spectra.shape[1]) + 1))
        best, cv_table = choose_components(
            X_train,
            y_train,
            candidates,
            args.cv_splits,
            args.derivative,
            args.snv,
            args.window_length,
            args.seed,
        )
        model = build_model(best, args.derivative, args.snv, args.window_length)
        model.fit(X_train, y_train)
        y_test = pd.to_numeric(metadata.iloc[test_valid][target], errors="coerce").to_numpy(float)
        y_prediction = model.predict(spectra.iloc[test_valid]).ravel()
        result = calculate_metrics(y_test, y_prediction)
        result.update({"target": target, "best_components": best, "n_train": len(X_train)})
        metrics.append(result)
        prediction_column = np.full(len(metadata), np.nan)
        prediction_column[test_valid] = y_prediction
        predictions[f"{target}_observed"] = metadata.iloc[test_idx][target].to_numpy()
        predictions[f"{target}_predicted"] = prediction_column[test_idx]
        cv_table.to_csv(output_dir / f"cv_{target}.csv", index=False)
        joblib.dump(model, output_dir / f"modelo_{target}.joblib")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "metricas.csv", index=False)
    predictions.to_csv(output_dir / "predicoes_teste.csv", index=False)
    config = vars(args).copy()
    config.update(
        {
            "n_samples_joined": len(metadata),
            "n_spectral_variables": len(wavelength_columns),
            "wavelength_first": wavelength_values[0],
            "wavelength_last": wavelength_values[-1],
            "targets_completed": list(metrics_df["target"]) if not metrics_df.empty else [],
        }
    )
    (output_dir / "configuracao.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".", help="Pasta dos CSVs de entrada.")
    parser.add_argument("--spectra", choices=SPECTRA_FILES, default="visnir")
    parser.add_argument("--output-dir", default="resultados_pipeline")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=20)
    parser.add_argument("--derivative", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--window-length", type=int, default=11)
    parser.add_argument("--snv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run_pipeline(arguments)
    if result.empty:
        raise SystemExit("Nenhum modelo foi treinado.")
    print(result.to_string(index=False))
