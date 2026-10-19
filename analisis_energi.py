from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
IMG = BASE / "gambar"
MAX_TRAINING_ROWS = 100_000
RANDOM_STATE = 42
TRAINING_FILE = DATASET / "power-laws-forecasting-energy-consumption-training-data.csv"
WEATHER_FILE = DATASET / "power-laws-forecasting-energy-consumption-weather.csv"
METADATA_FILE = DATASET / "power-laws-forecasting-energy-consumption-metadata.csv"
HOLIDAYS_FILE = DATASET / "power-laws-forecasting-energy-consumption-holidays.csv"


def sample_training_data(path, max_rows):
    """Read the large training CSV in chunks and keep a uniform reservoir sample."""
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir = None
    rows_seen = 0

    for chunk in pd.read_csv(
        path,
        sep=";",
        usecols=["SiteId", "Timestamp", "Value"],
        chunksize=250_000,
        dtype={"SiteId": "int32", "Value": "float64"},
        parse_dates=["Timestamp"],
    ):
        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"], utc=True, errors="coerce"
        )

        if rows_seen < max_rows:
            initial_count = min(max_rows - rows_seen, len(chunk))
            initial = chunk.iloc[:initial_count]
            reservoir = (
                initial.copy()
                if reservoir is None
                else pd.concat([reservoir, initial], ignore_index=True)
            )
            rows_seen += initial_count
            chunk = chunk.iloc[initial_count:]

        if not chunk.empty:
            positions = np.arange(rows_seen + 1, rows_seen + len(chunk) + 1)
            replacements = rng.integers(0, positions)
            candidate_rows = np.flatnonzero(replacements < max_rows)
            if len(candidate_rows):
                last_row_by_slot = {}
                for slot, row_index in zip(
                    replacements[candidate_rows], candidate_rows
                ):
                    last_row_by_slot[int(slot)] = int(row_index)
                slots = list(last_row_by_slot)
                rows = list(last_row_by_slot.values())
                replacements = chunk.iloc[rows]
                for column in reservoir.columns:
                    column_index = reservoir.columns.get_loc(column)
                    reservoir.iloc[slots, column_index] = replacements[
                        column
                    ].to_numpy()
            rows_seen += len(chunk)

    if reservoir is None or reservoir.empty:
        raise ValueError(f"Tidak ada baris data yang dapat dibaca dari {path}.")

    reservoir["SiteId"] = reservoir["SiteId"].astype("int32")
    reservoir["Value"] = reservoir["Value"].astype("float64")
    reservoir["Timestamp"] = pd.to_datetime(
        reservoir["Timestamp"], utc=True, errors="coerce"
    )
    print(f"Baris training dibaca: {rows_seen:,}")
    print(f"Baris training yang dipakai (sampel acak): {len(reservoir):,}")
    return reservoir


def load_model_data():
    required_files = [
        TRAINING_FILE,
        WEATHER_FILE,
        METADATA_FILE,
        HOLIDAYS_FILE,
    ]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(f"- {path.relative_to(BASE)}" for path in missing_files)
        raise FileNotFoundError(
            "File dataset Power Laws berikut tidak ditemukan:\n"
            f"{missing}\nPastikan file berada di folder dataset/."
        )

    data = sample_training_data(TRAINING_FILE, MAX_TRAINING_ROWS)
    data = data.drop_duplicates().dropna(subset=["SiteId", "Timestamp", "Value"])
    data = data[data["Value"].ge(0)].copy()
    print("Baris sampel setelah pembersihan:", f"{len(data):,}")

    weather = pd.read_csv(
        WEATHER_FILE,
        sep=";",
        usecols=["Timestamp", "Temperature", "Distance", "SiteId"],
        dtype={"Temperature": "float32", "Distance": "float32", "SiteId": "int32"},
        parse_dates=["Timestamp"],
    )
    weather["Timestamp"] = pd.to_datetime(
        weather["Timestamp"], utc=True, errors="coerce"
    )
    weather = (
        weather.dropna(subset=["Timestamp", "Temperature", "Distance", "SiteId"])
        .drop_duplicates(["SiteId", "Timestamp"], keep="last")
        .sort_values("Timestamp")
    )

    data = pd.merge_asof(
        data.sort_values("Timestamp"),
        weather,
        on="Timestamp",
        by="SiteId",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    data["weather_available"] = data["Temperature"].notna().astype("int8")
    matched_weather = data["weather_available"].sum()
    print(
        "Sampel dengan data cuaca dalam jarak maksimum 1 jam:",
        f"{matched_weather:,} dari {len(data):,}",
    )
    site_weather_medians = weather.groupby("SiteId")[["Temperature", "Distance"]].median()
    for column in ["Temperature", "Distance"]:
        site_median = data["SiteId"].map(site_weather_medians[column])
        data[column] = (
            data[column]
            .fillna(site_median)
            .fillna(float(weather[column].median()))
            .astype("float32")
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        sep=";",
        usecols=["SiteId", "Surface", "Sampling", "BaseTemperature"],
        dtype={
            "SiteId": "int32",
            "Surface": "float32",
            "Sampling": "float32",
            "BaseTemperature": "float32",
        },
    ).drop_duplicates("SiteId")
    data = data.merge(metadata, on="SiteId", how="left", validate="many_to_one")

    holidays = pd.read_csv(
        HOLIDAYS_FILE,
        sep=";",
        usecols=["Date", "SiteId"],
        dtype={"SiteId": "int32"},
    )
    holidays["Date"] = pd.to_datetime(holidays["Date"], utc=True, errors="coerce")
    holidays = holidays.dropna(subset=["Date"]).drop_duplicates(["Date", "SiteId"])
    holidays["is_holiday"] = 1
    data["Date"] = data["Timestamp"].dt.floor("D")
    holidays["Date"] = holidays["Date"].dt.floor("D")
    data = data.merge(
        holidays[["Date", "SiteId", "is_holiday"]],
        on=["Date", "SiteId"],
        how="left",
        validate="many_to_one",
    )
    data["is_holiday"] = data["is_holiday"].fillna(0).astype("int8")

    features = [
        "SiteId",
        "hour",
        "day_of_week",
        "month",
        "Temperature",
        "Distance",
        "weather_available",
        "Surface",
        "Sampling",
        "BaseTemperature",
        "is_holiday",
    ]
    data["hour"] = data["Timestamp"].dt.hour
    data["day_of_week"] = data["Timestamp"].dt.dayofweek
    data["month"] = data["Timestamp"].dt.month
    required_columns = features + ["Value", "Timestamp"]
    rows_before_drop = len(data)
    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_columns)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )
    print(
        "Baris siap dimodelkan:",
        f"{len(data):,} (dihapus karena fitur wajib tidak lengkap: "
        f"{rows_before_drop - len(data):,})",
    )
    if len(data) < 10:
        raise ValueError(
            "Data siap model kurang dari 10 baris. Periksa kecocokan SiteId, "
            "waktu, data cuaca, dan metadata."
        )

    return data, features


def main():
    IMG.mkdir(exist_ok=True)
    print("=== MEMUAT DATASET POWER LAWS ===")
    data, features = load_model_data()
    print("Rentang waktu:", data["Timestamp"].min(), "sampai", data["Timestamp"].max())
    print("Jumlah SiteId:", data["SiteId"].nunique())

    hourly = data.groupby("hour", as_index=False)["Value"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(hourly["hour"], hourly["Value"], marker="o")
    plt.title("Rata-rata Konsumsi Energi per Jam")
    plt.xlabel("Jam")
    plt.ylabel("Value (satuan asli dataset)")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG / "tren_konsumsi_per_jam.png", dpi=160)
    plt.close()

    split_index = int(len(data) * 0.8)
    if split_index == 0 or split_index == len(data):
        raise ValueError("Jumlah data tidak cukup untuk pembagian train-test 80:20.")
    split_timestamp = data["Timestamp"].iloc[split_index]
    train = data[data["Timestamp"] < split_timestamp]
    test = data[data["Timestamp"] >= split_timestamp]
    if train.empty or test.empty:
        raise ValueError("Pembagian kronologis menghasilkan data train atau test kosong.")
    X_train, X_test = train[features], test[features]
    y_train, y_test = train["Value"], test["Value"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print("\n=== EVALUASI RANDOM FOREST ===")
    print(
        "Pemisahan data kronologis:",
        f"training sampai {train['Timestamp'].max()},",
        f"test mulai {test['Timestamp'].min()}.",
    )
    print(f"MAE  : {mae:,.3f} (satuan Value)")
    print(f"RMSE : {rmse:,.3f} (satuan Value)")
    print(f"R²   : {r2:.3f}")

    pd.DataFrame(
        {
            "timestamp": test["Timestamp"].astype(str).to_numpy(),
            "SiteId": test["SiteId"].to_numpy(),
            "value_aktual": y_test.to_numpy(),
            "value_prediksi": pred,
        }
    ).to_csv(BASE / "hasil_prediksi.csv", index=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.5, s=12)
    low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Value Aktual vs Prediksi Random Forest")
    plt.xlabel("Value aktual (satuan dataset)")
    plt.ylabel("Value prediksi (satuan dataset)")
    plt.tight_layout()
    plt.savefig(IMG / "aktual_vs_prediksi.png", dpi=160)
    plt.close()

    importance = pd.DataFrame(
        {"fitur": features, "importance": model.feature_importances_}
    )
    importance.sort_values("importance", ascending=False).to_csv(
        BASE / "feature_importance.csv", index=False
    )
    plot_importance = importance.sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_importance["fitur"], plot_importance["importance"])
    plt.title("Feature Importance Random Forest")
    plt.xlabel("Importance (relatif)")
    plt.tight_layout()
    plt.savefig(IMG / "feature_importance.png", dpi=160)
    plt.close()

    baseline = data[features].copy()
    warmer_scenario = baseline.copy()
    warmer_scenario["Temperature"] = warmer_scenario["Temperature"] + 2
    baseline_prediction = model.predict(baseline)
    warmer_prediction = model.predict(warmer_scenario)
    avg_baseline = baseline_prediction.mean()
    avg_warmer = warmer_prediction.mean()
    difference = avg_warmer - avg_baseline
    percent_difference = difference / avg_baseline * 100 if avg_baseline else 0
    print("\n=== SIMULASI ILUSTRATIF ===")
    print("Asumsi: suhu luar dinaikkan 2°C pada data sampel.")
    print(f"Rata-rata prediksi awal       : {avg_baseline:,.3f}")
    print(f"Rata-rata prediksi skenario   : {avg_warmer:,.3f}")
    print(
        f"Perubahan prediksi            : {difference:,.3f} "
        f"({percent_difference:.2f}%)"
    )
    print("Skenario model ini bukan bukti hubungan sebab-akibat.")
    pd.DataFrame(
        {
            "timestamp": data["Timestamp"].astype(str).to_numpy(),
            "SiteId": data["SiteId"].to_numpy(),
            "prediksi_awal": baseline_prediction,
            "prediksi_suhu_naik_2c": warmer_prediction,
            "perubahan_prediksi": warmer_prediction - baseline_prediction,
        }
    ).to_csv(BASE / "hasil_simulasi_skenario.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Suhu aktual", "Suhu naik 2°C"],
        [avg_baseline, avg_warmer],
    )
    plt.title("Perbandingan Prediksi Skenario Suhu")
    plt.ylabel("Rata-rata prediksi (satuan dataset)")
    plt.tight_layout()
    plt.savefig(IMG / "simulasi_penghematan.png", dpi=160)
    plt.close()
    print("\nOutput tersimpan: folder gambar/ dan file CSV hasil.")
    print("Selesai.")


if __name__ == "__main__":
    main()
