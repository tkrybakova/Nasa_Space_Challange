import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import re

# -----------------------------
# Функции очистки и преобразования
# -----------------------------

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip().replace(" ", "")
        if any(c.isalpha() for c in val) and not re.match(r"^\d+\.?\d*$", val):
            return np.nan
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val

def hms_to_deg(ra_str):
    try:
        c = SkyCoord(ra_str, unit=(u.hourangle, u.deg))
        return c.ra.deg
    except Exception:
        return np.nan

def dms_to_deg(dec_str):
    try:
        dec_clean = dec_str.replace("d", "°").replace("m", "'").replace("s", '"')
        c = SkyCoord("0h0m0s", dec_clean, unit=(u.hourangle, u.deg))
        return c.dec.deg
    except Exception:
        return np.nan

def estimate_mass_from_radius(r_earth):
    if pd.isna(r_earth) or r_earth <= 0:
        return np.nan
    if r_earth < 1.5:
        return r_earth ** 3.7
    elif r_earth < 4:
        return 2.7 * (r_earth ** 2.06)
    else:
        return 10 * (r_earth ** 0.88)

def classify_planet(r):
    if pd.isna(r):
        return "Unknown"
    if r < 1.25:
        return "Earth-like"
    elif r < 2.0:
        return "Super-Earth"
    elif r < 4.0:
        return "Sub-Neptune"
    else:
        return "Neptune+"

# -----------------------------
# Основная функция обработки
# -----------------------------

def load_and_process_data():
    # Загрузка CSV
    df_toi = pd.read_csv('exoplanet_api/data/dataset_toi.csv', encoding='latin1', on_bad_lines='skip')
    df_cum = pd.read_csv('exoplanet_api/data/dataset_cumulative.csv', header=None)
    df_k2 = pd.read_csv('exoplanet_api/data/dataset_k2.csv', encoding='latin1', on_bad_lines='skip')

    # Добавим имена колонок
    df_cum.columns = [
        "kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
        "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "koi_period", "koi_depth", "koi_duration", "koi_ror", "koi_snr",
        "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_tce_plnt_num",
        "koi_tce_delivname", "koi_steff", "koi_slogg", "koi_srad", "ra", "dec", "koi_kepmag"
    ]

    df_toi.columns = [
        "toi", "tid", "tfopwg_disp", "rastr", "ra", "decstr", "dec",
        "st_pmra", "st_pmdec", "pl_tranmid", "pl_orbper", "pl_rade", "pl_insol", "pl_eqt",
        "st_tmag", "st_dist", "st_teff", "st_logg", "st_rad",
        "sy_vmag", "sy_kmag", "sy_gaiamag", "rowupdate"
    ]

    # -----------------------------
    # Очистка числовых данных
    # -----------------------------
    cum_cols = ["koi_steff","koi_slogg","koi_srad","koi_prad","koi_insol","koi_kepmag","ra","dec","koi_period"]
    for col in cum_cols:
        df_cum[col] = df_cum[col].apply(clean_numeric)

    toi_cols = ["ra","dec","st_teff","st_logg","st_rad","pl_rade","pl_insol","pl_eqt","pl_orbper","st_dist","st_tmag"]
    for col in toi_cols:
        df_toi[col] = df_toi[col].apply(clean_numeric)

    df_toi["ra_deg"] = df_toi["rastr"].apply(hms_to_deg)
    df_toi["dec_deg"] = df_toi["decstr"].apply(dms_to_deg)

    k2_cols = ["ra","dec","st_teff","st_logg","st_rad","pl_rade","pl_insol","pl_orbper","sy_dist","sy_vmag"]
    for col in k2_cols:
        if col in df_k2.columns:
            df_k2[col] = df_k2[col].apply(clean_numeric)

    # -----------------------------
    # Создание унифицированных DataFrame
    # -----------------------------
    df_cum_clean = pd.DataFrame({
        "source":"Kepler",
        "name": df_cum["kepler_name"].fillna(df_cum["kepoi_name"]),
        "disposition": df_cum["koi_disposition"],
        "ra": df_cum["ra"],
        "dec": df_cum["dec"],
        "st_teff": df_cum["koi_steff"],
        "st_logg": df_cum["koi_slogg"],
        "st_rad": df_cum["koi_srad"],
        "pl_rade": df_cum["koi_prad"],
        "pl_insol": df_cum["koi_insol"],
        "pl_orbper": df_cum["koi_period"],
        "pl_eqt": df_cum["koi_teq"],
        "mag": df_cum["koi_kepmag"],
        "distance_pc": np.nan
    })

    df_toi_clean = pd.DataFrame({
        "source":"TESS",
        "name":"TOI-"+df_toi["toi"].astype(str),
        "disposition": df_toi["tfopwg_disp"].replace({
            "PC":"CANDIDATE","FP":"FALSE POSITIVE","KP":"CONFIRMED","CP":"CONFIRMED","APC":"CANDIDATE"
        }),
        "ra": df_toi["ra"],
        "dec": df_toi["dec"],
        "st_teff": df_toi["st_teff"],
        "st_logg": df_toi["st_logg"],
        "st_rad": df_toi["st_rad"],
        "pl_rade": df_toi["pl_rade"],
        "pl_insol": df_toi["pl_insol"],
        "pl_orbper": df_toi["pl_orbper"],
        "pl_eqt": df_toi["pl_eqt"],
        "mag": df_toi["st_tmag"],
        "distance_pc": df_toi["st_dist"]
    })

    df_k2_clean = pd.DataFrame({
        "source":"K2",
        "name": df_k2.get("pl_name", pd.Series([np.nan]*len(df_k2))),
        "disposition": df_k2.get("disposition", pd.Series([np.nan]*len(df_k2))),
        "ra": df_k2.get("ra", pd.Series([np.nan]*len(df_k2))),
        "dec": df_k2.get("dec", pd.Series([np.nan]*len(df_k2))),
        "st_teff": df_k2.get("st_teff", pd.Series([np.nan]*len(df_k2))),
        "st_logg": df_k2.get("st_logg", pd.Series([np.nan]*len(df_k2))),
        "st_rad": df_k2.get("st_rad", pd.Series([np.nan]*len(df_k2))),
        "pl_rade": df_k2.get("pl_rade", pd.Series([np.nan]*len(df_k2))),
        "pl_insol": df_k2.get("pl_insol", pd.Series([np.nan]*len(df_k2))),
        "pl_orbper": df_k2.get("pl_orbper", pd.Series([np.nan]*len(df_k2))),
        "pl_eqt": df_k2.get("pl_eqt", pd.Series([np.nan]*len(df_k2))),
        "mag": df_k2.get("sy_vmag", pd.Series([np.nan]*len(df_k2))),
        "distance_pc": df_k2.get("sy_dist", pd.Series([np.nan]*len(df_k2)))
    })

    # -----------------------------
    # Объединение и расширение
    # -----------------------------
    df_all = pd.concat([df_cum_clean, df_toi_clean, df_k2_clean], ignore_index=True)
    df_all = df_all.dropna(subset=["ra","dec"])

    df_all["ra_round"] = df_all["ra"].round(4)
    df_all["dec_round"] = df_all["dec"].round(4)
    df_all = df_all.drop_duplicates(subset=["ra_round","dec_round"], keep="first").drop(columns=["ra_round","dec_round"])

    df_all["flag_confirmed"] = (df_all["disposition"]=="CONFIRMED").astype(int)
    df_all["pl_masse"] = df_all["pl_rade"].apply(estimate_mass_from_radius)

    # Плотность
    earth_mass_kg = 5.972e24
    earth_radius_m = 6.371e6
    G = 6.67430e-11
    df_all["pl_density"] = np.where(
        (df_all["pl_rade"]>0) & (df_all["pl_masse"]>0),
        (df_all["pl_masse"]*earth_mass_kg)/((4/3)*np.pi*(df_all["pl_rade"]*earth_radius_m)**3)*1e-3,
        np.nan
    )

    df_all["stellar_flux_proxy"] = df_all["st_teff"]**4 * df_all["st_rad"]**2
    df_all["planet_class"] = df_all["pl_rade"].apply(classify_planet)

    output_cols = [
        "source","name","disposition","flag_confirmed","planet_class",
        "ra","dec","pl_rade","pl_masse","pl_density","pl_orbper","pl_insol","pl_eqt",
        "st_teff","st_logg","st_rad","stellar_flux_proxy","distance_pc","mag"
    ]

    df_final = df_all[output_cols]
    df_final.to_csv("exoplanet_api/data/combined_exoplanet_dataset_extended.csv", index=False)
    print(f"Готово! Итоговый датасет: {len(df_final)} строк, {len(df_final.columns)} колонок.")

    return df_final
