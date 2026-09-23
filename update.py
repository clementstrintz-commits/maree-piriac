#!/usr/bin/env python3
"""Marée Piriac : récupère la hauteur d'eau mesurée (REFMAR / SHOM) et calcule
une prédiction harmonique à partir d'un an de mesures du même marégraphe.
Écrit site/data.json, lu par index.html.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "scripts":
    ROOT = os.path.dirname(ROOT)
with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
    CFG = json.load(f)
CACHE_DIR = os.path.join(ROOT, "cache")
SITE_DIR = os.path.join(ROOT, "site")
HARMO_FILE = os.path.join(CACHE_DIR, "harmoniques.json")
BASE = "https://services.data.shom.fr/maregraphie"
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
UA = {"User-Agent": "maree-piriac/1.0 (usage personnel)"}

# Vitesses angulaires des composantes (degrés / heure)
COMPOSANTES = {
    "SA": 0.0410686, "SSA": 0.0821373, "MM": 0.5443747, "MSF": 1.0158958,
    "MF": 1.0980331, "2Q1": 12.8542862, "Q1": 13.3986609, "O1": 13.9430356,
    "P1": 14.9589314, "K1": 15.0410686, "J1": 15.5854433, "OO1": 16.1391017,
    "2N2": 27.8953548, "MU2": 27.9682084, "N2": 28.4397295, "NU2": 28.5125831,
    "M2": 28.9841042, "LAM2": 29.4556253, "L2": 29.5284789, "T2": 29.9589333,
    "S2": 30.0, "K2": 30.0821373, "2SM2": 31.0158958, "M3": 43.4761563,
    "MK3": 44.0251729, "MN4": 57.4238337, "M4": 57.9682084, "MS4": 58.9841042,
    "S4": 60.0, "2MN6": 86.4079380, "M6": 86.9523127, "2MS6": 87.9682084,
    "M8": 115.9364166,
}
# Composantes qu'on ne sait pas séparer sur une série trop courte
TROP_PROCHES = [
    (330, ["SA", "T2"]),
    (180, ["SSA", "K2", "P1", "NU2", "LAM2", "MSF"]),
    (60, ["MM", "MF", "2Q1", "J1", "OO1", "MU2", "2N2", "L2", "2SM2"]),
]


def log(*a):
    print(*a, flush=True)


def http_get(url, essais=3):
    for i in range(essais):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            log(f"  échec {i + 1}/{essais} : {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Impossible de joindre {url}")


def heures(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def trouver_station():
    if CFG.get("station_id"):
        return int(CFG["station_id"]), CFG.get("station_nom", "")
    nom = CFG.get("station_nom", "CROISIC").upper()
    log(f"Recherche de l'identifiant du marégraphe « {nom} »…")
    xml = http_get(f"{BASE}/sos/service?request=GetCapabilities")
    blocs = re.split(r"ObservationOffering", xml)
    for bloc in blocs:
        if nom in bloc.upper():
            m = re.search(r"(?:procedure/|offering_)(\d+)", bloc)
            if m:
                log(f"  trouvé : identifiant {m.group(1)}")
                return int(m.group(1)), nom
    for m in re.finditer(nom, xml.upper()):
        fen = xml[max(0, m.start() - 3000): m.start()]
        ids = re.findall(r"(?:procedure/|offering_)(\d+)", fen)
        if ids:
            log(f"  trouvé (approx.) : identifiant {ids[-1]}")
            return int(ids[-1]), nom
    raise RuntimeError(
        f"Marégraphe « {nom} » introuvable. Renseignez station_id dans config.json."
    )


def telecharger(sid, debut, fin, sources):
    """Observations entre debut et fin (UTC), par tranches de 30 jours."""
    t_all, h_all = [], []
    cur = debut
    while cur < fin:
        nxt = min(cur + timedelta(days=30), fin)
        q = urllib.parse.urlencode({
            "sources": sources,
            "dtStart": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dtEnd": nxt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        try:
            data = json.loads(http_get(f"{BASE}/observation/json/{sid}?{q}")).get("data", [])
        except Exception as e:  # noqa: BLE001
            log(f"  tranche {cur:%Y-%m-%d} ignorée : {e}")
            data = []
        for d in data:
            v = d.get("value")
            if v is None:
                continue
            ts = datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            t_all.append(heures(ts))
            h_all.append(float(v))
        cur = nxt
    return np.array(t_all), np.array(h_all)


def moyenner(t, h, pas_h, filtre_median=True):
    """Moyenne par pas de temps, puis retrait des valeurs aberrantes."""
    if len(t) == 0:
        return t, h
    ok = (h > -2) & (h < 12)
    t, h = t[ok], h[ok]
    k = np.round(t / pas_h).astype(np.int64)
    uk, inv = np.unique(k, return_inverse=True)
    hm = np.bincount(inv, weights=h) / np.bincount(inv)
    tm = uk * pas_h
    if filtre_median and len(hm) >= 5:
        pad = np.pad(hm, 2, mode="edge")
        med = np.median(np.stack([pad[i:i + len(hm)] for i in range(5)]), axis=0)
        garde = np.abs(hm - med) < 0.4
        tm, hm = tm[garde], hm[garde]
    return tm, hm


def matrice(t, noms):
    cols = [np.ones_like(t)]
    for n in noms:
        w = np.radians(COMPOSANTES[n]) * t
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


def ajuster(sid):
    fin = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    debut = fin - timedelta(days=CFG.get("jours_analyse", 365))
    log(f"Analyse harmonique : téléchargement {debut:%Y-%m-%d} → {fin:%Y-%m-%d}…")
    t, h = telecharger(sid, debut, fin, "0")
    t, h = moyenner(t, h, 1.0, filtre_median=False)
    if len(t) < 24 * 30:
        raise RuntimeError(f"Trop peu de mesures pour l'analyse ({len(t)} heures).")
    duree = (t.max() - t.min()) / 24
    exclues = set()
    for seuil, noms in TROP_PROCHES:
        if duree < seuil:
            exclues.update(noms)
    noms = [n for n in COMPOSANTES if n not in exclues]
    coef, *_ = np.linalg.lstsq(matrice(t, noms), h, rcond=None)
    # 2e passe sans les heures aberrantes (capteur, fortes tempêtes)
    garde = np.abs(matrice(t, noms) @ coef - h) < 0.5
    t, h = t[garde], h[garde]
    coef, *_ = np.linalg.lstsq(matrice(t, noms), h, rcond=None)
    rms = float(np.sqrt(np.mean((matrice(t, noms) @ coef - h) ** 2)))
    log(f"  {len(t)} heures, {duree:.0f} jours, {len(noms)} composantes, écart type {rms:.3f} m")
    return {
        "station_id": sid,
        "mois": datetime.now(timezone.utc).strftime("%Y-%m"),
        "calcule_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duree_jours": round(duree),
        "ecart_type_m": round(rms, 3),
        "noms": noms,
        "coef": [float(c) for c in coef],
    }


def charger_harmoniques(sid):
    mois = datetime.now(timezone.utc).strftime("%Y-%m")
    if os.path.exists(HARMO_FILE):
        with open(HARMO_FILE, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("station_id") == sid and m.get("mois") == mois:
            return m
        log("Analyse harmonique à renouveler (nouveau mois).")
    m = ajuster(sid)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(HARMO_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f)
    return m


def predire(modele, t):
    return matrice(t, modele["noms"]) @ np.array(modele["coef"])


def ms(t_h):
    return int(round((EPOCH + timedelta(hours=float(t_h))).timestamp() * 1000))


def main():
    sid, nom = trouver_station()
    modele = charger_harmoniques(sid)
    now = datetime.now(timezone.utc)
    t_now = heures(now)

    t_obs, h_obs = np.array([]), np.array([])
    for src in ("1", "2", "0"):
        t_obs, h_obs = telecharger(sid, now - timedelta(hours=40), now + timedelta(hours=1), src)
        if len(t_obs):
            log(f"Mesures récentes : source {src}, {len(t_obs)} points")
            break
    t_obs, h_obs = moyenner(t_obs, h_obs, 5 / 60)

    pas = 10 / 60
    t_pred = np.arange(np.floor((t_now - 36) / pas) * pas, t_now + 60, pas)
    h_pred = predire(modele, t_pred)

    surcote = None
    if len(t_obs):
        recent = t_obs > t_now - 3
        if recent.sum() >= 6 and t_obs.max() > t_now - 2:
            surcote = float(np.mean(h_obs[recent] - predire(modele, t_obs[recent])))
            log(f"Écart mesure / prédiction : {surcote:+.2f} m")

    sortie = {
        "maj": now.isoformat(timespec="seconds"),
        "station": {"id": sid, "nom": nom},
        "surcote_m": None if surcote is None else round(surcote, 3),
        "analyse": {k: modele[k] for k in ("calcule_le", "duree_jours", "ecart_type_m")},
        "mesures": [[ms(t), round(float(h), 3)] for t, h in zip(t_obs, h_obs)],
        "prediction": [[ms(t), round(float(h), 3)] for t, h in zip(t_pred, h_pred)],
    }
    os.makedirs(SITE_DIR, exist_ok=True)
    with open(os.path.join(SITE_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, separators=(",", ":"))
    log(f"data.json écrit ({len(sortie['mesures'])} mesures, {len(sortie['prediction'])} prédictions)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"ERREUR : {e}")
        sys.exit(1)
