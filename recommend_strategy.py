from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from battle_simulator import (
    BaseConfig,
    DEFENSE_TYPES,
    GUARDIAN_TYPES,
    HERO_TYPES,
    PET_TYPES,
    SPELL_TYPES,
    TROOP_TYPES,
    candidate_attacks,
    flatten_attack_config,
    flatten_base_config,
)
from train_model import METRICS_PATH, MODEL_PATH, train_and_save_model


def load_or_train_model():
    if not MODEL_PATH.exists():
        model, _, _ = train_and_save_model()
        return model

    model = joblib.load(MODEL_PATH)
    try:
        sample_base = BaseConfig(
            base_level=14,
            anti_air_defense=7,
            splash_defense=6,
            wall_strength=7,
            inferno_strength=7,
            trap_pressure=5,
            defenses={name: 5 for name in DEFENSE_TYPES},
        )
        sample_row = {}
        sample_row.update(flatten_attack_config(candidate_attacks()[0]))
        sample_row.update(flatten_base_config(sample_base))
        model.predict(pd.DataFrame([sample_row]))
        return model
    except Exception:
        model, _, _ = train_and_save_model()
        return model


def recommend_for_base(base: BaseConfig, model=None) -> pd.DataFrame:
    model = model or load_or_train_model()
    candidates = candidate_attacks()

    rows = []
    for attack in candidates:
        row = {}
        row.update(flatten_attack_config(attack))
        row.update(flatten_base_config(base))
        rows.append(row)

    candidate_df = pd.DataFrame(rows)
    candidate_df["predicted_win_probability"] = model.predict(candidate_df)
    ranked = candidate_df.sort_values("predicted_win_probability", ascending=False)
    return ranked


def load_metrics() -> dict[str, float]:
    if not METRICS_PATH.exists():
        _, _, metrics = train_and_save_model()
        return metrics
    import json

    return json.loads(METRICS_PATH.read_text())


def attack_archetype(row: pd.Series) -> str:
    air = sum(float(row.get(f"troop_{name}", 0)) for name in ["balloon", "dragon", "baby_dragon", "electro_dragon", "dragon_rider", "minion", "lava_hound"])
    ground = sum(float(row.get(f"troop_{name}", 0)) for name in ["pekka", "miner", "root_rider", "hog_rider", "golem", "valkyrie", "giant", "yeti", "electro_titan"])
    support = sum(float(row.get(f"troop_{name}", 0)) for name in ["healer", "witch", "bowler", "apprentice_warden", "druid"])
    if air > ground * 1.15:
        return "Air-heavy assault"
    if ground > air * 1.15:
        return "Ground-heavy push"
    return "Hybrid pressure composition"


def top_units(row: pd.Series, prefix: str, keys: list[str], count: int = 3) -> list[str]:
    values = [(key, float(row.get(f"{prefix}_{key}", 0))) for key in keys]
    ranked = [key for key, value in sorted(values, key=lambda item: item[1], reverse=True) if value > 0]
    return ranked[:count]


def build_recommendation_reasons(row: pd.Series) -> list[str]:
    reasons = []
    archetype = attack_archetype(row)
    reasons.append(f"This is a {archetype.lower()} built around {', '.join(top_units(row, 'troop', TROOP_TYPES, 2)) or 'balanced units'}.")

    if row.get("defense_air_defence", 0) >= 7 and archetype.startswith("Ground"):
        reasons.append("It avoids overcommitting to air units against a strong anti-air base.")
    elif row.get("defense_wall", 0) >= 7 and row.get("siege_machine") in {"wall_wrecker", "log_launcher", "battle_drill"}:
        reasons.append("The selected siege machine helps break through high wall pressure.")
    elif row.get("defense_scattershot", 0) >= 7:
        reasons.append("The composition spreads pressure instead of relying on fragile clustered troops into splash-heavy defenses.")

    major_spells = top_units(row, "spell", SPELL_TYPES, 2)
    if major_spells:
        reasons.append(f"Primary spell support comes from {', '.join(major_spells)}, which fits this matchup.")

    major_heroes = top_units(row, "hero", HERO_TYPES, 2)
    if major_heroes:
        reasons.append(f"High-impact hero support is driven by {', '.join(major_heroes)}.")

    return reasons[:4]


def strategy_snapshot(row: pd.Series) -> dict[str, Any]:
    return {
        "archetype": attack_archetype(row),
        "reasons": build_recommendation_reasons(row),
    }


def main() -> None:
    base = BaseConfig(
        base_level=14,
        anti_air_defense=8,
        splash_defense=7,
        wall_strength=8,
        inferno_strength=7,
        trap_pressure=6,
        defenses={name: 5 for name in DEFENSE_TYPES},
    )
    ranked = recommend_for_base(base)
    best = ranked.iloc[0]

    print("Best recommended attack strategy")
    print(f"Predicted win probability: {best['predicted_win_probability']:.2%}")
    print(f"Clan castle donation: {best['clan_castle']}")
    print(f"Siege machine: {best['siege_machine']}")
    print(f"Archetype: {strategy_snapshot(best)['archetype']}")

    print("\nTroops")
    for column, value in best.items():
        if column.startswith("troop_") and value > 0:
            print(f"  {column.replace('troop_', '')}: {int(value)}")

    print("\nSpells")
    for column, value in best.items():
        if column.startswith("spell_") and value > 0:
            print(f"  {column.replace('spell_', '')}: {int(value)}")

    print("\nHeroes")
    for column, value in best.items():
        if column.startswith("hero_") and value > 0:
            print(f"  {column.replace('hero_', '')}: {int(value)}")

    print("\nPets")
    for column, value in best.items():
        if column.startswith("pet_") and value > 0:
            print(f"  {column.replace('pet_', '')}: {int(value)}")

    print("\nGuardians")
    for column, value in best.items():
        if column.startswith("guardian_") and value > 0:
            print(f"  {column.replace('guardian_', '')}: {int(value)}")


if __name__ == "__main__":
    main()
