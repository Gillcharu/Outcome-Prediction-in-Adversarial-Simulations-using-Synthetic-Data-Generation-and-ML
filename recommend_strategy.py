from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from battle_simulator import (
    AttackConfig,
    BaseConfig,
    DEFENSE_TYPES,
    GUARDIAN_TYPES,
    HERO_TYPES,
    PET_TYPES,
    SPELL_TYPES,
    TROOP_TYPES,
    candidate_attacks,
    aggregate_base_metrics,
    flatten_attack_config,
    flatten_base_config,
    mutate_attack_config,
)
from train_model import METRICS_PATH, MODEL_PATH, train_and_save_model
from train_model import FEATURE_IMPORTANCE_PATH


def load_or_train_model():
    if not MODEL_PATH.exists():
        model, _, _ = train_and_save_model()
        return model

    model = joblib.load(MODEL_PATH)
    try:
        sample_defenses = {name: 5 for name in DEFENSE_TYPES}
        sample_metrics = aggregate_base_metrics(sample_defenses)
        sample_base = BaseConfig(
            base_level=14,
            anti_air_defense=sample_metrics["anti_air_defense"],
            ground_pressure=sample_metrics["ground_pressure"],
            splash_defense=sample_metrics["splash_defense"],
            wall_strength=sample_metrics["wall_strength"],
            inferno_strength=sample_metrics["inferno_strength"],
            trap_pressure=sample_metrics["trap_pressure"],
            defenses=sample_defenses,
        )
        sample_row = {}
        sample_row.update(flatten_attack_config(candidate_attacks()[0]))
        sample_row.update(flatten_base_config(sample_base))
        model.predict(pd.DataFrame([sample_row]))
        return model
    except Exception:
        model, _, _ = train_and_save_model()
        return model


def load_feature_importance(limit: int = 10) -> list[dict[str, float | str]]:
    if not FEATURE_IMPORTANCE_PATH.exists():
        _, _, _ = train_and_save_model()
    frame = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(limit).copy()
    frame["feature"] = frame["feature"].str.replace("categorical__", "", regex=False).str.replace("numeric__", "", regex=False)
    return [
        {"feature": str(row["feature"]), "importance": float(row["importance"])}
        for _, row in frame.iterrows()
    ]


def _evaluate_candidates(candidates: list[AttackConfig], base: BaseConfig, model) -> pd.DataFrame:
    rows = []
    for attack in candidates:
        row = {}
        row.update(flatten_attack_config(attack))
        row.update(flatten_base_config(base))
        rows.append(row)
    candidate_df = pd.DataFrame(rows)
    candidate_df["predicted_win_probability"] = model.predict(candidate_df)
    return candidate_df


def _expand_candidates(seed_rows: pd.DataFrame) -> list[AttackConfig]:
    expanded: list[AttackConfig] = []
    clan_options = ["cc_yeti", "cc_balloon", "cc_dragon", "cc_ice_golem"]
    siege_options = [
        "wall_wrecker",
        "battle_blimp",
        "stone_slammer",
        "siege_barracks",
        "log_launcher",
        "flame_flinger",
        "battle_drill",
    ]

    for row_index, (_, row) in enumerate(seed_rows.iterrows()):
        attack = AttackConfig(
            troops={name: int(row[f"troop_{name}"]) for name in TROOP_TYPES},
            spells={name: int(row[f"spell_{name}"]) for name in SPELL_TYPES},
            heroes={name: int(row[f"hero_{name}"]) for name in HERO_TYPES},
            pets={name: int(row[f"pet_{name}"]) for name in PET_TYPES},
            guardians={name: int(row[f"guardian_{name}"]) for name in GUARDIAN_TYPES},
            clan_castle=str(row["clan_castle"]),
            siege_machine=str(row["siege_machine"]),
        )
        expanded.append(attack)
        for variant_index in range(4):
            variant = mutate_attack_config(attack, row_index * 4 + variant_index)
            expanded.append(variant)
            expanded.append(
                AttackConfig(
                    troops=variant.troops,
                    spells=variant.spells,
                    heroes=variant.heroes,
                    pets=variant.pets,
                    guardians=variant.guardians,
                    clan_castle=clan_options[(row_index + variant_index) % len(clan_options)],
                    siege_machine=siege_options[(row_index + variant_index) % len(siege_options)],
                )
            )
    return expanded


def recommend_for_base(base: BaseConfig, model=None) -> pd.DataFrame:
    model = model or load_or_train_model()

    baseline_df = _evaluate_candidates(candidate_attacks(), base, model)
    seed_rows = baseline_df.sort_values("predicted_win_probability", ascending=False).head(12)
    refined_df = _evaluate_candidates(_expand_candidates(seed_rows), base, model)

    ranked = pd.concat([baseline_df, refined_df], ignore_index=True)
    ranked = ranked.drop_duplicates(subset=[column for column in ranked.columns if column != "predicted_win_probability"])
    ranked = ranked.sort_values("predicted_win_probability", ascending=False)
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
    base_defenses = {name: 5 for name in DEFENSE_TYPES}
    base_metrics = aggregate_base_metrics(base_defenses)
    base = BaseConfig(
        base_level=14,
        anti_air_defense=base_metrics["anti_air_defense"],
        ground_pressure=base_metrics["ground_pressure"],
        splash_defense=base_metrics["splash_defense"],
        wall_strength=base_metrics["wall_strength"],
        inferno_strength=base_metrics["inferno_strength"],
        trap_pressure=base_metrics["trap_pressure"],
        defenses=base_defenses,
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
