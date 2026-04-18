from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from battle_simulator import (
    BaseConfig,
    candidate_attacks,
    flatten_attack_config,
    flatten_base_config,
)
from train_model import MODEL_PATH, train_and_save_model


def load_or_train_model():
    if not MODEL_PATH.exists():
        model, _, _, _ = train_and_save_model()
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
        )
        sample_row = {}
        sample_row.update(flatten_attack_config(candidate_attacks()[0]))
        sample_row.update(flatten_base_config(sample_base))
        model.predict(pd.DataFrame([sample_row]))
        return model
    except Exception:
        model, _, _, _ = train_and_save_model()
        return model


def recommend_for_base(base: BaseConfig) -> pd.DataFrame:
    model = load_or_train_model()
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


def main() -> None:
    base = BaseConfig(
        base_level=14,
        anti_air_defense=8,
        splash_defense=7,
        wall_strength=8,
        inferno_strength=7,
        trap_pressure=6,
    )
    ranked = recommend_for_base(base)
    best = ranked.iloc[0]

    print("Best recommended attack strategy")
    print(f"Predicted win probability: {best['predicted_win_probability']:.2%}")
    print(f"Clan castle donation: {best['clan_castle']}")
    print(f"Siege machine: {best['siege_machine']}")

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
