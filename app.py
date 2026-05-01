from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from battle_simulator import (
    BaseConfig,
    CLAN_CASTLE_TYPES,
    DEFENSE_TYPES,
    GUARDIAN_TYPES,
    HERO_TYPES,
    PET_TYPES,
    SIEGE_TYPES,
    SPELL_TYPES,
    TROOP_TYPES,
    aggregate_base_metrics,
    flatten_base_config,
)
from recommend_strategy import load_or_train_model, recommend_for_base


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "attack-strategy-local-secret")


SPECIAL_DISPLAY_NAMES = {
    "wall_breaker": "Wall Breaker",
    "baby_dragon": "Baby Dragon",
    "electro_dragon": "Electro Dragon",
    "dragon_rider": "Dragon Rider",
    "electro_titan": "Electro Titan",
    "root_rider": "Root Rider",
    "meteor_golem": "Meteor Golem",
    "hog_rider": "Hog Rider",
    "lava_hound": "Lava Hound",
    "ice_golem": "Ice Golem",
    "apprentice_warden": "Apprentice Warden",
    "barbarian_king": "Barbarian King",
    "archer_queen": "Archer Queen",
    "minion_prince": "Minion Prince",
    "grand_warden": "Grand Warden",
    "royal_champion": "Royal Champion",
    "dragon_duke": "Dragon Duke",
    "electro_owl": "Electro Owl",
    "mighty_yak": "Mighty Yak",
    "ground_guardian": "Ground Guardian",
    "air_guardian": "Air Guardian",
    "healing_guardian": "Healing Guardian",
    "cc_yeti": "Yeti",
    "cc_balloon": "Balloons",
    "cc_dragon": "Dragon",
    "cc_ice_golem": "Ice Golem",
    "wall_wrecker": "Wall Wrecker",
    "battle_blimp": "Battle Blimp",
    "stone_slammer": "Stone Slammer",
    "log_launcher": "Log Launcher",
    "flame_flinger": "Flame Flinger",
    "air_defence": "Air Defence",
    "air_sweeper": "Air Sweeper",
    "hidden_tesla": "Hidden Tesla",
    "bomb_tower": "Bomb Tower",
    "xbow": "X-Bow",
    "inferno_tower": "Inferno Tower",
    "eagle_artillery": "Eagle Artillery",
    "spell_tower": "Spell Tower",
    "lassi": "L.A.S.S.I",
}


def pretty_name(identifier: str) -> str:
    return SPECIAL_DISPLAY_NAMES.get(identifier, identifier.replace("_", " ").title())


DISPLAY_NAMES = {
    identifier: pretty_name(identifier)
    for identifier in (
        TROOP_TYPES
        + SPELL_TYPES
        + HERO_TYPES
        + PET_TYPES
        + GUARDIAN_TYPES
        + CLAN_CASTLE_TYPES
        + SIEGE_TYPES
        + DEFENSE_TYPES
    )
}


BASE_PRESETS = {
    "anti_air_fortress": {
        "label": "Anti-Air Fortress",
        "description": "Strong anti-air, air sweeper pressure, and heavy single-target defenses.",
        "values": {
            "defense_air_defence": 9,
            "defense_air_sweeper": 8,
            "defense_archer_tower": 8,
            "defense_hidden_tesla": 8,
            "defense_xbow": 8,
            "defense_inferno_tower": 8,
            "defense_spell_tower": 7,
            "defense_monolith": 7,
            "defense_traps": 7,
        },
    },
    "ring_base": {
        "label": "Ring Base",
        "description": "High wall pressure with dangerous splash and central core defenses.",
        "values": {
            "defense_wall": 9,
            "defense_mortar": 7,
            "defense_wizard_tower": 8,
            "defense_bomb_tower": 8,
            "defense_xbow": 8,
            "defense_inferno_tower": 7,
            "defense_eagle_artillery": 7,
            "defense_scattershot": 8,
            "defense_monolith": 7,
        },
    },
    "trap_heavy": {
        "label": "Trap Heavy",
        "description": "Moderate towers with extra traps, Tesla pressure, and hard punishment on pathing mistakes.",
        "values": {
            "defense_hidden_tesla": 8,
            "defense_bomb_tower": 7,
            "defense_spell_tower": 7,
            "defense_traps": 10,
            "defense_cannon": 7,
            "defense_archer_tower": 7,
            "defense_wizard_tower": 7,
            "defense_xbow": 6,
        },
    },
}


def build_default_form() -> dict[str, int | str]:
    defaults: dict[str, int | str] = {}

    troop_defaults = {
        "barbarian": 10,
        "archer": 12,
        "wizard": 8,
        "giant": 6,
        "wall_breaker": 4,
        "balloon": 6,
        "healer": 4,
        "dragon": 4,
        "pekka": 3,
        "miner": 8,
        "hog_rider": 6,
        "witch": 4,
    }
    for troop in TROOP_TYPES:
        defaults[f"troop_{troop}"] = troop_defaults.get(troop, 0)

    spell_defaults = {
        "rage": 2,
        "heal": 2,
        "freeze": 2,
        "lightning": 1,
        "jump": 1,
    }
    for spell in SPELL_TYPES:
        defaults[f"spell_{spell}"] = spell_defaults.get(spell, 0)

    hero_defaults = {
        "barbarian_king": 80,
        "archer_queen": 82,
        "minion_prince": 55,
        "grand_warden": 60,
        "royal_champion": 32,
        "dragon_duke": 30,
    }
    for hero in HERO_TYPES:
        defaults[f"hero_{hero}"] = hero_defaults.get(hero, 0)

    pet_defaults = {"lassi": 7, "electro_owl": 8, "mighty_yak": 7, "unicorn": 9}
    for pet in PET_TYPES:
        defaults[f"pet_{pet}"] = pet_defaults.get(pet, 0)

    guardian_defaults = {"ground_guardian": 6, "air_guardian": 6, "healing_guardian": 7}
    for guardian in GUARDIAN_TYPES:
        defaults[f"guardian_{guardian}"] = guardian_defaults.get(guardian, 0)

    defense_defaults = {defense: 5 for defense in DEFENSE_TYPES}
    defense_defaults.update(
        {
            "wall": 7,
            "air_defence": 7,
            "wizard_tower": 6,
            "hidden_tesla": 6,
            "xbow": 6,
            "inferno_tower": 6,
            "traps": 5,
        }
    )
    for defense in DEFENSE_TYPES:
        defaults[f"defense_{defense}"] = defense_defaults[defense]

    defaults["clan_castle"] = "cc_yeti"
    defaults["siege_machine"] = "log_launcher"
    defaults["base_level"] = 15
    return defaults


DEFAULT_FORM = build_default_form()
SESSION_KEY = "planner_form"


def parse_int(form: Dict[str, str], key: str, fallback: int) -> int:
    try:
        return int(form.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def extract_inputs(form: Dict[str, str]) -> tuple[dict[str, int | float | str], BaseConfig]:
    payload: dict[str, int | float | str] = {}

    for troop in TROOP_TYPES:
        key = f"troop_{troop}"
        payload[key] = parse_int(form, key, int(DEFAULT_FORM[key]))

    for spell in SPELL_TYPES:
        key = f"spell_{spell}"
        payload[key] = parse_int(form, key, int(DEFAULT_FORM[key]))

    for hero in HERO_TYPES:
        key = f"hero_{hero}"
        payload[key] = parse_int(form, key, int(DEFAULT_FORM[key]))

    for pet in PET_TYPES:
        key = f"pet_{pet}"
        payload[key] = parse_int(form, key, int(DEFAULT_FORM[key]))

    for guardian in GUARDIAN_TYPES:
        key = f"guardian_{guardian}"
        payload[key] = parse_int(form, key, int(DEFAULT_FORM[key]))

    clan_castle = form.get("clan_castle", str(DEFAULT_FORM["clan_castle"]))
    if clan_castle not in CLAN_CASTLE_TYPES:
        clan_castle = str(DEFAULT_FORM["clan_castle"])
    payload["clan_castle"] = clan_castle

    siege_machine = form.get("siege_machine", str(DEFAULT_FORM["siege_machine"]))
    if siege_machine not in SIEGE_TYPES:
        siege_machine = str(DEFAULT_FORM["siege_machine"])
    payload["siege_machine"] = siege_machine

    defenses = {
        defense: parse_int(form, f"defense_{defense}", int(DEFAULT_FORM[f"defense_{defense}"]))
        for defense in DEFENSE_TYPES
    }
    metrics = aggregate_base_metrics(defenses)

    base = BaseConfig(
        base_level=parse_int(form, "base_level", int(DEFAULT_FORM["base_level"])),
        anti_air_defense=metrics["anti_air_defense"],
        splash_defense=metrics["splash_defense"],
        wall_strength=metrics["wall_strength"],
        inferno_strength=metrics["inferno_strength"],
        trap_pressure=metrics["trap_pressure"],
        defenses=defenses,
    )
    payload.update(flatten_base_config(base))
    return payload, base


def predict_probability(model, payload: dict[str, int | float | str]) -> float:
    prediction = model.predict(pd.DataFrame([payload]))[0]
    return max(0.0, min(1.0, float(prediction)))


def summarize_payload(payload: dict[str, int | float | str]) -> dict[str, Any]:
    def collect(prefix: str, keys: list[str]) -> list[dict[str, int]]:
        values = []
        for key in keys:
            value = int(payload[f"{prefix}_{key}"])
            if value > 0:
                values.append({"name": DISPLAY_NAMES[key], "value": value})
        return values

    return {
        "clan_castle": DISPLAY_NAMES[str(payload["clan_castle"])],
        "siege_machine": DISPLAY_NAMES[str(payload["siege_machine"])],
        "troops": collect("troop", TROOP_TYPES),
        "spells": collect("spell", SPELL_TYPES),
        "heroes": collect("hero", HERO_TYPES),
        "pets": collect("pet", PET_TYPES),
        "guardians": collect("guardian", GUARDIAN_TYPES),
    }


def format_plan(row: pd.Series) -> dict[str, Any]:
    def collect(prefix: str, keys: list[str]) -> list[dict[str, int]]:
        values = []
        for key in keys:
            value = int(row[f"{prefix}_{key}"])
            if value > 0:
                values.append({"name": DISPLAY_NAMES[key], "value": value})
        return values

    return {
        "probability": float(row["predicted_win_probability"]),
        "clan_castle": DISPLAY_NAMES[str(row["clan_castle"])],
        "siege_machine": DISPLAY_NAMES[str(row["siege_machine"])],
        "troops": collect("troop", TROOP_TYPES),
        "spells": collect("spell", SPELL_TYPES),
        "heroes": collect("hero", HERO_TYPES),
        "pets": collect("pet", PET_TYPES),
        "guardians": collect("guardian", GUARDIAN_TYPES),
    }


def summarize_defenses(base: BaseConfig) -> list[dict[str, int]]:
    return [
        {"name": DISPLAY_NAMES[defense], "value": base.defenses[defense]}
        for defense in DEFENSE_TYPES
        if base.defenses.get(defense, 0) > 0
    ]


def build_chart_data(prediction: float, top_recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ["Your Army"]
    values = [round(prediction * 100, 1)]
    for index, item in enumerate(top_recommendations, start=1):
        labels.append(f"Option {index}")
        values.append(round(item["probability"] * 100, 1))
    return {"labels": labels, "values": values}


def build_insights(prediction: float, recommendation: dict[str, Any] | None) -> list[str]:
    if recommendation is None:
        return []
    uplift = round((recommendation["probability"] - prediction) * 100, 1)
    return [
        f"Your current army is predicted at {prediction * 100:.1f}% win probability.",
        f"The best recommended plan reaches {recommendation['probability'] * 100:.1f}%, which is {uplift:.1f} points higher.",
        f"The recommended setup uses {recommendation['clan_castle']} and {recommendation['siege_machine']} to better match this base.",
    ]


def run_analysis(form: Dict[str, str]) -> dict[str, Any]:
    payload, base = extract_inputs(form)
    model = load_or_train_model()
    prediction = predict_probability(model, payload)

    ranked = recommend_for_base(base).head(3)
    top_recommendations = [format_plan(row) for _, row in ranked.iterrows()]
    recommendation = top_recommendations[0] if top_recommendations else None

    return {
        "prediction": prediction,
        "recommendation": recommendation,
        "top_recommendations": top_recommendations,
        "chart_data": build_chart_data(prediction, top_recommendations),
        "insights": build_insights(prediction, recommendation),
        "army_summary": summarize_payload(payload),
        "defense_summary": summarize_defenses(base),
        "base_summary": flatten_base_config(base),
    }


def get_form_values() -> dict[str, int | str]:
    form_values = DEFAULT_FORM.copy()
    stored = session.get(SESSION_KEY, {})
    for key, value in stored.items():
        if key in form_values:
            form_values[key] = value
    return form_values


def update_form_values(source: Dict[str, str], keys: list[str]) -> dict[str, int | str]:
    form_values = get_form_values()
    for key in keys:
        if key in source:
            form_values[key] = source.get(key, form_values[key])
    session[SESSION_KEY] = form_values
    return form_values


def common_context() -> dict[str, Any]:
    return {
        "troop_types": TROOP_TYPES,
        "spell_types": SPELL_TYPES,
        "hero_types": HERO_TYPES,
        "pet_types": PET_TYPES,
        "guardian_types": GUARDIAN_TYPES,
        "defense_types": DEFENSE_TYPES,
        "clan_castle_types": CLAN_CASTLE_TYPES,
        "siege_types": SIEGE_TYPES,
        "display_names": DISPLAY_NAMES,
        "base_presets": BASE_PRESETS,
    }


def attack_keys() -> list[str]:
    return (
        [f"troop_{item}" for item in TROOP_TYPES]
        + [f"spell_{item}" for item in SPELL_TYPES]
        + [f"hero_{item}" for item in HERO_TYPES]
        + [f"pet_{item}" for item in PET_TYPES]
        + [f"guardian_{item}" for item in GUARDIAN_TYPES]
        + ["clan_castle", "siege_machine"]
    )


def defence_keys() -> list[str]:
    return ["base_level"] + [f"defense_{item}" for item in DEFENSE_TYPES]


@app.route("/")
def index():
    return redirect(url_for("attack"))


@app.route("/attack", methods=["GET", "POST"])
def attack():
    form_values = get_form_values()
    if request.method == "POST":
        form_values = update_form_values(request.form, attack_keys())
        return redirect(url_for("defence"))

    return render_template(
        "attack.html",
        form_values=form_values,
        step="attack",
        **common_context(),
    )


@app.route("/defence", methods=["GET", "POST"])
def defence():
    form_values = get_form_values()
    if request.method == "POST":
        form_values = update_form_values(request.form, defence_keys())
        return redirect(url_for("results"))

    return render_template(
        "defence.html",
        form_values=form_values,
        step="defence",
        **common_context(),
    )


@app.route("/results", methods=["GET"])
def results():
    form_values = get_form_values()
    analysis = run_analysis({key: str(value) for key, value in form_values.items()})
    return render_template(
        "results.html",
        form_values=form_values,
        prediction=analysis["prediction"],
        recommendation=analysis["recommendation"],
        top_recommendations=analysis["top_recommendations"],
        chart_data=analysis["chart_data"],
        insights=analysis["insights"],
        army_summary=analysis["army_summary"],
        defense_summary=analysis["defense_summary"],
        step="results",
        **common_context(),
    )


@app.route("/reset", methods=["POST"])
def reset():
    session.pop(SESSION_KEY, None)
    return redirect(url_for("attack"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    payload = {key: str(value) for key, value in body.items()}
    return jsonify(run_analysis(payload))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
