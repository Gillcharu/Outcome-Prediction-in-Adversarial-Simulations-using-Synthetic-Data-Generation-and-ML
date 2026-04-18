from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import random
from typing import Dict, List, Tuple

import pandas as pd


TROOP_TYPES = [
    "barbarian",
    "archer",
    "wizard",
    "goblin",
    "giant",
    "wall_breaker",
    "balloon",
    "healer",
    "dragon",
    "pekka",
    "baby_dragon",
    "miner",
    "electro_dragon",
    "yeti",
    "dragon_rider",
    "electro_titan",
    "root_rider",
    "thrower",
    "meteor_golem",
    "minion",
    "hog_rider",
    "valkyrie",
    "golem",
    "witch",
    "lava_hound",
    "bowler",
    "ice_golem",
    "apprentice_warden",
    "headhunter",
    "druid",
    "furnace",
]

SPELL_TYPES = [
    "lightning",
    "heal",
    "rage",
    "jump",
    "freeze",
    "clone",
    "invisibility",
    "recall",
    "revive",
    "totem",
    "poison",
    "earthquake",
    "haste",
    "skeleton",
    "bat",
    "overgrowth",
    "ice_block",
]

HERO_TYPES = [
    "barbarian_king",
    "archer_queen",
    "minion_prince",
    "grand_warden",
    "royal_champion",
    "dragon_duke",
]

PET_TYPES = [
    "lassi",
    "electro_owl",
    "mighty_yak",
    "unicorn",
]

GUARDIAN_TYPES = [
    "ground_guardian",
    "air_guardian",
    "healing_guardian",
]

CLAN_CASTLE_TYPES = [
    "cc_yeti",
    "cc_balloon",
    "cc_dragon",
    "cc_ice_golem",
]

SIEGE_TYPES = [
    "wall_wrecker",
    "battle_blimp",
    "stone_slammer",
    "barracks",
    "log_launcher",
    "flame_flinger",
    "drill",
    "launcher",
]

DEFENSE_TYPES = [
    "cannon",
    "archer_tower",
    "wall",
    "mortar",
    "air_defence",
    "wizard_tower",
    "air_sweeper",
    "hidden_tesla",
    "bomb_tower",
    "xbow",
    "inferno_tower",
    "eagle_artillery",
    "scattershot",
    "spell_tower",
    "monolith",
    "traps",
]

TROOP_MAX_COUNTS = {
    "barbarian": 30,
    "archer": 30,
    "wizard": 20,
    "goblin": 20,
    "giant": 14,
    "wall_breaker": 12,
    "balloon": 16,
    "healer": 8,
    "dragon": 12,
    "pekka": 10,
    "baby_dragon": 10,
    "miner": 18,
    "electro_dragon": 10,
    "yeti": 10,
    "dragon_rider": 10,
    "electro_titan": 8,
    "root_rider": 8,
    "thrower": 12,
    "meteor_golem": 6,
    "minion": 24,
    "hog_rider": 18,
    "valkyrie": 14,
    "golem": 6,
    "witch": 14,
    "lava_hound": 6,
    "bowler": 12,
    "ice_golem": 8,
    "apprentice_warden": 6,
    "headhunter": 8,
    "druid": 8,
    "furnace": 6,
}

SPELL_MAX_COUNTS = {spell: 4 for spell in SPELL_TYPES}
SPELL_MAX_COUNTS.update({"clone": 2, "recall": 2, "revive": 2, "totem": 2})

HERO_MAX_LEVELS = {
    "barbarian_king": 95,
    "archer_queen": 95,
    "minion_prince": 70,
    "grand_warden": 70,
    "royal_champion": 45,
    "dragon_duke": 45,
}

TROOP_AIR_POWER = {
    "archer": 0.15,
    "wizard": 0.35,
    "balloon": 1.7,
    "dragon": 1.9,
    "baby_dragon": 1.25,
    "electro_dragon": 2.2,
    "dragon_rider": 1.85,
    "minion": 0.8,
    "lava_hound": 1.25,
}

TROOP_GROUND_POWER = {
    "barbarian": 0.45,
    "archer": 0.2,
    "wizard": 1.0,
    "goblin": 0.45,
    "giant": 1.1,
    "wall_breaker": 0.7,
    "pekka": 2.2,
    "miner": 1.25,
    "yeti": 1.7,
    "electro_titan": 1.9,
    "root_rider": 1.8,
    "thrower": 1.4,
    "meteor_golem": 2.1,
    "hog_rider": 1.4,
    "valkyrie": 1.3,
    "golem": 1.8,
    "witch": 0.85,
    "bowler": 1.5,
    "ice_golem": 0.85,
    "apprentice_warden": 0.35,
    "headhunter": 0.7,
    "druid": 0.25,
    "furnace": 0.8,
}

TROOP_SUPPORT_POWER = {
    "archer": 0.15,
    "wizard": 0.25,
    "healer": 1.6,
    "baby_dragon": 0.2,
    "yeti": 0.3,
    "witch": 0.95,
    "lava_hound": 0.3,
    "ice_golem": 0.9,
    "apprentice_warden": 1.1,
    "headhunter": 0.25,
    "druid": 1.25,
    "furnace": 0.55,
}

SPELL_AIR_POWER = {
    "rage": 2.0,
    "freeze": 1.8,
    "clone": 1.8,
    "haste": 1.6,
    "invisibility": 1.1,
    "bat": 1.0,
    "overgrowth": 0.8,
    "ice_block": 0.9,
}

SPELL_GROUND_POWER = {
    "heal": 1.9,
    "rage": 1.7,
    "jump": 1.4,
    "freeze": 1.3,
    "earthquake": 1.5,
    "skeleton": 0.8,
    "overgrowth": 1.0,
    "totem": 1.0,
    "revive": 1.2,
}

SPELL_SUPPORT_POWER = {
    "heal": 1.3,
    "rage": 1.0,
    "recall": 0.9,
    "revive": 1.3,
    "poison": 0.7,
    "invisibility": 1.2,
    "ice_block": 1.0,
}

DEFENSE_AIR_WEIGHTS = {
    "archer_tower": 1.2,
    "air_defence": 2.3,
    "wizard_tower": 1.1,
    "air_sweeper": 1.6,
    "hidden_tesla": 1.1,
    "xbow": 1.0,
    "inferno_tower": 1.2,
    "eagle_artillery": 1.0,
    "scattershot": 1.15,
    "spell_tower": 0.9,
    "monolith": 1.1,
    "traps": 0.8,
}

DEFENSE_GROUND_WEIGHTS = {
    "cannon": 0.95,
    "archer_tower": 0.7,
    "mortar": 1.2,
    "wizard_tower": 1.2,
    "hidden_tesla": 0.9,
    "bomb_tower": 1.3,
    "xbow": 1.0,
    "inferno_tower": 1.3,
    "eagle_artillery": 1.2,
    "scattershot": 1.6,
    "spell_tower": 1.0,
    "monolith": 1.7,
    "traps": 1.0,
}

DEFENSE_WALL_WEIGHTS = {
    "wall": 2.5,
    "cannon": 0.4,
    "xbow": 0.5,
    "monolith": 0.8,
}

DEFENSE_SPLASH_WEIGHTS = {
    "mortar": 1.7,
    "wizard_tower": 1.8,
    "bomb_tower": 1.6,
    "eagle_artillery": 1.4,
    "scattershot": 2.0,
    "spell_tower": 1.4,
}


def make_zero_map(keys: List[str]) -> Dict[str, int]:
    return {key: 0 for key in keys}


def merge_setup(keys: List[str], updates: Dict[str, int]) -> Dict[str, int]:
    values = make_zero_map(keys)
    values.update(updates)
    return values


@dataclass(frozen=True)
class AttackConfig:
    troops: Dict[str, int]
    spells: Dict[str, int]
    heroes: Dict[str, int]
    pets: Dict[str, int]
    guardians: Dict[str, int]
    clan_castle: str
    siege_machine: str


@dataclass(frozen=True)
class BaseConfig:
    base_level: int
    anti_air_defense: float
    splash_defense: float
    wall_strength: float
    inferno_strength: float
    trap_pressure: float
    defenses: Dict[str, int] = field(default_factory=dict)


def aggregate_base_metrics(defenses: Dict[str, int]) -> Dict[str, float]:
    return {
        "anti_air_defense": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_AIR_WEIGHTS.items()),
        "splash_defense": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_SPLASH_WEIGHTS.items()),
        "wall_strength": sum(defenses.get(name, 0) * weight for name, weight in DEFENSE_WALL_WEIGHTS.items()),
        "inferno_strength": (
            defenses.get("inferno_tower", 0) * 2.2
            + defenses.get("monolith", 0) * 1.8
            + defenses.get("xbow", 0) * 1.0
            + defenses.get("spell_tower", 0) * 1.1
        ),
        "trap_pressure": defenses.get("traps", 0) * 2.3 + defenses.get("bomb_tower", 0) * 0.7,
    }


def random_attack_config(rng: random.Random) -> AttackConfig:
    troops = {name: rng.randint(0, TROOP_MAX_COUNTS[name]) for name in TROOP_TYPES}
    spells = {name: rng.randint(0, SPELL_MAX_COUNTS[name]) for name in SPELL_TYPES}
    heroes = {name: rng.randint(0, HERO_MAX_LEVELS[name]) for name in HERO_TYPES}
    pets = {name: rng.randint(0, 10) for name in PET_TYPES}
    guardians = {name: rng.randint(0, 10) for name in GUARDIAN_TYPES}
    return AttackConfig(
        troops=troops,
        spells=spells,
        heroes=heroes,
        pets=pets,
        guardians=guardians,
        clan_castle=rng.choice(CLAN_CASTLE_TYPES),
        siege_machine=rng.choice(SIEGE_TYPES),
    )


def random_base_config(rng: random.Random) -> BaseConfig:
    defenses = {name: rng.randint(1, 10) for name in DEFENSE_TYPES}
    metrics = aggregate_base_metrics(defenses)
    return BaseConfig(
        base_level=rng.randint(9, 17),
        anti_air_defense=metrics["anti_air_defense"],
        splash_defense=metrics["splash_defense"],
        wall_strength=metrics["wall_strength"],
        inferno_strength=metrics["inferno_strength"],
        trap_pressure=metrics["trap_pressure"],
        defenses=defenses,
    )


def attack_style_scores(attack: AttackConfig) -> Tuple[float, float, float]:
    air_power = sum(attack.troops[name] * weight for name, weight in TROOP_AIR_POWER.items())
    air_power += sum(attack.spells[name] * weight for name, weight in SPELL_AIR_POWER.items())

    ground_power = sum(attack.troops[name] * weight for name, weight in TROOP_GROUND_POWER.items())
    ground_power += sum(attack.spells[name] * weight for name, weight in SPELL_GROUND_POWER.items())

    support_power = sum(attack.troops[name] * weight for name, weight in TROOP_SUPPORT_POWER.items())
    support_power += sum(attack.spells[name] * weight for name, weight in SPELL_SUPPORT_POWER.items())

    hero_boost = sum(attack.heroes.values()) * 0.05
    pet_boost = (
        attack.pets["electro_owl"] * 0.9
        + attack.pets["mighty_yak"] * 0.9
        + attack.pets["lassi"] * 0.6
        + attack.pets["unicorn"] * 1.0
    )
    guardian_boost = (
        attack.guardians["air_guardian"] * 0.8
        + attack.guardians["ground_guardian"] * 0.8
        + attack.guardians["healing_guardian"] * 0.9
    )

    air_power += attack.pets["electro_owl"] * 0.8 + attack.guardians["air_guardian"] * 0.8
    ground_power += attack.pets["mighty_yak"] * 0.8 + attack.guardians["ground_guardian"] * 0.8
    support_power += hero_boost + pet_boost * 0.35 + guardian_boost

    return air_power, ground_power, support_power


def battle_score(attack: AttackConfig, base: BaseConfig, rng: random.Random) -> float:
    air_power, ground_power, support_power = attack_style_scores(attack)

    ground_pressure = sum(base.defenses.get(name, 0) * weight for name, weight in DEFENSE_GROUND_WEIGHTS.items())
    attack_total = air_power + ground_power + support_power
    defense_total = (
        base.anti_air_defense * 1.05
        + base.splash_defense * 0.95
        + base.wall_strength * 1.15
        + base.inferno_strength * 1.2
        + base.trap_pressure * 0.9
        + ground_pressure * 0.24
        + base.base_level * 6.0
    )

    score = 46.0
    score += attack_total * 0.34
    score -= defense_total * 0.58

    if attack.siege_machine == "wall_wrecker":
        score += max(0.0, 18.0 - base.wall_strength) * 1.1
    elif attack.siege_machine == "log_launcher":
        score += max(0.0, 20.0 - base.wall_strength - base.inferno_strength * 0.25) * 0.9
    elif attack.siege_machine == "battle_blimp":
        score += air_power * 0.22 - base.anti_air_defense * 0.2
    elif attack.siege_machine == "stone_slammer":
        score += air_power * 0.25 - base.anti_air_defense * 0.18
    elif attack.siege_machine == "barracks":
        score += ground_power * 0.15 + support_power * 0.08
    elif attack.siege_machine == "flame_flinger":
        score += support_power * 0.12 + max(0.0, 14.0 - ground_pressure * 0.08)
    elif attack.siege_machine == "drill":
        score += ground_power * 0.16 + max(0.0, 14.0 - base.wall_strength * 0.3)
    elif attack.siege_machine == "launcher":
        score += air_power * 0.12 + ground_power * 0.12

    if attack.clan_castle == "cc_balloon":
        score += air_power * 0.12 - base.anti_air_defense * 0.12
    elif attack.clan_castle == "cc_dragon":
        score += air_power * 0.13 - base.anti_air_defense * 0.1
    elif attack.clan_castle == "cc_yeti":
        score += ground_power * 0.12
    elif attack.clan_castle == "cc_ice_golem":
        score += support_power * 0.1 + attack.spells["freeze"] * 0.8

    score += rng.uniform(-8.0, 8.0)
    probability = 1.0 / (1.0 + pow(2.71828, -(score - 50.0) / 11.0))
    return max(0.0, min(1.0, probability))


def flatten_attack_config(attack: AttackConfig) -> Dict[str, int | str]:
    row: Dict[str, int | str] = {}
    for troop in TROOP_TYPES:
        row[f"troop_{troop}"] = attack.troops[troop]
    for spell in SPELL_TYPES:
        row[f"spell_{spell}"] = attack.spells[spell]
    for hero in HERO_TYPES:
        row[f"hero_{hero}"] = attack.heroes[hero]
    for pet in PET_TYPES:
        row[f"pet_{pet}"] = attack.pets[pet]
    for guardian in GUARDIAN_TYPES:
        row[f"guardian_{guardian}"] = attack.guardians[guardian]
    row["clan_castle"] = attack.clan_castle
    row["siege_machine"] = attack.siege_machine
    return row


def flatten_base_config(base: BaseConfig) -> Dict[str, int | float]:
    row: Dict[str, int | float] = {
        "base_level": base.base_level,
        "anti_air_defense": base.anti_air_defense,
        "splash_defense": base.splash_defense,
        "wall_strength": base.wall_strength,
        "inferno_strength": base.inferno_strength,
        "trap_pressure": base.trap_pressure,
    }
    for defense in DEFENSE_TYPES:
        row[f"defense_{defense}"] = base.defenses.get(defense, 0)
    return row


def generate_dataset(num_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict[str, int | float | str]] = []
    for _ in range(num_samples):
        attack = random_attack_config(rng)
        base = random_base_config(rng)
        row: Dict[str, int | float | str] = {}
        row.update(flatten_attack_config(attack))
        row.update(flatten_base_config(base))
        row["win_probability"] = battle_score(attack, base, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_attacks() -> List[AttackConfig]:
    hero_defaults = {
        "barbarian_king": 88,
        "archer_queen": 90,
        "minion_prince": 60,
        "grand_warden": 65,
        "royal_champion": 38,
        "dragon_duke": 35,
    }
    pet_defaults = {"lassi": 8, "electro_owl": 8, "mighty_yak": 7, "unicorn": 9}
    guardian_defaults = {"ground_guardian": 7, "air_guardian": 7, "healing_guardian": 8}

    troop_options = [
        merge_setup(TROOP_TYPES, {"electro_dragon": 8, "dragon_rider": 6, "balloon": 10, "baby_dragon": 4, "minion": 8}),
        merge_setup(TROOP_TYPES, {"yeti": 8, "electro_titan": 4, "root_rider": 6, "wizard": 8, "healer": 4, "apprentice_warden": 2, "wall_breaker": 6}),
        merge_setup(TROOP_TYPES, {"miner": 14, "hog_rider": 12, "healer": 4, "wizard": 6, "headhunter": 3, "ice_golem": 2}),
        merge_setup(TROOP_TYPES, {"pekka": 5, "bowler": 10, "witch": 6, "golem": 2, "ice_golem": 2, "wall_breaker": 6}),
    ]

    spell_options = [
        merge_setup(SPELL_TYPES, {"rage": 3, "freeze": 3, "clone": 1, "haste": 1}),
        merge_setup(SPELL_TYPES, {"heal": 3, "rage": 2, "freeze": 1, "jump": 1}),
        merge_setup(SPELL_TYPES, {"invisibility": 2, "freeze": 2, "rage": 2, "recall": 1}),
    ]

    attacks: List[AttackConfig] = []
    for troops, spells, clan_castle, siege_machine in product(
        troop_options,
        spell_options,
        CLAN_CASTLE_TYPES,
        SIEGE_TYPES,
    ):
        attacks.append(
            AttackConfig(
                troops=troops,
                spells=spells,
                heroes=hero_defaults,
                pets=pet_defaults,
                guardians=guardian_defaults,
                clan_castle=clan_castle,
                siege_machine=siege_machine,
            )
        )
    return attacks
