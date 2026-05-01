import random
import unittest

from battle_simulator import DEFENSE_TYPES, MAX_ARMY_CAPACITY, TROOP_HOUSING_SPACE, candidate_attacks, generate_dataset
from battle_simulator import AttackConfig, BaseConfig, battle_score, random_attack_config
from recommend_strategy import load_feature_importance, load_metrics, recommend_for_base


class MLPipelineTestCase(unittest.TestCase):
    def test_generated_dataset_has_expected_schema(self):
        df = generate_dataset(num_samples=25, seed=11)
        self.assertEqual(len(df), 25)
        self.assertIn("win_probability", df.columns)
        self.assertIn("troop_barbarian", df.columns)
        self.assertIn("spell_rage", df.columns)
        self.assertIn("hero_barbarian_king", df.columns)
        self.assertIn("defense_cannon", df.columns)
        self.assertIn("defense_monolith", df.columns)

    def test_candidate_attacks_exist(self):
        candidates = candidate_attacks()
        self.assertGreater(len(candidates), 0)
        self.assertTrue(any(candidate.siege_machine for candidate in candidates))
        for candidate in candidates:
            total_capacity = sum(candidate.troops[name] * TROOP_HOUSING_SPACE[name] for name in candidate.troops)
            self.assertLessEqual(total_capacity, MAX_ARMY_CAPACITY)

    def test_recommendation_returns_ranked_predictions(self):
        base = BaseConfig(
            base_level=15,
            anti_air_defense=10.0,
            ground_pressure=10.0,
            splash_defense=10.0,
            wall_strength=10.0,
            inferno_strength=10.0,
            trap_pressure=10.0,
            defenses={name: 5 for name in DEFENSE_TYPES},
        )
        ranked = recommend_for_base(base)
        self.assertIn("predicted_win_probability", ranked.columns)
        self.assertGreater(len(ranked), 0)

    def test_metrics_are_available(self):
        metrics = load_metrics()
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

    def test_feature_importance_is_available(self):
        features = load_feature_importance(limit=5)
        self.assertEqual(len(features), 5)
        self.assertIn("feature", features[0])
        self.assertIn("importance", features[0])

    def test_battle_score_reflects_air_matchup_pressure(self):
        air_attack = AttackConfig(
            troops={
                "barbarian": 0,
                "archer": 0,
                "wizard": 0,
                "goblin": 0,
                "giant": 0,
                "wall_breaker": 0,
                "balloon": 12,
                "healer": 0,
                "dragon": 8,
                "pekka": 0,
                "baby_dragon": 4,
                "miner": 0,
                "electro_dragon": 6,
                "yeti": 0,
                "dragon_rider": 4,
                "electro_titan": 0,
                "root_rider": 0,
                "thrower": 0,
                "meteor_golem": 0,
                "minion": 8,
                "hog_rider": 0,
                "valkyrie": 0,
                "golem": 0,
                "witch": 0,
                "lava_hound": 2,
                "bowler": 0,
                "ice_golem": 0,
                "apprentice_warden": 0,
                "headhunter": 0,
                "druid": 0,
                "furnace": 0,
            },
            spells={name: 0 for name in [
                "lightning", "heal", "rage", "jump", "freeze", "clone", "invisibility",
                "recall", "revive", "totem", "poison", "earthquake", "haste", "skeleton",
                "bat", "overgrowth", "ice_block"
            ]},
            heroes={
                "barbarian_king": 85,
                "archer_queen": 85,
                "minion_prince": 60,
                "grand_warden": 65,
                "royal_champion": 35,
                "dragon_duke": 32,
            },
            pets={"lassi": 4, "electro_owl": 9, "mighty_yak": 2, "unicorn": 7},
            guardians={"ground_guardian": 2, "air_guardian": 8, "healing_guardian": 6},
            clan_castle="cc_balloon",
            siege_machine="stone_slammer",
        )
        low_aa_base = BaseConfig(
            base_level=15,
            anti_air_defense=25.0,
            ground_pressure=22.0,
            splash_defense=35.0,
            wall_strength=22.0,
            inferno_strength=24.0,
            trap_pressure=18.0,
            defenses={name: 3 for name in DEFENSE_TYPES},
        )
        high_aa_base = BaseConfig(
            base_level=15,
            anti_air_defense=80.0,
            ground_pressure=58.0,
            splash_defense=35.0,
            wall_strength=22.0,
            inferno_strength=24.0,
            trap_pressure=18.0,
            defenses={name: 8 for name in DEFENSE_TYPES},
        )
        rng = random.Random(7)
        low_score = battle_score(air_attack, low_aa_base, rng)
        high_score = battle_score(air_attack, high_aa_base, random.Random(7))
        self.assertGreater(low_score, high_score)

    def test_random_attack_respects_capacity(self):
        attack = random_attack_config(random.Random(13))
        total_capacity = sum(attack.troops[name] * TROOP_HOUSING_SPACE[name] for name in attack.troops)
        self.assertLessEqual(total_capacity, MAX_ARMY_CAPACITY)

    def test_battle_score_is_deterministic_without_noise(self):
        attack = candidate_attacks()[0]
        base = BaseConfig(
            base_level=15,
            anti_air_defense=40.0,
            ground_pressure=35.0,
            splash_defense=30.0,
            wall_strength=26.0,
            inferno_strength=28.0,
            trap_pressure=20.0,
            defenses={name: 5 for name in DEFENSE_TYPES},
        )
        score_a = battle_score(attack, base, random.Random(21))
        score_b = battle_score(attack, base, random.Random(99))
        self.assertEqual(score_a, score_b)


if __name__ == "__main__":
    unittest.main()
