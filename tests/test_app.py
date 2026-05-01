import unittest

from app import (
    app,
    CLAN_CASTLE_TYPES,
    DEFAULT_FORM,
    DEFENSE_TYPES,
    GUARDIAN_TYPES,
    HERO_TYPES,
    PET_TYPES,
    SIEGE_TYPES,
    SPELL_TYPES,
    TROOP_TYPES,
)


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def build_full_payload(self):
        payload = {
            "base_level": int(DEFAULT_FORM["base_level"]),
            "clan_castle": CLAN_CASTLE_TYPES[0],
            "siege_machine": SIEGE_TYPES[0],
        }
        for troop in TROOP_TYPES:
            payload[f"troop_{troop}"] = int(DEFAULT_FORM[f"troop_{troop}"])
        for spell in SPELL_TYPES:
            payload[f"spell_{spell}"] = int(DEFAULT_FORM[f"spell_{spell}"])
        for hero in HERO_TYPES:
            payload[f"hero_{hero}"] = int(DEFAULT_FORM[f"hero_{hero}"])
        for pet in PET_TYPES:
            payload[f"pet_{pet}"] = int(DEFAULT_FORM[f"pet_{pet}"])
        for guardian in GUARDIAN_TYPES:
            payload[f"guardian_{guardian}"] = int(DEFAULT_FORM[f"guardian_{guardian}"])
        for defense in DEFENSE_TYPES:
            payload[f"defense_{defense}"] = int(DEFAULT_FORM[f"defense_{defense}"])
        return payload

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_home_page_loads(self):
        response = self.client.get("/", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Build the attack in steps.", response.data)
        self.assertIn(b"Attack Setup", response.data)

    def test_api_predict_returns_analysis(self):
        payload = self.build_full_payload()
        payload.update(
            {
                "troop_barbarian": 12,
                "troop_archer": 16,
                "troop_giant": 8,
                "troop_wizard": 10,
                "troop_dragon": 4,
                "troop_balloon": 6,
                "troop_healer": 2,
                "troop_pekka": 3,
                "spell_rage": 2,
                "spell_heal": 2,
                "spell_freeze": 2,
                "spell_lightning": 1,
                "hero_barbarian_king": 75,
                "hero_archer_queen": 80,
                "hero_grand_warden": 55,
                "hero_royal_champion": 30,
                "clan_castle": "cc_yeti",
                "siege_machine": "log_launcher",
                "base_level": 14,
                "defense_cannon": 5,
                "defense_air_defence": 7,
                "defense_wall": 7,
                "defense_inferno_tower": 7,
                "defense_traps": 5,
            }
        )
        response = self.client.post("/api/predict", json=payload)
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", body)
        self.assertIn("recommendation", body)
        self.assertIn("top_recommendations", body)
        self.assertIn("chart_data", body)

    def test_api_predict_rejects_invalid_payload(self):
        response = self.client.post(
            "/api/predict",
            json={"troop_barbarian": -1, "clan_castle": "bad_cc", "siege_machine": "bad_siege"},
        )
        body = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("errors", body)


if __name__ == "__main__":
    unittest.main()
