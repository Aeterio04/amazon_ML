"""
test_normalization.py — Unit tests for Stage 0 normalization
"""

import unittest
from src.normalization import normalize_record, transliterate_devanagari, parse_and_canonicalize_address, extract_dba


class TestNormalization(unittest.TestCase):

    def test_devanagari_transliteration(self):
        deva_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        translit = transliterate_devanagari(deva_name)
        self.assertIn("ram", translit)
        self.assertIn("marketing", translit)
        self.assertIn("private", translit)
        self.assertIn("limited", translit)

        deva_state = "दिल्ली"
        self.assertEqual(transliterate_devanagari(deva_state), "delhi")

    def test_dba_splitting(self):
        raw = "Ectolumdrex dba X+ Madison Inc"
        primary, dba = extract_dba(raw)
        self.assertEqual(primary, "Ectolumdrex")
        self.assertEqual(dba, "X+ Madison Inc")

    def test_us_state_canonicalization(self):
        addr1 = "1795 Westchester Drive, High Point, North Carolina"
        res1 = parse_and_canonicalize_address(addr1, "US")
        self.assertEqual(res1["state_code"], "NC")
        self.assertIn("nc", res1["addr_core"])

        addr2 = "GREENSBORO, NC, 19 1/2 STARDUST TRAIL"
        res2 = parse_and_canonicalize_address(addr2, "US")
        self.assertEqual(res2["state_code"], "NC")

        addr3 = "5780 Fawn Ct, Fort Worth, Texas"
        res3 = parse_and_canonicalize_address(addr3, "US")
        self.assertEqual(res3["state_code"], "TX")

    def test_house_number_and_redaction(self):
        addr_redacted = "##8 Willow Oak Lane, Charlotte, NC"
        res_red = parse_and_canonicalize_address(addr_redacted, "US")
        self.assertEqual(res_red["is_house_redacted"], "1")
        self.assertEqual(res_red["house_number"], "")

        addr_valid = "1795 Westchester Drive, High Point, NC"
        res_val = parse_and_canonicalize_address(addr_valid, "US")
        self.assertEqual(res_val["is_house_redacted"], "0")
        self.assertEqual(res_val["house_number"], "1795")

    def test_france_fallback_and_accents(self):
        name = "<< Team Ecole"
        addr = "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine"
        res = normalize_record("S1-156285671", name, addr, "France")
        self.assertEqual(res["name_core"], "team ecole")
        self.assertIn("president", res["addr_core"])
        self.assertEqual(res["house_number"], "175")
        self.assertEqual(res["country"], "France")

    def test_legal_suffix_stripping(self):
        rec1 = normalize_record("1", "Summit Inc", "10 Elm St", "US")
        self.assertEqual(rec1["name_core"], "summit")
        self.assertEqual(rec1["legal_suffix"], "inc")

        rec2 = normalize_record("2", "LLC Moncada Learning Center", "5780 Fawn Ct", "US")
        self.assertEqual(rec2["name_core"], "moncada learning center")
        self.assertEqual(rec2["legal_suffix"], "llc")


if __name__ == "__main__":
    unittest.main()
