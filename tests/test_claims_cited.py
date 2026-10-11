import unittest

from answerlab.checks import extract_claims
from answerlab.claims_cited import extract_cited_claims


class CitedClaimsTests(unittest.TestCase):
    def test_multiple_sentences_and_citations(self):
        text = "It is 3.5 kg.[web_2] It works indoors. [device_z_operating][web_1]"
        result = extract_cited_claims(text)
        self.assertEqual([c.text for c in result], ["It is 3.5 kg.", "It works indoors."])
        self.assertEqual([c.source_id for c in result], ["web_2", "device_z_operating,web_1"])
        self.assertEqual(result[0].original_line, "It is 3.5 kg.[web_2]")

    def test_citations_before_punctuation(self):
        result = extract_cited_claims("It works [web_1]. It is safe [web_2][web_3]!")
        self.assertEqual([c.text for c in result], ["It works.", "It is safe!"])
        self.assertEqual(result[1].source_id, "web_2,web_3")

    def test_uncited_sentences_are_retained(self):
        result = extract_cited_claims("First assertion. Second assertion! Last fragment")
        self.assertEqual([c.text for c in result], ["First assertion.", "Second assertion!", "Last fragment"])
        self.assertTrue(all(c.source_id is None for c in result))

    def test_abbreviations_decimals_and_urls(self):
        text = "Dr. Smith used 3.5 kg, e.g. in the U.S. study. Visit https://example.org/v1.2?q=3.5. Done!"
        result = extract_cited_claims(text)
        self.assertEqual([c.text for c in result], [
            "Dr. Smith used 3.5 kg, e.g. in the U.S. study.",
            "Visit https://example.org/v1.2?q=3.5.", "Done!"])

    def test_legacy_complete_passage_lines_are_unchanged(self):
        text = '  Device Z works from 0 to 40 C. Keep it dry. [device_z_operating]  \n\n"Quoted passage." [web_1]'
        self.assertEqual(extract_cited_claims(text), extract_claims(text))

    def test_abstention_and_blank(self):
        for text in ("INSUFFICIENT_EVIDENCE", "\n INSUFFICIENT_EVIDENCE \n", "", " \n"):
            self.assertEqual(extract_cited_claims(text), [])
        self.assertTrue(extract_cited_claims("INSUFFICIENT_EVIDENCE but guessing"))

    def test_invalid_citation_remains_unchecked_text(self):
        claim = extract_cited_claims("Unsupported [bad/id].")[0]
        self.assertIsNone(claim.source_id)
        self.assertIn("[bad/id]", claim.text)

    def test_adjacent_cited_sentences_without_space(self):
        result = extract_cited_claims("One.[web_1]Two.[web_2]")
        self.assertEqual([c.text for c in result], ["One.", "Two."])


if __name__ == "__main__":
    unittest.main()
