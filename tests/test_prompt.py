"""Tests for vibewarp.utils.prompt."""

from vibewarp.utils.prompt import parse_prompt


class TestParsePrompt:
    def test_simple_prompt(self):
        text, weight = parse_prompt("a beautiful painting")
        assert text == "a beautiful painting"
        assert weight == 1.0

    def test_weighted_prompt(self):
        text, weight = parse_prompt("a beautiful painting:0.8")
        assert text == "a beautiful painting"
        assert weight == 0.8

    def test_url_prompt(self):
        text, weight = parse_prompt("https://example.com/image.jpg:0.5")
        assert text == "https://example.com/image.jpg"
        assert weight == 0.5

    def test_url_no_weight(self):
        text, weight = parse_prompt("https://example.com/image.jpg")
        assert text == "https://example.com/image.jpg"
        assert weight == 1.0

    def test_negative_weight(self):
        text, weight = parse_prompt("ugly:-0.5")
        assert text == "ugly"
        assert weight == -0.5

    def test_zero_weight(self):
        text, weight = parse_prompt("neutral:0")
        assert text == "neutral"
        assert weight == 0.0
