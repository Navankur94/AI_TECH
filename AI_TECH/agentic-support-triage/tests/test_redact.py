from app.redact import redact_pii


def test_each_identifier_type_is_masked():
    text = ("Mail me at asha.rao@example.com, PAN ABCDE1234F, card 4111 1111 1111 1111, "
            "Aadhaar 2345 6789 0123, phone +91 9876543210.")
    masked, found = redact_pii(text)
    assert found == ["AADHAAR", "CARD", "EMAIL", "PAN", "PHONE"]
    for raw in ("asha.rao@example.com", "ABCDE1234F", "4111", "2345 6789 0123", "9876543210"):
        assert raw not in masked


def test_card_like_number_failing_luhn_is_not_masked_as_card():
    masked, found = redact_pii("reference 1234 5678 9012 3456")
    assert "CARD" not in found and "1234 5678 9012 3456" in masked


def test_text_without_pii_is_unchanged():
    text = "When will my redemption reach my bank account?"
    assert redact_pii(text) == (text, [])


def test_found_list_has_types_only_never_values():
    _, found = redact_pii("PAN ABCDE1234F")
    assert found == ["PAN"]
