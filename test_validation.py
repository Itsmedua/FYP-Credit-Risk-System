
def is_valid_duration(duration):
    return 4 <= duration <= 72


def test_duration_validation():
    assert is_valid_duration(3) is False
    assert is_valid_duration(4) is True
    assert is_valid_duration(72) is True
    assert is_valid_duration(73) is False

    print("All duration validation tests passed!")


if __name__ == "__main__":
    test_duration_validation()
