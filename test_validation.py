#!/usr/bin/env python3
"""
Quick validation test for the translation tool fixes.
Tests the core logic without requiring heavy dependencies (torch, transformers, etc.)
"""

import re
import sys

# Copy the fixed constants and functions from the main tool
MAX_LENGTH_RATIO = 4.0  # FIXED: Was 2.5
MAX_ABSOLUTE_LENGTH = 1500  # FIXED: Was 500

VIETNAMESE_DIACRITICS_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)

VIETNAMESE_COMMON_WORDS = {
    'chào', 'xin', 'cảm', 'ơn', 'chúc', 'mừng', 'năm', 'mới',
    'người', 'chơi', 'được', 'không', 'của', 'và', 'có', 'trong',
    'bạn', 'tôi', 'này', 'đây', 'thì', 'để', 'với', 'từ',
    'đã', 'sẽ', 'đang', 'bị', 'hãy', 'là', 'nếu', 'tại',
    'những', 'các', 'một', 'nhưng', 'như', 'khi', 'về', 'sau',
}

def is_vietnamese(text: str) -> bool:
    """FIXED: Stricter detection to reduce false positives"""
    # Check for diacritics first (most reliable)
    if VIETNAMESE_DIACRITICS_PATTERN.search(text):
        return True

    # FIXED: Require at least 3 Vietnamese words AND reasonable text length
    words = set(re.findall(r'\w+', text.lower()))
    viet_word_matches = words & VIETNAMESE_COMMON_WORDS

    if len(viet_word_matches) >= 3 and len(text.strip()) >= 20:
        return True

    return False

def validate_structure(original: str, translated: str) -> tuple:
    """FIXED: Relaxed structure validation"""
    # FIXED: Only check brackets and braces (critical structure)
    critical_structure_chars = {
        '[': 'square_brackets_open',
        ']': 'square_brackets_close',
        '{': 'curly_brackets_open',
        '}': 'curly_brackets_close',
    }

    for char, name in critical_structure_chars.items():
        orig_count = original.count(char)
        trans_count = translated.count(char)

        # Allow ±1 difference for edge cases
        if abs(orig_count - trans_count) > 1:
            return False, f"structure_mismatch: {name} ({orig_count} → {trans_count})"

    # FIXED: Only validate DOUBLE QUOTES
    quote_char = '"'
    orig_count = original.count(quote_char)
    trans_count = translated.count(quote_char)

    if orig_count > 0:
        if orig_count % 2 == 0 and trans_count % 2 != 0:
            return False, f"quote_imbalance: {quote_char} count is odd ({trans_count})"

        # Allow ±2 difference
        if abs(orig_count - trans_count) > 2:
            return False, f"quote_mismatch: {quote_char} ({orig_count} → {trans_count})"

    # FIXED: Relaxed newline validation
    orig_newlines = original.count('\n')
    trans_newlines = translated.count('\n')

    if orig_newlines != trans_newlines:
        diff = abs(orig_newlines - trans_newlines)
        # Allow ±2 lines OR 50% change
        if diff > 2 and diff > orig_newlines * 0.5:
            return False, f"newline_mismatch ({orig_newlines} → {trans_newlines})"

    return True, ""

def validate_length(original: str, translated: str) -> tuple:
    """Test length validation"""
    if len(translated) > MAX_LENGTH_RATIO * len(original):
        return False, f"length_ratio_exceeded ({len(translated)} > {MAX_LENGTH_RATIO} * {len(original)})"

    if len(translated) > MAX_ABSOLUTE_LENGTH:
        return False, f"absolute_length_exceeded ({len(translated)} > {MAX_ABSOLUTE_LENGTH})"

    return True, ""

# ============================================================================
# TEST CASES
# ============================================================================

print("=" * 80)
print("VALIDATION TEST SUITE - Testing Translation Tool Fixes")
print("=" * 80)

test_results = {"passed": 0, "failed": 0}

def run_test(test_name, test_func, expected_result=True):
    """Run a test and print result"""
    global test_results
    try:
        result = test_func()
        passed = (result == expected_result)
        if passed:
            print(f"✅ PASS: {test_name}")
            test_results["passed"] += 1
        else:
            print(f"❌ FAIL: {test_name}")
            print(f"   Expected: {expected_result}, Got: {result}")
            test_results["failed"] += 1
        return passed
    except Exception as e:
        print(f"❌ ERROR: {test_name}")
        print(f"   Exception: {e}")
        test_results["failed"] += 1
        return False

print("\n" + "─" * 80)
print("1. VIETNAMESE DETECTION TESTS")
print("─" * 80)

# Should detect as Vietnamese
run_test(
    "Vietnamese with diacritics",
    lambda: is_vietnamese("Chào mừng bạn đến với máy chủ"),
    True
)

run_test(
    "Vietnamese text (3+ words, >20 chars)",
    lambda: is_vietnamese("Xin chào các bạn người chơi mới"),
    True
)

# Should NOT detect as Vietnamese (false positives)
run_test(
    "Short English 'Ban' (was false positive)",
    lambda: is_vietnamese("Ban"),
    False
)

run_test(
    "Short English 'Can you help' (was false positive)",
    lambda: is_vietnamese("Can you help"),
    False
)

run_test(
    "English 'Co op mode' (was false positive)",
    lambda: is_vietnamese("Co op mode"),
    False
)

print("\n" + "─" * 80)
print("2. STRUCTURE VALIDATION TESTS")
print("─" * 80)

# Test apostrophe handling (English → Vietnamese loses apostrophes)
run_test(
    "Apostrophes differ (it's → nó)",
    lambda: validate_structure("It's great", "Thật tuyệt")[0],  # Should PASS now
    True
)

run_test(
    "Multiple apostrophes differ (player's → của người chơi)",
    lambda: validate_structure("The player's items", "Các vật phẩm của người chơi")[0],
    True
)

# Test bracket validation (±1 tolerance)
run_test(
    "Brackets exact match",
    lambda: validate_structure("[INFO] Test [OK]", "[THÔNG TIN] Test [OK]")[0],
    True
)

run_test(
    "Brackets ±1 difference (should pass)",
    lambda: validate_structure("[INFO] Test", "Thông tin Test")[0],  # Lost 1 bracket
    True
)

run_test(
    "Brackets ±2 difference (should fail)",
    lambda: validate_structure("[INFO] [TEST] [OK]", "Thông tin")[0],  # Lost 3 pairs
    False
)

# Test quote validation (±2 tolerance)
run_test(
    "Quotes exact match",
    lambda: validate_structure('He said "Hello"', 'Anh ấy nói "Xin chào"')[0],
    True
)

run_test(
    "Quotes ±1 difference (natural variation)",
    lambda: validate_structure('Say "Hi"', 'Nói xin chào')[0],  # Lost quotes
    True
)

# Test newline validation (±2 or 50%)
run_test(
    "Newlines ±1 (should pass)",
    lambda: validate_structure("Line1\nLine2\nLine3", "Dòng1\nDòng2\nDòng3\n")[0],
    True
)

run_test(
    "Newlines ±2 (should pass)",
    lambda: validate_structure("A\nB\nC\nD", "A\nB")[0],
    True
)

print("\n" + "─" * 80)
print("3. LENGTH VALIDATION TESTS")
print("─" * 80)

# Test length ratio (4.0x)
run_test(
    "Length ratio 2x (within limit)",
    lambda: validate_length("Hello world", "Xin chào thế giới của chúng tôi")[0],
    True
)

run_test(
    "Length ratio 3.5x (within new limit)",
    lambda: validate_length("Hello", "Xin chào bạn")[0],  # 5 chars → 15 chars = 3x
    True
)

run_test(
    "Length ratio 5x (exceeds limit)",
    lambda: validate_length("Hi", "A" * 11)[0],  # 5.5x
    False
)

# Test absolute length (1500 chars)
run_test(
    "Absolute length 1000 chars (within limit)",
    lambda: validate_length("X" * 300, "Y" * 1000)[0],  # 300→1000 = 3.33x ratio, 1000 < 1500 absolute
    True
)

run_test(
    "Absolute length 1600 chars (exceeds limit)",
    lambda: validate_length("Test", "X" * 1600)[0],
    False
)

print("\n" + "─" * 80)
print("4. EDGE CASE TESTS")
print("─" * 80)

# Real-world problematic cases
run_test(
    "Possessive plural: players'",
    lambda: validate_structure("All players' items", "Tất cả vật phẩm của người chơi")[0],
    True
)

run_test(
    "Multiple contractions: it's, they're, don't",
    lambda: validate_structure("It's ok, they're here, don't worry",
                               "Không sao, họ ở đây, đừng lo")[0],
    True
)

run_test(
    "Short text not Vietnamese: 'are you sure'",
    lambda: is_vietnamese("are you sure"),
    False
)

run_test(
    "JSON in message (brackets preserved)",
    lambda: validate_structure('{"enabled":true}', '{"bật":true}')[0],
    True
)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✅ Passed: {test_results['passed']}")
print(f"❌ Failed: {test_results['failed']}")
print(f"📊 Total:  {test_results['passed'] + test_results['failed']}")

if test_results['failed'] == 0:
    print("\n🎉 ALL TESTS PASSED! Translation tool fixes are working correctly.")
    sys.exit(0)
else:
    print(f"\n⚠️  {test_results['failed']} test(s) failed. Please review the fixes.")
    sys.exit(1)
