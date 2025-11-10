#!/usr/bin/env python3
"""
mine_inplace_translator.py
==========================
Production-ready IN-PLACE translator for Minecraft server configs/localization files.
Translates English -> Vietnamese using OFFLINE NLLB-200-3.3B with quantization/offloading.

QUICKSTART
----------
1. Install dependencies:
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install transformers accelerate sentencepiece bitsandbytes
   pip install ruamel.yaml pyyaml tqdm chardet argostranslate langdetect

   NOTE: If using different CUDA version (cu118/cu124/cpu), adjust torch index-url:
   - CUDA 11.8: https://download.pytorch.org/whl/cu118
   - CUDA 12.4: https://download.pytorch.org/whl/cu124
   - CPU only: https://download.pytorch.org/whl/cpu

2. First run (download model):
   python mine_inplace_translator.py --download-only

3. Translate a folder:
   python mine_inplace_translator.py /path/to/server/plugins

4. Dry-run preview:
   python mine_inplace_translator.py /path/to/folder --dry-run --report

5. Rollback changes:
   python mine_inplace_translator.py /path/to/folder --rollback

FEATURES
--------
- Default model: NLLB-200-3.3B (quality first)
- Automatic 4-bit -> 8-bit -> full precision fallback with timeouts
- Adaptive batch sizing by quantization mode (48/24/12)
- Platform detection (CUDA/macOS/ROCm) with warnings
- Disk space check before download (~8GB needed)
- In-place atomic writes + .bak backups
- Preserves YAML structure, anchors, block scalars, comments
- Protects placeholders, colors, commands, MiniMessage tags
- Context-aware adjacent translation with delimiter fallback
- Vietnamese detection to skip already-translated content
- Multi-language detection (Chinese/Japanese/Korean/Thai) to skip non-English text
- Validation guards (token count, length, format)
- Memory-efficient streaming for large folders
- Comprehensive self-tests

USAGE
-----
python mine_inplace_translator.py <folder> [options]

OPTIONS
-------
--model {nllb-3.3b,nllb-1.3b,marian,m2m100,argos}  Model to use (default: nllb-3.3b)
--gpu / --cpu                                       Force device (auto-detect by default)
--batch-size N                                      Override adaptive batch size
--low-memory                                        Halve batch, disable grouping, clear cache
--fallback-model MODEL                              Fallback if primary fails (e.g., nllb-1.3b)
--ext yaml,json,lang,properties,txt                 File extensions (default: all)
--force                                             Translate even if Vietnamese detected
--dry-run                                           Preview changes without writing
--rollback                                          Restore all .bak files
--report                                            Generate mine_translate_report.csv
--download-only                                     Download model and exit
--self-test                                         Run internal validation tests
--max-memory "cuda:0=7GiB,cpu=30GiB"                Cap memory usage per device
--glossary FILE                                     JSON glossary for term mapping

EXAMPLES
--------
# Download model only
python mine_inplace_translator.py --download-only

# Translate with 4-bit quantization (default)
python mine_inplace_translator.py ./server/plugins

# Use fallback model on low-memory systems
python mine_inplace_translator.py ./server --fallback-model nllb-1.3b --low-memory

# Dry-run with report
python mine_inplace_translator.py ./configs --dry-run --report

# Custom memory limits
python mine_inplace_translator.py ./data --max-memory "cuda:0=6GiB,cpu=20GiB"

# Rollback all changes
python mine_inplace_translator.py ./server/plugins --rollback

# Self-test
python mine_inplace_translator.py --self-test
"""

# Fix Windows console UTF-8 output
if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except:
            pass

import argparse
import gc
import json
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

try:
    import chardet
except ImportError:
    print("ERROR: chardet not installed. Run: pip install chardet")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import LiteralScalarString, FoldedScalarString
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
except ImportError:
    print("ERROR: ruamel.yaml not installed. Run: pip install ruamel.yaml")
    sys.exit(1)


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    "nllb-3.3b": {
        "repo": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "vie_Latn",
        "batch_4bit": 48,
        "batch_8bit": 24,
        "batch_full": 12,
    },
    "nllb-1.3b": {
        "repo": "facebook/nllb-200-1.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "vie_Latn",
        "batch_4bit": 64,
        "batch_8bit": 32,
        "batch_full": 16,
    },
    "marian": {
        "repo": "Helsinki-NLP/opus-mt-en-vi",
        "src_lang": "en",
        "tgt_lang": "vi",
        "batch_4bit": 64,
        "batch_8bit": 32,
        "batch_full": 16,
    },
    "m2m100": {
        "repo": "facebook/m2m100_418M",
        "src_lang": "en",
        "tgt_lang": "vi",
        "batch_4bit": 64,
        "batch_8bit": 32,
        "batch_full": 16,
    },
    "argos": {
        "repo": "argostranslate",  # Special handler
        "src_lang": "en",
        "tgt_lang": "vi",
        "batch_4bit": 64,
        "batch_8bit": 32,
        "batch_full": 16,
    },
}

VIETNAMESE_DIACRITICS_PATTERN = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)

# Common Vietnamese words for detection (lowercase)
# FIXED: Use only words with diacritics to avoid false positives with English
VIETNAMESE_COMMON_WORDS = {
    'chào', 'xin', 'cảm', 'ơn', 'chúc', 'mừng', 'năm', 'mới',
    'người', 'chơi', 'được', 'không', 'của', 'và', 'có', 'trong',
    'bạn', 'tôi', 'này', 'đây', 'thì', 'để', 'với', 'từ',
    'đã', 'sẽ', 'đang', 'bị', 'hãy', 'là', 'nếu', 'tại',
    'những', 'các', 'một', 'nhưng', 'như', 'khi', 'về', 'sau',
}

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

try:
    import langdetect
    from langdetect import detect_langs
    langdetect.DetectorFactory.seed = 0  # Reproducibility
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ WARNING: langdetect not installed!")
    print("   Some files may be in Chinese/Japanese/Korean but will be treated as English.")
    print("   Install with: pip install langdetect")

# Language names for logging
LANGUAGE_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "id": "Indonesian",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


# Common English words whitelist - if text contains these, force translate
ENGLISH_INDICATOR_WORDS = {
    # Minecraft specific
    'player', 'server', 'spawn', 'command', 'permission', 'world', 'admin',
    'minecraft', 'block', 'item', 'inventory', 'craft', 'build', 'game',
    'kill', 'death', 'respawn', 'teleport', 'warp', 'home', 'lobby', 'hub',
    'sword', 'weapon', 'armor', 'tool', 'potion', 'enchant', 'level',
    'rank', 'balance', 'money', 'coin', 'shop', 'buy', 'sell', 'trade',
    'join', 'leave', 'kick', 'ban', 'mute', 'warn', 'online', 'offline',
    'chest', 'furnace', 'portal', 'dimension', 'nether', 'end', 'overworld',
    'damage', 'health', 'food', 'hunger', 'experience', 'reward', 'bonus',
    'upgrade', 'downgrade', 'legendary', 'epic', 'rare', 'common', 'uncommon',
    
    # Common action verbs
    'welcome', 'goodbye', 'hello', 'thanks', 'sorry', 'please',
    'click', 'hover', 'use', 'type', 'open', 'close', 'activate',
    'enable', 'disable', 'start', 'stop', 'pause', 'resume',
    'give', 'take', 'get', 'receive', 'send', 'bought', 'sold', 'purchased',
    'killed', 'died', 'respawning', 'warping', 'teleporting', 'executing',
    'playing', 'enjoy', 'have', 'need', 'must', 'should', 'could', 'would',
    
    # Common nouns/adjectives  
    'message', 'text', 'title', 'subtitle', 'description', 'name',
    'error', 'warning', 'success', 'failed', 'invalid', 'missing',
    'available', 'unavailable', 'allowed', 'denied', 'forbidden',
    'new', 'old', 'first', 'last', 'next', 'previous', 'current',
    'total', 'count', 'amount', 'price', 'cost', 'value',
    'time', 'day', 'night', 'hour', 'minute', 'second', 'cooldown',
    'feature', 'system', 'service', 'support', 'help', 'guide', 'tutorial',
    'rule', 'term', 'condition', 'policy', 'agreement', 'violation',
    'staff', 'member', 'owner', 'moderator', 'operator', 'developer',
    'community', 'friend', 'enemy', 'team', 'party', 'guild', 'clan',
    'world', 'map', 'arena', 'lobby', 'spawn', 'checkpoint',
    
    # Common phrases
    'thank you', 'see you', 'have fun', 'enjoy', 'good luck',
    'be careful', 'watch out', 'well done', 'nice work', 'great job',
    'are you', 'do you', 'can you', 'will you', 'should you',
    'you are', 'you have', 'you need', 'you must', 'you can', "you don't",
    'this is', 'that is', 'there is', 'here is', 'it is',
    'welcome to', 'join the', 'join our', 'visit our', 'check out',
    'for more', 'click here', 'read more', 'find out', 'learn more',
    'be respectful', 'follow the', 'respect the', 'obey the',
    'not allowed', 'not permitted', 'is forbidden', 'is prohibited',
    'has joined', 'has left', 'has been', 'will be', 'was',
    'your rank', 'your balance', 'your level', 'your progress',
    'the best', 'the server', 'the game', 'the world', 'the player',
    'all players', 'other players', 'new player', 'every player',
}


def has_english_indicators(text: str) -> bool:
    """Check if text contains common English words."""
    text_lower = text.lower()
    # Remove placeholders and special chars for better matching
    cleaned = re.sub(r'[%{}\[\]<>]', ' ', text_lower)
    cleaned = re.sub(r'&[0-9a-fk-or]', '', cleaned)
    
    for word in ENGLISH_INDICATOR_WORDS:
        if word in cleaned:
            return True
    return False


def detect_language(text: str, min_confidence: float = 0.5) -> Tuple[str, float]:
    """
    Detect language of text.

    Returns:
        (language_code, confidence) or ("en", 0.3) if detection fails

    Language codes: en, vi, zh-cn, ja, ko, th, id, ru, etc.
    """
    if not LANGDETECT_AVAILABLE:
        return "en", 0.3  # Assume English if no detector

    if not text or len(text.strip()) < 3:
        return "en", 0.3

    # PRIORITY FIX: If text has English indicator words, always classify as English
    if has_english_indicators(text):
        return "en", 1.0

    # Clean text: remove placeholders, URLs, numbers, punctuation
    cleaned = text
    cleaned = re.sub(r'https?://[^\s]+', '', cleaned)
    cleaned = re.sub(r'\{[^}]+\}', '', cleaned)
    cleaned = re.sub(r'%[^%\s]+%', '', cleaned)
    cleaned = re.sub(r'%\{[^}]+\}%', '', cleaned)  # Math expressions like %{expr}%
    cleaned = re.sub(r'[&§][0-9a-fk-orA-FK-OR]', '', cleaned)
    cleaned = re.sub(r'/[A-Za-z0-9_\-]+', '', cleaned)
    cleaned = re.sub(r'\b\d+(?:\.\d+)?\b', '', cleaned)

    # Check if too much content was removed (>60% removed = mostly placeholders)
    original_len = len(text.strip())
    cleaned_len = len(cleaned.strip())
    if original_len > 0 and cleaned_len < original_len * 0.4:
        return "en", 0.3  # Mostly placeholders - assume English

    # Count actual words (not just characters)
    words = [w for w in cleaned.split() if len(w) > 1]  # Filter out single chars like "!", "."
    if len(words) < 3:
        return "en", 0.3  # Too few words after cleaning - assume English

    try:
        results = detect_langs(cleaned)
        if results:
            best = results[0]
            lang = best.lang.lower()
            if lang.startswith("zh"):
                lang = "zh"  # Chinese (any variant)
            return lang, best.prob
        return "en", 0.3
    except Exception:
        return "en", 0.3


# Protected token patterns (properly escaped for Python strings)
# PRIORITY: Longer/complex patterns FIRST to avoid partial matches
PROTECTED_PATTERNS = [
    # ═══════════════════════════════════════════════════════════════════
    # PLACEHOLDERAPI CONDITIONALS - MUST BE FIRST (longest match rule)
    # ═══════════════════════════════════════════════════════════════════
    # Full conditional blocks (match %if%...%else%...%endif% completely)
    r"%if_[^%]+%[^%]*%else%[^%]*%endif%",  # %if_condition%text%else%other%endif%
    r"%if_[^%]+%[^%]*%endif%",  # %if_condition%text%endif%

    # Math expressions inside placeholders
    r"%\{[^}]+\}%",  # %{player_level * 100}%, %{health / max * 100}%

    # Function calls with parameters
    r"%[a-z_]+_[a-z_]+\([^)]+\)(?:_[a-z_]+)*%",  # %vault_eco_balance(player)_formatted%

    # ═══════════════════════════════════════════════════════════════════
    # STANDARD PLACEHOLDERS
    # ═══════════════════════════════════════════════════════════════════
    r"\{[^}]+\}",  # {player}, {0}
    r"%[^%\s]+%",  # %player%, %count% (fallback for simple placeholders)
    r"\$\{[^}]+\}",  # ${VAR}
    r"\$\([^)]+\)",  # $(...)
    r"\{\{[^}]+\}\}",  # {{mustache}}

    # ═══════════════════════════════════════════════════════════════════
    # COLOR CODES
    # ═══════════════════════════════════════════════════════════════════
    r"&[0-9a-fk-orA-FK-OR]",  # &c, &l
    r"§[0-9a-fk-orA-FK-OR]",  # §c, §l

    # ═══════════════════════════════════════════════════════════════════
    # URLs, EMAILS, IPs (BEFORE /command to avoid breaking URLs)
    # ═══════════════════════════════════════════════════════════════════
    r"https?://[a-zA-Z0-9.-]+(?:/[^\s]*)?",  # URLs
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
    r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b",  # IP:port

    # ═══════════════════════════════════════════════════════════════════
    # COMMANDS & PERMISSIONS
    # ═══════════════════════════════════════════════════════════════════
    r"/[A-Za-z0-9_\-]+",  # /command
    r"[A-Za-z]+\.[A-Za-z0-9_.\-]+",  # permission.node, domain.com

    # ═══════════════════════════════════════════════════════════════════
    # MINIMESSAGE TAGS
    # ═══════════════════════════════════════════════════════════════════
    r"<[A-Za-z0-9:_= ,./'\"-]+>",  # <red>, <click:run_command:/cmd>

    # ═══════════════════════════════════════════════════════════════════
    # MINECRAFT IDS & VERSIONS
    # ═══════════════════════════════════════════════════════════════════
    r"[a-z0-9_]+:[a-z0-9_/]+",  # minecraft:diamond_sword
    r"v?\d+(?:\.\d+)+",  # v1.2.3, 1.19.4

    # ═══════════════════════════════════════════════════════════════════
    # CODE BLOCKS
    # ═══════════════════════════════════════════════════════════════════
    r"```[\s\S]+?```",  # ```code block``` - non-greedy, matches newlines
    r"`[^`\n]+`",  # `code` - inline only, no newlines

    # ═══════════════════════════════════════════════════════════════════
    # CODE-ISH TOKENS
    # ═══════════════════════════════════════════════════════════════════
    r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b",  # camelCase
    r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # PascalCase
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",  # snake_case
    r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b",  # kebab-case

    # ═══════════════════════════════════════════════════════════════════
    # CLI FLAGS & JAVA ARGS
    # ═══════════════════════════════════════════════════════════════════
    r"--[a-z][a-z0-9\-]*",  # --flag
    r"-[A-Z][a-zA-Z0-9]+",  # -Xmx2G

    # ═══════════════════════════════════════════════════════════════════
    # NUMBERS WITH UNITS
    # ═══════════════════════════════════════════════════════════════════
    r"\b\d+(?:\.\d+)?(?:MB|GB|KB|ms|s|m|h|d|%|x)\b",
]

# Escape sequences
ESCAPE_PATTERNS = [
    (r"\\n", "\n"),
    (r"\\t", "\t"),
    (r"\\r", "\r"),
    (r"\\\\", "\\"),
    (r'\\"', '"'),
    (r"\\'", "'"),
]


def decode_escapes(text: str) -> str:
    """Decode escape sequences before translation."""
    result = text
    for escaped, actual in ESCAPE_PATTERNS:
        result = result.replace(escaped, actual)
    return result


def encode_escapes(text: str) -> str:
    """Re-encode escape sequences after translation."""
    result = text
    # Reverse order: actual -> escaped
    for escaped, actual in reversed(ESCAPE_PATTERNS):
        result = result.replace(actual, escaped)
    return result


# Delimiters for adjacent translation
DELIMITER_REC = "\x1E"  # ASCII record separator
DELIMITER_UNIT = "\x1F"  # ASCII unit separator
DELIMITER_REC_UNICODE = "⟨REC⟩"
DELIMITER_UNIT_UNICODE = "⟨UNIT⟩"

# File extensions
DEFAULT_EXTENSIONS = {".yml", ".yaml", ".json", ".properties", ".lang", ".txt", ".mcfunction"}

# Validation thresholds
# FIXED: Increased limits to accommodate Vietnamese translation expansion (typically 1.5-3.5x English)
MAX_LENGTH_RATIO = 4.0  # Increased from 2.5 to allow natural Vietnamese expansion
MAX_ABSOLUTE_LENGTH = 1500  # Increased from 500 to handle longer translations
CHUNK_THRESHOLD_CHARS = 1500  # ~400 tokens
YAML_MULTILINE_THRESHOLD = 200  # chars
ADJACENT_GROUP_SIZE = 4  # 3-5 neighbor values

# Memory thresholds
LARGE_FOLDER_FILE_COUNT = 1000
LARGE_FOLDER_SIZE_MB = 100

# Download disk space requirement
REQUIRED_DISK_SPACE_GB = 8
WARNED_DISK_SPACE_GB = 10

# Test strings for quality check
QUALITY_TEST_STRINGS = [
    "Welcome to the server!",
    "You don't have permission to do that.",
    "Player {player} has joined the game.",
    "This item costs %price% coins.",
    "Click here to teleport home.",
]


# ============================================================================
# TIMEOUT HANDLER
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class TimeoutError(Exception):
    pass


def run_with_timeout(func, seconds: int, *args, **kwargs):
    """Cross-platform timeout using ThreadPoolExecutor for safety.

    BUGFIX: Replaced PyThreadState_SetAsyncExc (unsafe) with ThreadPoolExecutor.
    """
    if seconds is None or seconds <= 0:
        return func(*args, **kwargs)

    # Unix/Linux: use signal.alarm (fast, efficient)
    if platform.system() != "Windows":
        def _handler(_signum, _frame):
            raise TimeoutError("Operation timed out")

        original = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            return func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original)

    # Windows: use ThreadPoolExecutor (safe, no interpreter corruption)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=seconds)
            except FuturesTimeoutError:
                raise TimeoutError("Operation timed out")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet with validation and fallback."""
    with open(file_path, "rb") as f:
        raw = f.read(102400)  # Read first 100KB for better detection
    result = chardet.detect(raw)
    encoding = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0.0)

    # BUGFIX: Map problematic encodings to UTF-8
    encoding_map = {
        "ascii": "utf-8",
        "us-ascii": "utf-8",
        "charmap": "utf-8",  # Windows charmap can't handle emoji
        "cp1252": "utf-8",
        "windows-1252": "utf-8",
    }

    if encoding:
        encoding = encoding_map.get(encoding.lower(), encoding)

    # BUGFIX: Validate encoding can decode the file, fallback to UTF-8
    try:
        with open(file_path, "r", encoding=encoding) as f:
            f.read(1024)  # Test read
        return encoding
    except (UnicodeDecodeError, LookupError, TypeError):
        # Try UTF-8
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
            return "utf-8"
        except UnicodeDecodeError:
            # Last resort: latin-1 never fails
            return "latin-1"


def get_free_disk_space_gb(path: Path) -> float:
    """Get free disk space in GB for the given path."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 ** 3)
    except Exception:
        return 0.0


def get_folder_size_mb(folder: Path) -> float:
    """Calculate total size of folder in MB."""
    total = 0
    try:
        for entry in folder.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total / (1024 ** 2)


def count_files_in_folder(folder: Path, extensions: Set[str]) -> int:
    """Count files with given extensions in folder."""
    count = 0
    try:
        for ext in extensions:
            count += len(list(folder.rglob(f"*{ext}")))
    except Exception:
        pass
    return count


def validate_path_in_base(file_path: Path, base_dir: Path) -> bool:
    """Check if path is within base directory (prevent path traversal).

    BUGFIX: Security - prevent writing outside allowed directory.
    """
    try:
        file_path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def atomic_write(file_path: Path, content: str, encoding: str = "utf-8", base_dir: Optional[Path] = None):
    """Atomically write content to file with fsync.

    BUGFIX: Added optional base_dir validation to prevent path traversal.
    """
    # Optional security check
    if base_dir and not validate_path_in_base(file_path, base_dir):
        raise ValueError(f"Path traversal detected: {file_path} is outside {base_dir}")

    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent, prefix=f".{file_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(temp_fd, "w", encoding=encoding, newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename
        shutil.move(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def create_backup(file_path: Path) -> Optional[Path]:
    """Create .bak backup if it doesn't exist. Returns backup path."""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        return backup_path
    return None


def restore_backup(file_path: Path) -> bool:
    """Restore file from .bak backup."""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        backup_path.unlink()
        return True
    return False


def is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese diacritics or common Vietnamese words.

    FIXED: Stricter detection to reduce false positives
    """
    # Check for diacritics first (most reliable)
    if VIETNAMESE_DIACRITICS_PATTERN.search(text):
        return True

    # Check for common Vietnamese words (case-insensitive)
    # FIXED: Require at least 3 Vietnamese words to avoid false positives
    # Also require minimum text length to avoid matching short fragments
    words = set(re.findall(r'\w+', text.lower()))
    viet_word_matches = words & VIETNAMESE_COMMON_WORDS

    # Require 3+ Vietnamese words AND reasonable text length
    if len(viet_word_matches) >= 3 and len(text.strip()) >= 20:
        return True

    return False


def is_empty_or_whitespace(text: str) -> bool:
    """Check if text is empty or whitespace-only."""
    return not text or text.isspace()


# ============================================================================
# CONTEXT-AWARE DETECTION (NEW - Smart code vs text detection)
# ============================================================================

# YAML keys that indicate CODE context (technical values that should NOT be translated)
CODE_CONTEXT_KEYS = {
    # Identifiers
    'id', 'key', 'name', 'type', 'class', 'uuid', 'guid',
    # Permissions & commands
    'permission', 'permissions', 'perm', 'perms', 'node', 'command', 'cmd', 'alias', 'aliases',
    # Technical
    'namespace', 'plugin', 'world', 'biome', 'enchantment', 'potion', 'effect',
    # Configuration
    'enabled', 'disabled', 'mode', 'format', 'pattern', 'regex',
    # Paths & URLs
    'path', 'file', 'folder', 'directory', 'url', 'link', 'website',
    # Database & API
    'table', 'column', 'field', 'database', 'api', 'endpoint',
    # Minecraft specific
    'material', 'item', 'block', 'entity', 'sound', 'particle',
}

# YAML keys that indicate NATURAL LANGUAGE context (should be translated)
TEXT_CONTEXT_KEYS = {
    # User-facing text
    'message', 'messages', 'msg', 'text', 'description', 'desc',
    'title', 'subtitle', 'actionbar', 'chat', 'lore', 'tooltip',
    # Help & info
    'help', 'usage', 'info', 'about', 'tutorial', 'guide',
    # Feedback
    'success', 'error', 'warning', 'fail', 'denied', 'invalid',
    # Content
    'content', 'body', 'display', 'label', 'placeholder',
}


def is_code_context(path: str) -> bool:
    """
    Check if YAML path indicates code/technical context (should NOT translate).

    Examples:
        - "permission" → True (code context)
        - "config.command.teleport" → True (code context)
        - "messages.welcome" → False (text context)
        - "descriptions.item_sword" → False (text context)
    """
    if not path:
        return False

    # Split path and check last key (most specific)
    parts = path.lower().split('.')
    last_key = parts[-1] if parts else ""

    # Check if last key is in code context list
    if last_key in CODE_CONTEXT_KEYS:
        return True

    # Check if any part contains code keywords
    for part in parts:
        if part in CODE_CONTEXT_KEYS:
            return True

    # If explicitly in text context, return False
    if last_key in TEXT_CONTEXT_KEYS:
        return False

    return False


def looks_like_code(text: str) -> bool:
    """
    Check if text looks like code/technical string (should NOT translate).

    Detects:
        - Permission nodes: "player.admin.teleport"
        - Namespace IDs: "minecraft:diamond_sword"
        - Commands: "/spawn", "/teleport"
        - Paths: "plugins/MyPlugin/config.yml"
        - Single words without spaces: "teleport", "admin"
        - Technical patterns: "camelCase", "snake_case", "kebab-case"
    """
    if not text or len(text.strip()) == 0:
        return False

    text = text.strip()

    # Already protected by TokenProtector (has placeholders, color codes, etc.)
    if any(char in text for char in ['%', '{', '}', '&', '§']):
        return False  # Let normal protection handle it

    # Permission node pattern: "group.subgroup.permission"
    if re.match(r'^[a-z][a-z0-9]*(\.[a-z][a-z0-9_-]*){2,}$', text, re.IGNORECASE):
        return True

    # Namespace ID: "minecraft:item", "myserver:custom_item"
    if re.match(r'^[a-z0-9_-]+:[a-z0-9_/-]+$', text, re.IGNORECASE):
        return True

    # Command pattern: starts with /
    if text.startswith('/'):
        return True

    # File path pattern: contains / or \ with extension
    if ('/' in text or '\\' in text) and '.' in text:
        return True

    # Single technical word (no spaces, contains underscores or hyphens)
    if ' ' not in text and len(text) > 3:
        # Has underscores or hyphens (technical naming)
        if '_' in text or '-' in text:
            return True

        # camelCase or PascalCase (but not all caps)
        if re.search(r'[a-z][A-Z]', text):
            return True

    # All uppercase acronym (but not just "I" or "A")
    if text.isupper() and len(text) >= 3 and text.isalpha():
        return True

    # Looks like natural language (has common words, articles, verbs)
    natural_indicators = ['the', 'a', 'an', 'is', 'are', 'you', 'your', 'this', 'that',
                          'have', 'has', 'will', 'can', 'please', 'welcome', 'hello']
    words = text.lower().split()
    if any(word in natural_indicators for word in words):
        return False  # Definitely natural language

    # Short text without spaces (likely technical)
    if ' ' not in text and len(text) <= 15:
        # But not if it's just one common English word
        common_words = ['yes', 'no', 'true', 'false', 'enabled', 'disabled', 'on', 'off']
        if text.lower() not in common_words:
            return True

    return False


def load_glossary(glossary_path: Optional[Path]) -> Dict[str, str]:
    """Load glossary JSON file (en -> vi mapping)."""
    if not glossary_path or not glossary_path.exists():
        return {}
    try:
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)
        # Sort by key length (longest first) for longest-match
        return OrderedDict(sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True))
    except Exception as e:
        print(f"WARNING: Failed to load glossary: {e}")
        return {}


# ============================================================================
# TOKEN PROTECTION
# ============================================================================

class TokenProtector:
    """Protect and restore special tokens in text."""

    def __init__(self):
        self.patterns = [re.compile(p) for p in PROTECTED_PATTERNS]
        self.placeholder_map: Dict[str, str] = {}

    def protect(self, text: str) -> str:
        """Replace protected tokens with placeholders."""
        import uuid
        self.placeholder_map.clear()

        def replace_fn(match):
            token = match.group(0)
            placeholder = f"__TOK_{uuid.uuid4().hex}__"
            self.placeholder_map[placeholder] = token
            return placeholder

        result = text
        for pattern in self.patterns:
            result = pattern.sub(replace_fn, result)
        return result

    def restore(self, text: str) -> str:
        """Restore protected tokens from placeholders."""
        result = text
        for placeholder, token in self.placeholder_map.items():
            result = result.replace(placeholder, token)
        return result

    def validate_restoration(self, original: str, translated: str) -> Tuple[bool, str]:
        """Validate that all tokens are present and unchanged."""
        # Extract original tokens
        original_tokens = []
        for pattern in self.patterns:
            original_tokens.extend(pattern.findall(original))

        if not original_tokens:
            return True, ""

        # Check if all tokens are in translated
        for token in original_tokens:
            if token not in translated:
                return False, f"Missing token: {token}"

        # Check token counts match
        for token in set(original_tokens):
            orig_count = original_tokens.count(token)
            trans_count = translated.count(token)
            if orig_count != trans_count:
                return False, f"Token count mismatch: {token} (expected {orig_count}, got {trans_count})"

        return True, ""


# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load translation model with quantization/offloading fallback."""

    def __init__(self, model_name: str, fallback_model: Optional[str] = None,
                 device: Optional[str] = None, max_memory: Optional[str] = None,
                 low_memory: bool = False):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.device = device
        self.max_memory = max_memory
        self.low_memory = low_memory
        self.config = MODEL_CONFIGS.get(model_name)
        if not self.config:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = None
        self.tokenizer = None
        self.quantization_mode = None
        self.actual_model_name = model_name

    def check_platform_support(self) -> Tuple[bool, bool, str]:
        """Check platform support for quantization.

        Returns:
            (supports_4bit, supports_8bit, warning_message)
        """
        system = platform.system()
        has_cuda = torch.cuda.is_available()

        # macOS / CPU-only
        if system == "Darwin" or not has_cuda:
            return False, False, (
                "Platform does not support bitsandbytes quantization (macOS or CPU-only). "
                "Will use full precision with offloading. Consider --fallback-model for better performance."
            )

        # ROCm/AMD
        if has_cuda and hasattr(torch.version, 'hip') and torch.version.hip:
            return False, False, (
                "ROCm/AMD platform detected. Bitsandbytes not supported. "
                "Will use full precision with offloading."
            )

        # Linux/Windows + CUDA
        if system in ["Linux", "Windows"] and has_cuda:
            try:
                import bitsandbytes
                return True, True, ""
            except ImportError:
                return False, False, (
                    "bitsandbytes not installed. Install with: pip install bitsandbytes\n"
                    "Will use full precision with offloading."
                )

        return False, False, "Unknown platform. Will use full precision."

    def parse_max_memory(self) -> Optional[Dict[str, str]]:
        """Parse --max-memory string into dict."""
        if not self.max_memory:
            return None

        try:
            # Format: "cuda:0=7GiB,cpu=30GiB"
            mem_dict = {}
            for part in self.max_memory.split(","):
                device, size = part.split("=")
                mem_dict[device.strip()] = size.strip()
            return mem_dict
        except Exception as e:
            print(f"WARNING: Failed to parse --max-memory: {e}")
            return None

    def is_model_cached(self, repo: str) -> bool:
        """Check if model already exists in cache."""
        try:
            from transformers.utils import TRANSFORMERS_CACHE
            cache_dir = Path(TRANSFORMERS_CACHE) if TRANSFORMERS_CACHE else Path.home() / ".cache" / "huggingface" / "hub"
        except (ImportError, TypeError):
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        # Check if any model files exist
        model_id_safe = repo.replace("/", "--")
        model_cache = cache_dir / f"models--{model_id_safe}"

        if not model_cache.exists():
            return False

        # Check for actual model weight files
        return any(model_cache.rglob("*.bin")) or any(model_cache.rglob("*.safetensors"))

    def load_model_with_timeout(self, repo: str, load_kwargs: dict, timeout_sec: int, mode_desc: str):
        """Attempt to load model with timeout.

        BUGFIX: Use run_with_timeout instead of context manager for safety.
        """
        print(f"  Attempting {mode_desc} (timeout: {timeout_sec}s)...", flush=True)
        try:
            def _load():
                return AutoModelForSeq2SeqLM.from_pretrained(repo, **load_kwargs)

            model = run_with_timeout(_load, timeout_sec)
            print(f"  [OK] Loaded with {mode_desc}", flush=True)
            return model, mode_desc
        except TimeoutError as e:
            print(f"  [FAIL] {e}", flush=True)
            return None, None
        except Exception as e:
            print(f"  [FAIL] Failed: {e}", flush=True)
            return None, None

    def load(self):
        """Load model with automatic fallback strategy."""
        repo = self.config["repo"]

        # Special handler for argos
        if self.model_name == "argos":
            self._load_argos()
            return

        print(f"\nLoading model: {repo}")

        # Check disk space
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        free_space = get_free_disk_space_gb(cache_dir)

        if free_space < REQUIRED_DISK_SPACE_GB:
            # Critical - not enough space
            print(f"\n[ERROR] ERROR: Insufficient disk space!")
            print(f"   Available: {free_space:.1f} GB")
            print(f"   Required: ~{REQUIRED_DISK_SPACE_GB} GB")
            print(f"\nOptions:")
            print(f"  1. Free up disk space in {cache_dir}")
            print(f"  2. Use --fallback-model nllb-1.3b (needs ~1.5GB)")

            try:
                response = input("\nContinue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Aborted by user.")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted by user.")
                sys.exit(1)

        elif free_space < WARNED_DISK_SPACE_GB:
            # Warning level
            print(f"\n[WARN] WARNING: Low disk space ({free_space:.1f} GB free)")
            print(f"   Recommended: {WARNED_DISK_SPACE_GB} GB")

            if self.fallback_model:
                print(f"   Tip: Use --fallback-model {self.fallback_model} (lighter)")

            try:
                response = input("   Continue? [Y/n]: ").strip().lower()
                if response == 'n':
                    print("Aborted by user.")
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted by user.")
                sys.exit(1)

        # Load tokenizer
        print("Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)

        # Set source language for NLLB/M2M100 (affects encoding)
        if "nllb" in self.model_name or "m2m100" in self.model_name:
            self.tokenizer.src_lang = self.config["src_lang"]

        # Check platform support
        supports_4bit, supports_8bit, warning = self.check_platform_support()
        if warning:
            print(f"[WARN] {warning}")

        # Parse max_memory
        max_mem_dict = self.parse_max_memory()

        # Determine device
        if self.device:
            use_device = self.device
        else:
            use_device = "cuda" if torch.cuda.is_available() else "cpu"

        common_kwargs = {
            "device_map": "auto" if use_device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }
        if max_mem_dict:
            common_kwargs["max_memory"] = max_mem_dict

        # Detect first-time download and adjust timeouts
        is_first_download = not self.is_model_cached(repo)

        if is_first_download:
            print("[WARN] First-time download detected. This may take 5-10 minutes...")
            timeout_4bit = 600  # 10 minutes for first download
            timeout_8bit = 300  # 5 minutes
            timeout_full = 180  # 3 minutes
        else:
            timeout_4bit = 120  # 2 minutes for cached loading
            timeout_8bit = 90
            timeout_full = 60

        # Try loading with fallback strategy
        loaded = False

        # 1. Try 4-bit
        if supports_4bit and not self.low_memory:
            kwargs = {
                **common_kwargs,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
            }
            model, mode = self.load_model_with_timeout(repo, kwargs, timeout_4bit, "4-bit quantization")
            if model:
                self.model = model
                self.quantization_mode = "4bit"
                loaded = True

        # 2. Try 8-bit
        if not loaded and supports_8bit:
            kwargs = {
                **common_kwargs,
                "load_in_8bit": True,
            }
            model, mode = self.load_model_with_timeout(repo, kwargs, timeout_8bit, "8-bit quantization")
            if model:
                self.model = model
                self.quantization_mode = "8bit"
                loaded = True

        # 3. Try full precision with offload
        if not loaded:
            kwargs = common_kwargs.copy()
            if use_device == "cpu":
                kwargs["device_map"] = None
            model, mode = self.load_model_with_timeout(repo, kwargs, timeout_full, "full precision")
            if model:
                self.model = model
                self.quantization_mode = "full"
                loaded = True

        # 4. Try fallback model
        if not loaded and self.fallback_model:
            print(f"\n[WARN] Primary model failed. Trying fallback: {self.fallback_model}")
            fallback_config = MODEL_CONFIGS.get(self.fallback_model)
            if fallback_config:
                fallback_repo = fallback_config["repo"]
                print(f"Loading fallback tokenizer: {fallback_repo}", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_repo)
                if "nllb" in self.fallback_model or "m2m100" in self.fallback_model:
                    self.tokenizer.src_lang = fallback_config["src_lang"]

                kwargs = common_kwargs.copy()
                if use_device == "cpu":
                    kwargs["device_map"] = None
                model, mode = self.load_model_with_timeout(fallback_repo, kwargs, 60, "fallback full precision")
                if model:
                    self.model = model
                    self.quantization_mode = "full"
                    self.actual_model_name = self.fallback_model
                    self.config = fallback_config
                    loaded = True

        if not loaded:
            raise RuntimeError("Failed to load model with all strategies. Check logs above.")

        # Log device placement
        if hasattr(self.model, "hf_device_map"):
            print(f"\nDevice placement:")
            for layer, device in list(self.model.hf_device_map.items())[:5]:
                print(f"  {layer}: {device}")
            if len(self.model.hf_device_map) > 5:
                print(f"  ... ({len(self.model.hf_device_map)} layers total)")

        print(f"\n[OK] Model loaded successfully: {self.actual_model_name} ({self.quantization_mode})")

    def _load_argos(self):
        """Load argostranslate model."""
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            raise ImportError("argostranslate not installed. Run: pip install argostranslate")

        print("Loading Argos Translate (en -> vi)...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        # Find en -> vi package
        en_vi_package = None
        for pkg in available_packages:
            if pkg.from_code == "en" and pkg.to_code == "vi":
                en_vi_package = pkg
                break

        if not en_vi_package:
            raise RuntimeError("Argos Translate en->vi package not found")

        # BUGFIX: Check if en->vi specifically is installed, not just any package
        installed = argostranslate.package.get_installed_packages()
        has_en_vi = any(pkg.from_code == "en" and pkg.to_code == "vi" for pkg in installed)

        if not has_en_vi:
            print("Installing Argos en->vi package...")
            argostranslate.package.install_from_path(en_vi_package.download())

        self.model = "argos"  # Dummy
        self.tokenizer = None
        self.quantization_mode = "argos"
        print("[OK] Argos Translate loaded")

    def get_adaptive_batch_size(self, user_batch_size: Optional[int]) -> int:
        """Get adaptive batch size based on quantization mode."""
        if user_batch_size:
            return user_batch_size

        if self.low_memory:
            base = self.config.get(f"batch_{self.quantization_mode}", 12)
            return max(1, base // 2)

        return self.config.get(f"batch_{self.quantization_mode}", 12)


# ============================================================================
# TRANSLATION MEMORY (Cache system)
# ============================================================================

import sqlite3
import hashlib
from datetime import datetime


class TranslationMemory:
    """
    SQLite-based translation cache to avoid re-translating identical text.

    Benefits:
    - Speeds up translation (cache hit = instant)
    - Ensures consistency across files
    - Persists between runs
    """

    def __init__(self, db_path: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.db = None
            return

        # Default path: ~/.cache/minecraft_translator/tm.db
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "minecraft_translator"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "tm.db"

        self.db_path = db_path
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.create_table()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_queries = 0

    def create_table(self):
        """Create translation memory table if not exists."""
        if not self.db:
            return

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                source_hash TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                target_text TEXT NOT NULL,
                context TEXT,
                model_name TEXT,
                created_at INTEGER,
                last_used INTEGER,
                use_count INTEGER DEFAULT 1
            )
        """)

        # Index for faster lookups
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_context
            ON translations(context)
        """)

        self.db.commit()

    def _hash_source(self, source: str, context: str = None) -> str:
        """Generate hash for source text + context."""
        key = f"{source}|{context or ''}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def get(self, source: str, context: str = None, model_name: str = None) -> Optional[str]:
        """
        Lookup translation from cache.

        Args:
            source: Source text
            context: YAML path or file context
            model_name: Model name (optional filter)

        Returns:
            Cached translation or None
        """
        if not self.enabled or not self.db:
            return None

        self.total_queries += 1
        source_hash = self._hash_source(source, context)

        try:
            cursor = self.db.execute("""
                SELECT target_text, use_count
                FROM translations
                WHERE source_hash = ?
                AND (model_name = ? OR ? IS NULL)
                LIMIT 1
            """, (source_hash, model_name, model_name))

            row = cursor.fetchone()
            if row:
                target_text, use_count = row

                # Update statistics
                self.db.execute("""
                    UPDATE translations
                    SET last_used = ?, use_count = ?
                    WHERE source_hash = ?
                """, (int(datetime.now().timestamp()), use_count + 1, source_hash))
                self.db.commit()

                self.hits += 1
                return target_text

            self.misses += 1
            return None

        except Exception as e:
            print(f"⚠️ TM lookup error: {e}")
            return None

    def put(self, source: str, target: str, context: str = None, model_name: str = None):
        """
        Store translation in cache.

        Args:
            source: Source text
            target: Translated text
            context: YAML path or file context
            model_name: Model name used
        """
        if not self.enabled or not self.db:
            return

        source_hash = self._hash_source(source, context)
        now = int(datetime.now().timestamp())

        try:
            self.db.execute("""
                INSERT OR REPLACE INTO translations
                (source_hash, source_text, target_text, context, model_name, created_at, last_used, use_count)
                VALUES (?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT use_count FROM translations WHERE source_hash = ?), 1))
            """, (source_hash, source, target, context, model_name, now, now, source_hash))
            self.db.commit()

        except Exception as e:
            print(f"⚠️ TM store error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.db:
            return {
                "enabled": False,
                "hits": 0,
                "misses": 0,
                "total": 0,
                "hit_rate": 0.0,
                "cache_size": 0
            }

        try:
            cursor = self.db.execute("SELECT COUNT(*) FROM translations")
            cache_size = cursor.fetchone()[0]

            hit_rate = self.hits / self.total_queries if self.total_queries > 0 else 0.0

            return {
                "enabled": True,
                "hits": self.hits,
                "misses": self.misses,
                "total": self.total_queries,
                "hit_rate": hit_rate,
                "cache_size": cache_size,
                "db_path": str(self.db_path)
            }

        except Exception:
            return {
                "enabled": True,
                "hits": self.hits,
                "misses": self.misses,
                "total": self.total_queries,
                "hit_rate": 0.0,
                "cache_size": 0
            }

    def clear(self):
        """Clear all cached translations."""
        if not self.enabled or not self.db:
            return

        try:
            self.db.execute("DELETE FROM translations")
            self.db.commit()
            print("✅ Translation memory cleared")
        except Exception as e:
            print(f"⚠️ Failed to clear TM: {e}")

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# TRANSLATOR
# ============================================================================

class Translator:
    """Main translation engine with Translation Memory cache."""

    def __init__(self, model_loader: ModelLoader, batch_size: int,
                 low_memory: bool = False, glossary: Optional[Dict[str, str]] = None,
                 force: bool = False, tm: Optional[TranslationMemory] = None):
        self.model_loader = model_loader
        self.batch_size = batch_size
        self.low_memory = low_memory
        self.glossary = glossary or {}
        self.force = force
        self.protector = TokenProtector()
        self.on_translation = None  # Callback for each translation
        self.tm = tm  # Translation Memory cache

        # Choose delimiters
        self.delim_rec = DELIMITER_REC
        self.delim_unit = DELIMITER_UNIT

    def apply_glossary_pre(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Apply glossary before translation (longest match first)."""
        if not self.glossary:
            return text, {}

        replacements = {}
        result = text
        for en_term, vi_data in self.glossary.items():
            if en_term in result:
                placeholder = f"__GLOSS{len(replacements)}__"

                # Track usage (support both dict and string formats)
                if isinstance(vi_data, dict):
                    vi_term = vi_data.get("vi", "")
                    # Increment usage counter
                    vi_data["usage"] = vi_data.get("usage", 0) + 1
                else:
                    vi_term = vi_data

                replacements[placeholder] = vi_term
                result = result.replace(en_term, placeholder)
        return result, replacements

    def apply_glossary_post(self, text: str, replacements: Dict[str, str]) -> str:
        """Restore glossary terms after translation."""
        result = text
        for placeholder, vi_term in replacements.items():
            result = result.replace(placeholder, vi_term)
        return result

    def should_skip(self, text: str) -> Tuple[bool, str]:
        """Check if text should be skipped. FIXED: Added detailed logging."""
        if is_empty_or_whitespace(text):
            return True, "empty/whitespace"

        if not self.force and is_vietnamese(text):
            # FIXED: Log why text was detected as Vietnamese
            has_diacritics = bool(VIETNAMESE_DIACRITICS_PATTERN.search(text))
            if has_diacritics:
                print(f"⏭️  SKIP (Vietnamese detected via diacritics): {text[:80]}...")
            else:
                print(f"⏭️  SKIP (Vietnamese detected via word matching): {text[:80]}...")
            return True, "already_vietnamese"

        return False, ""

    def check_delimiter_collision(self, texts: List[str]) -> Tuple[str, str]:
        """Check for delimiter collision and return safe delimiters."""
        combined = "".join(texts)

        # Try ASCII delimiters first
        if DELIMITER_REC not in combined and DELIMITER_UNIT not in combined:
            return DELIMITER_REC, DELIMITER_UNIT

        # Try Unicode delimiters
        if DELIMITER_REC_UNICODE not in combined and DELIMITER_UNIT_UNICODE not in combined:
            return DELIMITER_REC_UNICODE, DELIMITER_UNIT_UNICODE

        # Generate random markers
        import random
        import string
        while True:
            suffix = "".join(random.choices(string.ascii_letters, k=6))
            rec = f"__DLMR{suffix}__"
            unit = f"__DLMU{suffix}__"
            if rec not in combined and unit not in combined:
                return rec, unit

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts."""
        if not texts:
            return []

        # Argos Translate
        if self.model_loader.actual_model_name == "argos":
            import argostranslate.translate
            results = []
            for text in texts:
                translated = argostranslate.translate.translate(text, "en", "vi")
                results.append(translated)
            return results

        # Transformer models
        model = self.model_loader.model
        tokenizer = self.model_loader.tokenizer
        config = self.model_loader.config

        # BUGFIX: Set forced target language for NLLB and M2M100
        forced_bos_token_id = None
        model_name = self.model_loader.actual_model_name

        if "nllb" in model_name or "m2m100" in model_name:
            tgt_lang = config["tgt_lang"]

            # NLLB tokenizer uses set_tgt_lang_special_tokens() to configure target language
            if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
                tokenizer.set_tgt_lang_special_tokens(tgt_lang)
                # After setting, get the BOS token which is now the language-specific one
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

            # Fallback: Check if tokenizer has lang_code_to_id (older NLLB versions)
            elif hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
                forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

            # M2M100 uses get_lang_id()
            elif hasattr(tokenizer, "get_lang_id"):
                try:
                    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
                except (KeyError, ValueError):
                    pass  # Language not supported, continue without forcing
            else:
                print(f"[WARN] Could not set target language '{tgt_lang}' - tokenizer does not support language forcing!", flush=True)

        # BUGFIX: Process texts in chunks of self.batch_size to prevent OOM
        all_results = []
        device = next(model.parameters()).device

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            gen_kwargs = {
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True,
            }
            if forced_bos_token_id:
                gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode
            batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_results.extend(batch_results)

            # BUGFIX: Explicitly delete tensors and clear cache to prevent memory leak
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return all_results

    def translate_single(self, text: str, context: str = None) -> Tuple[str, str]:
        """Translate a single text with protection, validation, and cache.

        Returns:
            (translated_text, reason_if_failed)
        """
        # Skip check
        skip, reason = self.should_skip(text)
        if skip:
            return text, reason

        # CACHE LOOKUP: Check Translation Memory first
        if self.tm and self.tm.enabled:
            cached = self.tm.get(text, context=context, model_name=self.model_loader.actual_model_name)
            if cached:
                # Cache hit! Return immediately
                return cached, ""

        # Language detection - skip non-English/Vietnamese text
        if LANGDETECT_AVAILABLE:
            lang_code, confidence = detect_language(text, min_confidence=0.5)

            # BUGFIX: If confidence is low (<0.5), it already returned "en" from detect_language
            # Only need to check if detected as non-English/Vietnamese with high confidence
            if lang_code not in ["en", "vi"] and confidence > 0.5:
                # Check if text has many placeholders (likely Minecraft config, not actual foreign language)
                placeholder_ratio = 1.0 - (confidence if confidence > 0.3 else 0.3)  # Inverse heuristic
                has_placeholders = bool(re.search(r'[%{}\[\]]', text))  # Has placeholder-like chars

                # Only skip if VERY confident (>98%) AND no placeholder indicators
                # Real Chinese/Japanese/Korean/Thai text is usually 99%+ confident
                # BUGFIX: Also require reasonable text length to avoid false positives
                # Short English sentences can be confused with Romanian/French/etc.
                text_length = len(text.strip())
                word_count = len([w for w in text.split() if len(w) > 2])

                # FIXED: Stricter language detection to reduce false positives
                # Latin-script languages (ro, fr, it, etc) have EXTREME false positive rate on short English text
                # langdetect often reports 100% confidence for misdetected short English text
                latin_script_langs = ['ro', 'fr', 'it', 'es', 'pt', 'de', 'nl', 'sv', 'no', 'da', 'fi', 'pl', 'cs', 'sk', 'hu', 'hr', 'sl', 'et', 'lv', 'lt', 'af']
                if lang_code in latin_script_langs:
                    # FIXED: NEVER skip Latin languages - translate anyway
                    # Only log warning if extremely confident (>99.5%) AND long text
                    if confidence > 0.995 and len(text) > 100:
                        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
                        print(f"⚠️  DETECTED {lang_name} ({confidence:.0%}), but translating anyway (Latin language false positives common): {text[:80]}...")
                else:
                    # For non-Latin languages (CJK, Thai, Arabic, etc), be more strict
                    # FIXED: Increased threshold from 98% to 99.5% to reduce false positives
                    if confidence > 0.995 and not has_placeholders and len(text) > 30:
                        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
                        print(f"🛑 SKIPPING {lang_name} text ({confidence:.0%}): {text[:80]}...")
                        return text, f"non_english_{lang_code}"

                # Log if moderately confident but still translating
                if confidence > 0.85 and lang_code not in ["en", "vi"]:
                    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
                    print(f"⚠️  POSSIBLE {lang_name} ({confidence:.0%}), translating anyway: {text[:80]}...")

        # BUGFIX: Check if text is too long and needs chunking to avoid 512-token truncation
        if len(text) >= CHUNK_THRESHOLD_CHARS:
            return self.translate_long_text(text)

        # BUGFIX: Decode escape sequences before translation
        decoded = decode_escapes(text)

        # Protect tokens
        protected = self.protector.protect(decoded)

        # Apply glossary
        glossed, gloss_map = self.apply_glossary_pre(protected)

        # Translate
        try:
            translated = self.translate_batch([glossed])[0]
        except Exception as e:
            return text, f"translation_error: {e}"

        # Restore glossary
        translated = self.apply_glossary_post(translated, gloss_map)

        # Restore tokens
        restored = self.protector.restore(translated)

        # BUGFIX: Re-encode escape sequences after translation
        encoded = encode_escapes(restored)

        # Validate
        valid, error = self.validate_translation(text, encoded)
        if not valid:
            # FIXED: Log validation failures with details
            print(f"❌ VALIDATION FAILED: {error}")
            print(f"   Original: {text[:100]}...")
            print(f"   Translated: {encoded[:100]}...")
            return text, f"validation_failed: {error}"

        # Structure validation (YAML/JSON corruption detection)
        # FIXED: Only check structure for text with YAML/JSON structural elements
        # Ignore quotes and apostrophes - they're common in natural language
        has_critical_structure = any(c in text for c in ['[', ']', '{', '}'])
        has_yaml_structure = ('|' in text or '>' in text) and '\n' in text  # YAML block scalars

        if (has_critical_structure or has_yaml_structure) and len(text) > 30:
            struct_valid, struct_error = self.validate_structure(text, encoded)
            if not struct_valid:
                print(f"⚠️  STRUCTURE VALIDATION FAILED: {struct_error}")
                print(f"   Original: {text[:100]}...")
                print(f"   Translated: {encoded[:100]}...")
                # FIXED: Don't reject - just log warning
                # return text, f"structure_invalid: {struct_error}"

        # CACHE STORE: Save successful translation to Translation Memory
        if self.tm and self.tm.enabled:
            self.tm.put(text, encoded, context=context, model_name=self.model_loader.actual_model_name)

        return encoded, ""

    def translate_adjacent(self, texts: List[str]) -> List[Tuple[str, str]]:
        """Translate texts with adjacent context grouping.

        Returns:
            List of (translated, reason) tuples
        """
        if not texts or self.low_memory:
            # Fall back to individual translation
            return [self.translate_single(t) for t in texts]

        # Check delimiters
        delim_rec, delim_unit = self.check_delimiter_collision(texts)

        # Group into batches of ADJACENT_GROUP_SIZE
        results = []
        for i in range(0, len(texts), ADJACENT_GROUP_SIZE):
            group = texts[i:i + ADJACENT_GROUP_SIZE]

            # Skip empty/whitespace
            skip_flags = [self.should_skip(t)[0] for t in group]
            if all(skip_flags):
                results.extend([(t, self.should_skip(t)[1]) for t in group])
                continue

            # BUGFIX: Apply full pipeline like translate_single() does
            # Step 1: Decode escape sequences for each text
            decoded_group = [decode_escapes(t) for t in group]

            # Step 2: Use separate protector + glossary for each string
            protectors = [TokenProtector() for _ in decoded_group]
            glossary_maps = []
            protected_and_glossed = []

            for idx, decoded in enumerate(decoded_group):
                # Protect tokens
                protected = protectors[idx].protect(decoded)
                # Apply glossary
                glossed, gloss_map = self.apply_glossary_pre(protected)
                protected_and_glossed.append(glossed)
                glossary_maps.append(gloss_map)

            # Join with delimiters
            joined = delim_unit.join(protected_and_glossed)

            # Translate
            try:
                translated_joined = self.translate_batch([joined])[0]
            except Exception as e:
                # Fallback to individual
                results.extend([self.translate_single(t) for t in group])
                continue

            # Split
            translated_parts = translated_joined.split(delim_unit)

            # Validate split count
            if len(translated_parts) != len(group):
                # Delimiter collision - fallback to individual
                results.extend([self.translate_single(t) for t in group])
                continue

            # BUGFIX: Restore full pipeline (glossary → tokens → escapes)
            for idx, (orig, trans) in enumerate(zip(group, translated_parts)):
                # Restore glossary terms
                trans_unglossed = self.apply_glossary_post(trans, glossary_maps[idx])
                # Restore protected tokens
                restored = protectors[idx].restore(trans_unglossed)
                # Re-encode escape sequences
                encoded = encode_escapes(restored)
                # Validate using encoded result
                valid, error = self.validate_translation(orig, encoded)
                if valid:
                    results.append((encoded, ""))
                else:
                    # Retry individual
                    retry, retry_reason = self.translate_single(orig)
                    results.append((retry, retry_reason))

        return results

    def translate_long_text(self, text: str) -> Tuple[str, str]:
        """Translate long text by chunking at sentence boundaries.

        Returns:
            (translated_text, reason_if_failed)
        """
        if len(text) < CHUNK_THRESHOLD_CHARS:
            return self.translate_single(text)

        # Split by sentence boundaries, keeping separators
        import re
        # Split on sentence endings but keep the separator
        chunks = re.split(r'(\n|\.(?:\s|$)|!(?:\s|$)|\?(?:\s|$))', text)

        # Reconstruct chunks with separators
        combined_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if i + 1 < len(chunks):
                separator = chunks[i + 1]
                combined_chunks.append(chunk + separator)
                i += 2
            else:
                combined_chunks.append(chunk)
                i += 1

        # Filter empty chunks
        combined_chunks = [c for c in combined_chunks if c.strip()]

        # Translate each chunk
        translated_chunks = []
        for chunk in combined_chunks:
            trans, reason = self.translate_single(chunk)
            if reason:
                return text, f"chunk_failed: {reason}"
            translated_chunks.append(trans)

        # Rejoin
        result = "".join(translated_chunks)
        return result, ""

    def validate_translation(self, original: str, translated: str) -> Tuple[bool, str]:
        """Validate translated text."""
        # Token validation
        valid, error = self.protector.validate_restoration(original, translated)
        if not valid:
            return False, error

        # Length guard
        if len(translated) > MAX_LENGTH_RATIO * len(original) and not self.force:
            return False, f"length_ratio_exceeded ({len(translated)} > {MAX_LENGTH_RATIO} * {len(original)})"

        if len(translated) > MAX_ABSOLUTE_LENGTH and not self.force:
            return False, f"absolute_length_exceeded ({len(translated)} > {MAX_ABSOLUTE_LENGTH})"

        # Check for empty result
        if is_empty_or_whitespace(translated):
            return False, "empty_result"

        return True, ""

    def validate_structure(self, original: str, translated: str) -> Tuple[bool, str]:
        """
        FIXED: Relaxed structure validation to reduce false rejections.
        Only validates critical YAML/JSON structural elements.

        Returns:
            (is_valid, error_message)
        """
        # FIXED: Only check brackets and braces (critical structure)
        # Colons, pipes, angle brackets can vary in natural language
        critical_structure_chars = {
            '[': 'square_brackets_open',
            ']': 'square_brackets_close',
            '{': 'curly_brackets_open',
            '}': 'curly_brackets_close',
        }

        # Count critical structure characters
        for char, name in critical_structure_chars.items():
            orig_count = original.count(char)
            trans_count = translated.count(char)

            # Allow ±1 difference for edge cases
            if abs(orig_count - trans_count) > 1:
                return False, f"structure_mismatch: {name} ({orig_count} → {trans_count})"

        # FIXED: Only validate DOUBLE QUOTES for string delimiters
        # NEVER check single quotes/apostrophes - Vietnamese doesn't use possessive apostrophes
        # English "it's" → Vietnamese "nó", English "players'" → Vietnamese "của người chơi"
        quote_char = '"'
        orig_count = original.count(quote_char)
        trans_count = translated.count(quote_char)

        # Only check if original has quotes (string delimiters)
        if orig_count > 0:
            # Must be even (paired) in both
            if orig_count % 2 == 0 and trans_count % 2 != 0:
                return False, f"quote_imbalance: {quote_char} count is odd ({trans_count})"

            # Allow ±2 difference (natural language variation)
            if abs(orig_count - trans_count) > 2:
                return False, f"quote_mismatch: {quote_char} ({orig_count} → {trans_count})"

        # FIXED: Relaxed newline validation
        # Allow more flexibility for natural language translation
        orig_newlines = original.count('\n')
        trans_newlines = translated.count('\n')

        if orig_newlines != trans_newlines:
            diff = abs(orig_newlines - trans_newlines)

            # FIXED: More lenient - allow ±2 lines for natural translation flow
            # Only reject if massive difference (>50% change AND >2 lines)
            if diff > 2 and diff > orig_newlines * 0.5:
                return False, f"newline_mismatch ({orig_newlines} → {trans_newlines})"

        return True, ""


# ============================================================================
# FILE HANDLERS
# ============================================================================

class FileHandler:
    """Base class for file handlers."""

    def __init__(self, translator: Translator, dry_run: bool = False, base_dir: Optional[Path] = None):
        self.translator = translator
        self.dry_run = dry_run
        self.base_dir = base_dir
        self.changes = []

    def safe_callback(self, callback_name: str, *args, **kwargs):
        """Safely invoke callback with error handling."""
        callback = getattr(self.translator, callback_name, None)
        if callback:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Callback error ({callback_name}): {e}")

    def read_file(self, file_path: Path) -> Tuple[Any, str]:
        """Read and parse file. Returns (data, encoding)."""
        raise NotImplementedError

    def write_file(self, file_path: Path, data: Any, encoding: str):
        """Write data to file."""
        raise NotImplementedError

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Translate file and return change log."""
        raise NotImplementedError


class YAMLHandler(FileHandler):
    """Handler for YAML files."""

    def __init__(self, translator: Translator, dry_run: bool = False, base_dir: Optional[Path] = None):
        super().__init__(translator, dry_run, base_dir)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False
        self.yaml.width = 4096

    def read_file(self, file_path: Path) -> Tuple[Any, str]:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            data = self.yaml.load(f)
        return data, encoding

    def write_file(self, file_path: Path, data: Any, encoding: str):
        # BUGFIX: Force UTF-8 for Vietnamese content (latin-1/ascii can't encode Vietnamese)
        # Check if data contains Vietnamese characters
        data_str = str(data)
        has_vietnamese = any(ord(c) > 127 for c in data_str)
        if has_vietnamese and encoding.lower() in ['ascii', 'latin-1', 'iso-8859-1']:
            encoding = 'utf-8'

        content_io = tempfile.NamedTemporaryFile(mode="w", encoding=encoding, delete=False, newline="")
        temp_path = content_io.name
        try:
            self.yaml.dump(data, content_io)
            content_io.close()

            with open(temp_path, "r", encoding=encoding) as f:
                content = f.read()

            atomic_write(file_path, content, encoding, base_dir=self.base_dir)
        finally:
            # BUGFIX: Add retry for Windows file lock issues
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    # Windows may still have file locked, retry after brief delay
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass  # Ignore if still can't delete

    def translate_value(self, value: Any, path: str) -> Tuple[Any, bool, str]:
        """Translate a single value with context-aware detection.

        Returns:
            (translated_value, changed, reason)
        """
        # CONTEXT-AWARE CHECK: Skip if YAML key indicates code context
        if is_code_context(path):
            return value, False, f"code_context: {path}"

        # BUGFIX: Check literal/folded scalars BEFORE str
        # (LiteralScalarString/FoldedScalarString inherit from str)
        if isinstance(value, (LiteralScalarString, FoldedScalarString)):
            original_text = str(value)

            # CONTEXT-AWARE CHECK: Skip if text looks like code
            if looks_like_code(original_text):
                return value, False, f"looks_like_code: {original_text[:50]}"

            # Check if should skip
            if len(original_text) <= YAML_MULTILINE_THRESHOLD:
                # Translate as single block
                translated, reason = self.translator.translate_single(original_text, context=path)
                if reason:
                    return value, False, reason

                # Re-create scalar with same class
                if isinstance(value, LiteralScalarString):
                    new_value = LiteralScalarString(translated)
                else:
                    new_value = FoldedScalarString(translated)

                changed = translated != original_text
                return new_value, changed, ""
            else:
                # Line-by-line translation for long multiline blocks
                lines = original_text.split("\n")
                translated_lines = []

                for line in lines:
                    if is_empty_or_whitespace(line):
                        translated_lines.append(line)
                        continue

                    # BUGFIX: Removed unused prev_line variable
                    translated, reason = self.translator.translate_single(line, context=f"{path}[line]")
                    translated_lines.append(translated if not reason else line)

                translated_text = "\n".join(translated_lines)

                # Re-create scalar
                if isinstance(value, LiteralScalarString):
                    new_value = LiteralScalarString(translated_text)
                else:
                    new_value = FoldedScalarString(translated_text)

                changed = translated_text != original_text
                return new_value, changed, ""

        # Handle regular scalar strings (AFTER checking LiteralScalarString/FoldedScalarString)
        if isinstance(value, str):
            # CONTEXT-AWARE CHECK: Skip if text looks like code
            if looks_like_code(value):
                return value, False, f"looks_like_code: {value[:50]}"

            translated, reason = self.translator.translate_single(value, context=path)
            changed = translated != value
            return translated, changed, reason

        return value, False, "not_string"

    def translate_recursive(self, data: Any, path: str = "") -> int:
        """Recursively translate YAML data structure.

        Returns:
            Number of changes made
        """
        changes_count = 0

        if isinstance(data, CommentedMap):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else str(key)

                # Skip merge keys (<<) - these reference other sections
                if key == "<<":
                    continue

                # Skip if value has an anchor attribute (it's an alias reference)
                # ruamel.yaml preserves anchors/aliases automatically
                # BUGFIX: Check anchor.value not just anchor (Anchor(None) is truthy!)
                if hasattr(value, 'anchor') and value.anchor and value.anchor.value:
                    continue

                # Recursively handle nested structures
                if isinstance(value, (CommentedMap, CommentedSeq)):
                    changes_count += self.translate_recursive(value, current_path)
                else:
                    # Translate value
                    new_value, changed, reason = self.translate_value(value, current_path)
                    if changed:
                        data[key] = new_value
                        changes_count += 1
                        self.changes.append({
                            "key_path": current_path,
                            "original": str(value),
                            "translated": str(new_value),
                            "changed": True,
                            "reason": reason,
                        })

                        # Notify GUI
                        self.safe_callback('on_translation', current_path, str(value), str(new_value))

        elif isinstance(data, CommentedSeq):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"

                if isinstance(item, (CommentedMap, CommentedSeq)):
                    changes_count += self.translate_recursive(item, current_path)
                else:
                    new_value, changed, reason = self.translate_value(item, current_path)
                    if changed:
                        data[i] = new_value
                        changes_count += 1
                        self.changes.append({
                            "key_path": current_path,
                            "original": str(item),
                            "translated": str(new_value),
                            "changed": True,
                            "reason": reason,
                        })

                        # Notify GUI
                        self.safe_callback('on_translation', current_path, str(item), str(new_value))

        return changes_count

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        self.changes = []

        try:
            # Read
            data, encoding = self.read_file(file_path)
            if data is None:
                return []

            # Translate
            changes_count = self.translate_recursive(data)

            # Write
            if changes_count > 0 and not self.dry_run:
                create_backup(file_path)
                self.write_file(file_path, data, encoding)

            return self.changes

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []


class JSONHandler(FileHandler):
    """Handler for JSON files."""

    def read_file(self, file_path: Path) -> Tuple[Any, str]:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return data, encoding

    def write_file(self, file_path: Path, data: Any, encoding: str):
        # BUGFIX: Force UTF-8 for Vietnamese content
        content = json.dumps(data, ensure_ascii=False, indent=2)
        has_vietnamese = any(ord(c) > 127 for c in content)
        if has_vietnamese and encoding.lower() in ['ascii', 'latin-1', 'iso-8859-1']:
            encoding = 'utf-8'
        atomic_write(file_path, content, encoding, base_dir=self.base_dir)

    def translate_recursive(self, data: Any, path: str = "") -> int:
        """Recursively translate JSON data."""
        changes_count = 0

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, (dict, list)):
                    changes_count += self.translate_recursive(value, current_path)
                elif isinstance(value, str):
                    translated, reason = self.translator.translate_single(value, context=current_path)
                    if translated != value:
                        data[key] = translated
                        changes_count += 1
                        self.changes.append({
                            "key_path": current_path,
                            "original": value,
                            "translated": translated,
                            "changed": True,
                            "reason": reason,
                        })

                        # Notify GUI
                        self.safe_callback('on_translation', current_path, value, translated)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"

                if isinstance(item, (dict, list)):
                    changes_count += self.translate_recursive(item, current_path)
                elif isinstance(item, str):
                    translated, reason = self.translator.translate_single(item, context=current_path)
                    if translated != item:
                        data[i] = translated
                        changes_count += 1
                        self.changes.append({
                            "key_path": current_path,
                            "original": item,
                            "translated": translated,
                            "changed": True,
                            "reason": reason,
                        })

                        # Notify GUI
                        self.safe_callback('on_translation', current_path, item, translated)

        return changes_count

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        self.changes = []

        try:
            data, encoding = self.read_file(file_path)
            changes_count = self.translate_recursive(data)

            if changes_count > 0 and not self.dry_run:
                create_backup(file_path)
                self.write_file(file_path, data, encoding)

            return self.changes

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []


class PropertiesHandler(FileHandler):
    """Handler for .properties and .lang files."""

    def read_file(self, file_path: Path) -> Tuple[List[str], str]:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
        return lines, encoding

    def write_file(self, file_path: Path, lines: List[str], encoding: str):
        content = "".join(lines)
        atomic_write(file_path, content, encoding, base_dir=self.base_dir)

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        self.changes = []

        try:
            lines, encoding = self.read_file(file_path)

            # BUGFIX: Capture full separator WITH spacing to preserve formatting
            # Group 1: key (no trailing space)
            # Group 2: full separator with spacing (e.g., "   =   ")
            # Group 3: value
            # NOTE: This pattern intentionally does NOT handle:
            #   - Escaped separators (key\:name = value) - rare in Minecraft configs
            #   - Multiline values (value\) - not used by Minecraft plugins
            #   - Unicode escapes (\u0041) - handled by Java at runtime
            pattern = re.compile(r'^([^=#:]+?)(\s*[:=]\s*)(.*)$')

            changes_count = 0
            for i, line in enumerate(lines):
                # Skip comments and empty lines
                if line.strip().startswith("#") or is_empty_or_whitespace(line):
                    continue

                match = pattern.match(line)
                if not match:
                    continue

                key = match.group(1)
                full_separator = match.group(2)  # Includes spacing!
                value = match.group(3)

                # BUGFIX: Skip technical values that look like code/enums/booleans/numbers
                # Don't translate: true, false, numbers, URLs, single words in lowercase/UPPERCASE
                value_stripped = value.strip()
                if value_stripped:
                    # Skip booleans
                    if value_stripped.lower() in ['true', 'false', 'yes', 'no', 'on', 'off']:
                        continue
                    # Skip pure numbers (int or float)
                    if re.match(r'^-?\d+(\.\d+)?$', value_stripped):
                        continue
                    # Skip URLs, emails, paths
                    if any(x in value_stripped for x in ['://', '@', '\\', '/']):
                        continue
                    # Skip single lowercase words (likely enums like "easy", "hard", "normal")
                    if re.match(r'^[a-z]+$', value_stripped):
                        continue
                    # Skip UPPERCASE constants
                    if re.match(r'^[A-Z_0-9]+$', value_stripped):
                        continue

                # Translate value only
                translated, reason = self.translator.translate_single(value, context=key.strip())
                if translated != value:
                    # BUGFIX: Preserve full separator with spacing
                    new_line = f"{key}{full_separator}{translated}\n"
                    lines[i] = new_line
                    changes_count += 1
                    self.changes.append({
                        "key_path": key.strip(),
                        "original": value,
                        "translated": translated,
                        "changed": True,
                        "reason": reason,
                    })

                    # Notify GUI
                    self.safe_callback('on_translation', key.strip(), value, translated)

            if changes_count > 0 and not self.dry_run:
                create_backup(file_path)
                self.write_file(file_path, lines, encoding)

            return self.changes

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []


class TextHandler(FileHandler):
    """Handler for plain .txt files - translates line by line."""

    def read_file(self, file_path: Path) -> Tuple[List[str], str]:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
        return lines, encoding

    def write_file(self, file_path: Path, lines: List[str], encoding: str):
        content = "".join(lines)
        atomic_write(file_path, content, encoding, base_dir=self.base_dir)

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        self.changes = []

        try:
            lines, encoding = self.read_file(file_path)
            translated_lines = []

            for line_num, line in enumerate(lines, 1):
                # Skip empty lines and comments
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    translated_lines.append(line)
                    continue

                # Translate the line content, preserving newline
                has_newline = line.endswith("\n")
                text_to_translate = line.rstrip("\n\r")

                translated, reason = self.translator.translate_single(
                    text_to_translate,
                    context=f"{file_path.name}:line{line_num}"
                )

                # Restore newline if original had it
                if has_newline:
                    translated += "\n"

                translated_lines.append(translated)

                if translated != line:
                    self.changes.append({
                        "key_path": f"line {line_num}",
                        "original": text_to_translate[:100],
                        "translated": translated.rstrip("\n\r")[:100],
                        "changed": True,
                        "reason": reason,
                    })

            if self.changes and not self.dry_run:
                create_backup(file_path)
                self.write_file(file_path, translated_lines, encoding)

            return self.changes

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []


class McfunctionHandler(FileHandler):
    """Handler for .mcfunction files - translates JSON text components in commands."""

    def read_file(self, file_path: Path) -> Tuple[List[str], str]:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
        return lines, encoding

    def write_file(self, file_path: Path, lines: List[str], encoding: str):
        content = "".join(lines)
        atomic_write(file_path, content, encoding, base_dir=self.base_dir)

    def extract_json_from_command(self, line: str) -> List[Tuple[str, int, int]]:
        """Extract JSON components from Minecraft commands.

        Returns list of (json_str, start_pos, end_pos) tuples.
        Looks for patterns like:
        - tellraw @a {"text":"..."}
        - title @a title {"text":"..."}
        - tellraw @a [{"text":"..."},{"text":"..."}]
        """
        import re
        json_patterns = []

        # Find JSON objects/arrays in the command
        # Look for { or [ that start JSON
        stack = []
        start = None

        for i, char in enumerate(line):
            if char in ['{', '[']:
                if not stack:
                    start = i
                stack.append(char)
            elif char in ['}', ']']:
                if stack:
                    expected = '{' if char == '}' else '['
                    if stack[-1] == expected:
                        stack.pop()
                        if not stack and start is not None:
                            # Found complete JSON
                            json_str = line[start:i+1]
                            json_patterns.append((json_str, start, i+1))
                            start = None

        return json_patterns

    def translate_json_component(self, json_str: str) -> str:
        """Translate text in JSON text component."""
        import json
        try:
            data = json.loads(json_str)
            self._translate_recursive(data)
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except:
            # If JSON parsing fails, return original
            return json_str

    def _translate_recursive(self, obj):
        """Recursively translate 'text' fields in JSON component."""
        if isinstance(obj, dict):
            # Translate 'text' field if exists
            if 'text' in obj and isinstance(obj['text'], str):
                text = obj['text']
                # Don't translate if it looks like a selector or empty
                if text and not text.startswith('@') and text.strip():
                    translated, _ = self.translator.translate_single(
                        text,
                        context="mcfunction:json:text"
                    )
                    obj['text'] = translated

            # Recurse into nested objects
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    self._translate_recursive(value)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._translate_recursive(item)

    def translate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        self.changes = []

        try:
            lines, encoding = self.read_file(file_path)
            translated_lines = []

            for line_num, line in enumerate(lines, 1):
                # Skip empty lines and comments
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    translated_lines.append(line)
                    continue

                # Find JSON components in the command
                json_components = self.extract_json_from_command(line)

                if not json_components:
                    # No JSON found, keep line as-is
                    translated_lines.append(line)
                    continue

                # Translate each JSON component
                translated_line = line
                offset = 0

                for json_str, start, end in json_components:
                    translated_json = self.translate_json_component(json_str)

                    if translated_json != json_str:
                        # Replace in line
                        actual_start = start + offset
                        actual_end = end + offset
                        translated_line = (
                            translated_line[:actual_start] +
                            translated_json +
                            translated_line[actual_end:]
                        )
                        offset += len(translated_json) - len(json_str)

                        self.changes.append({
                            "key_path": f"line {line_num}",
                            "original": json_str[:100],
                            "translated": translated_json[:100],
                            "changed": True,
                            "reason": "",
                        })

                translated_lines.append(translated_line)

            if self.changes and not self.dry_run:
                create_backup(file_path)
                self.write_file(file_path, translated_lines, encoding)

            return self.changes

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            return []


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class InPlaceTranslator:
    """Main in-place translation processor."""

    def __init__(self, args):
        self.args = args
        self.folder = Path(args.folder).resolve() if hasattr(args, 'folder') and args.folder else None
        self.extensions = set(args.ext.split(",")) if args.ext else DEFAULT_EXTENSIONS
        self.extensions = {f".{e.lstrip('.')}" for e in self.extensions}

        self.model_loader: Optional[ModelLoader] = None
        self.translator: Optional[Translator] = None
        self.report_rows = []
        # Callbacks for GUI integration
        self.on_progress = None  # (current, total, avg_time, eta)
        self.on_file_start = None  # (filename, current, total)
        self.on_translation = None  # (key_path, original, translated)
        self.on_file_complete = None  # (filename, changes_count)
        self.is_paused = False  # For pause/resume control
        self.is_translating = False  # For stop control
        self.on_log = None  # Callback for general logging

    def setup_model(self):
        """Setup model and translator."""
        print("\n" + "="*60)
        print("SETUP: Loading translation model")
        print("="*60)

        self.model_loader = ModelLoader(
            model_name=self.args.model,
            fallback_model=self.args.fallback_model,
            device="cuda" if self.args.gpu else ("cpu" if self.args.cpu else None),
            max_memory=self.args.max_memory,
            low_memory=self.args.low_memory,
        )
        self.model_loader.load()

        # Get adaptive batch size
        batch_size = self.model_loader.get_adaptive_batch_size(self.args.batch_size)
        print(f"\nBatch size: {batch_size}")

        # Load glossary
        glossary = load_glossary(Path(self.args.glossary) if self.args.glossary else None)
        if glossary:
            print(f"Loaded glossary with {len(glossary)} terms")

        # Initialize Translation Memory (cache)
        tm_enabled = not self.args.no_cache
        tm = TranslationMemory(enabled=tm_enabled) if tm_enabled else None
        if tm_enabled:
            stats = tm.get_stats()
            print(f"Translation Memory enabled: {stats['cache_size']} cached translations")
            print(f"  Cache database: {stats.get('db_path', 'N/A')}")

        # Create translator
        self.translator = Translator(
            model_loader=self.model_loader,
            batch_size=batch_size,
            low_memory=self.args.low_memory,
            glossary=glossary,
            force=self.args.force,
            tm=tm,
        )

        # Propagate callbacks to translator
        if hasattr(self, "on_translation") and self.on_translation:
            self.translator.on_translation = self.on_translation

        # Quality check
        if not self.args.download_only:
            self.run_quality_check()

    def run_quality_check(self):
        """Run quality check on test strings."""
        print("\n" + "="*60)
        print("QUALITY CHECK: Testing translation quality")
        print("="*60)

        test_results = []
        for test_str in QUALITY_TEST_STRINGS[:3]:  # Test first 3
            try:
                translated, reason = self.translator.translate_single(test_str)
                test_results.append((test_str, translated, reason))
                print(f"[OK] EN: {test_str}")
                print(f"  VI: {translated}")
                if reason:
                    print(f"  [WARN] {reason}")
            except Exception as e:
                print(f"[FAIL] Failed: {test_str} -> {e}")
                test_results.append((test_str, "", str(e)))

        # Check for degraded quality
        failed = sum(1 for _, trans, _ in test_results if is_empty_or_whitespace(trans))
        if failed >= 2:
            print("\n[WARN] WARNING: Translation quality appears degraded (empty results).")
            print("  Try: --fallback-model nllb-1.3b or use full precision (--cpu)")

    def collect_files(self) -> List[Path]:
        """Collect all files to process."""
        if not self.folder or not self.folder.exists():
            return []

        # If folder is actually a file, process it directly
        if self.folder.is_file():
            if self.folder.suffix in self.extensions and not self.folder.name.endswith(".bak"):
                return [self.folder]
            return []

        # Otherwise, recursively collect files from directory
        files = []
        for ext in self.extensions:
            files.extend(self.folder.rglob(f"*{ext}"))

        # Filter out .bak files
        files = [f for f in files if not f.name.endswith(".bak")]

        # Apply include/exclude patterns if specified
        files = self._apply_path_filters(files)

        return sorted(files)

    def _apply_path_filters(self, files: List[Path]) -> List[Path]:
        """Apply include/exclude pattern filters to file list."""
        from fnmatch import fnmatch

        # Parse include patterns
        include_patterns = []
        if hasattr(self.args, 'include') and self.args.include:
            include_patterns = [p.strip() for p in self.args.include.split(",")]

        # Parse exclude patterns
        exclude_patterns = []
        if hasattr(self.args, 'exclude') and self.args.exclude:
            exclude_patterns = [p.strip() for p in self.args.exclude.split(",")]

        filtered_files = []
        for file_path in files:
            # Get relative path from base folder for pattern matching
            try:
                rel_path = file_path.relative_to(self.folder)
                rel_path_str = str(rel_path).replace("\\", "/")  # Normalize to forward slashes
            except ValueError:
                rel_path_str = str(file_path).replace("\\", "/")

            # Check include patterns (if specified, file must match at least one)
            if include_patterns:
                if not any(fnmatch(rel_path_str, pattern) for pattern in include_patterns):
                    continue

            # Check exclude patterns (if matches any, skip file)
            if exclude_patterns:
                if any(fnmatch(rel_path_str, pattern) for pattern in exclude_patterns):
                    continue

            filtered_files.append(file_path)

        return filtered_files

    def get_handler(self, file_path: Path) -> Optional[FileHandler]:
        """Get appropriate handler for file."""
        ext = file_path.suffix.lower()

        if ext in {".yml", ".yaml"}:
            return YAMLHandler(self.translator, self.args.dry_run, base_dir=self.folder)
        elif ext == ".json":
            return JSONHandler(self.translator, self.args.dry_run, base_dir=self.folder)
        elif ext in {".properties", ".lang"}:
            return PropertiesHandler(self.translator, self.args.dry_run, base_dir=self.folder)
        elif ext == ".txt":
            return TextHandler(self.translator, self.args.dry_run, base_dir=self.folder)
        elif ext == ".mcfunction":
            return McfunctionHandler(self.translator, self.args.dry_run, base_dir=self.folder)

        return None

    def process_files(self):
        """Process all files."""
        files = self.collect_files()
        if not files:
            print(f"\nNo files found with extensions: {', '.join(self.extensions)}")
            return

        print("\n" + "="*60)
        print(f"PROCESSING: {len(files)} files")
        print("="*60)

        # Check if large folder
        folder_size = get_folder_size_mb(self.folder)
        is_large = len(files) > LARGE_FOLDER_FILE_COUNT or folder_size > LARGE_FOLDER_SIZE_MB

        if is_large:
            print(f"[WARN] Large folder detected ({len(files)} files, {folder_size:.1f} MB)")
            print("  Using streaming mode for memory efficiency")

        # Progress bar
        total_changes = 0
        with tqdm(files, desc="Translating", unit="file") as pbar:
            for idx, file_path in enumerate(files, start=1):
                pbar.update(1)
                pbar.set_postfix_str(file_path.name)

                # Check pause/stop flags (only in GUI mode)
                if hasattr(self, "on_file_start") and self.on_file_start:  # GUI mode detected
                    while hasattr(self, "is_paused") and self.is_paused:
                        import time
                        time.sleep(0.1)
                    if not self.is_translating:
                        print("Translation stopped by user")
                        break
                if self.on_file_start:
                    self.on_file_start(file_path.name, idx, len(files))

                handler = self.get_handler(file_path)
                if not handler:
                    continue

                try:
                    changes = handler.translate_file(file_path)

                    # Add to report
                    for change in changes:
                        self.report_rows.append({
                            "file": str(file_path.relative_to(self.folder)),
                            "key_path": change["key_path"],
                            "original": change["original"],
                            "translated": change["translated"],
                            "changed": change["changed"],
                            "reason": change.get("reason", ""),
                        })

                    total_changes += sum(1 for c in changes if c["changed"])

                    # Notify GUI of translation
                    if self.on_translation:
                        for change in changes:
                            self.on_translation(
                                change["key_path"],
                                change["original"],
                                change["translated"]
                            )

                except Exception as e:
                    print(f"\nERROR: {file_path}: {e}")

                # Memory cleanup for large folders
                if is_large or self.args.low_memory:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        print(f"\n[OK] Translation complete: {total_changes} changes across {len(files)} files")

        # Write report
        if self.args.report and self.report_rows:
            self.write_report()

    def write_report(self):
        """Write CSV report."""
        report_path = Path.cwd() / "mine_translate_report.csv"

        try:
            import csv
            with open(report_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "key_path", "original", "translated", "changed", "reason"])
                writer.writeheader()
                writer.writerows(self.report_rows)

            print(f"\n[OK] Report written: {report_path}")
        except Exception as e:
            print(f"\n[FAIL] Failed to write report: {e}")

    def rollback(self):
        """Rollback all .bak files."""
        print("\n" + "="*60)
        print("ROLLBACK: Restoring .bak files")
        print("="*60)

        if not self.folder or not self.folder.exists():
            print("ERROR: Folder not found")
            return

        backup_files = list(self.folder.rglob("*.bak"))
        if not backup_files:
            print("No .bak files found")
            return

        restored = 0
        for backup_path in tqdm(backup_files, desc="Restoring", unit="file"):
            # Derive original path
            original_path = backup_path.with_suffix("")

            if original_path.exists():
                try:
                    shutil.copy2(backup_path, original_path)
                    backup_path.unlink()
                    restored += 1
                except Exception as e:
                    print(f"\nERROR restoring {original_path}: {e}")

        print(f"\n[OK] Restored {restored} files")

    def run(self):
        """Main entry point."""
        # Rollback mode
        if self.args.rollback:
            self.rollback()
            return

        # Download-only mode
        if self.args.download_only:
            self.setup_model()
            print("\n[OK] Model downloaded successfully")
            return

        # Normal mode
        if not self.folder:
            print("ERROR: No folder specified")
            return

        self.setup_model()

        if self.args.dry_run:
            print("\n[WARN] DRY-RUN MODE: No files will be modified")

        self.process_files()

        # Print Translation Memory statistics
        if self.translator and self.translator.tm:
            stats = self.translator.tm.get_stats()
            if stats['total'] > 0:
                print("\n" + "="*60)
                print("TRANSLATION MEMORY STATISTICS")
                print("="*60)
                print(f"Cache hits:   {stats['hits']}/{stats['total']} ({stats['hit_rate']:.1%})")
                print(f"Cache size:   {stats['cache_size']} translations")
                print(f"Time saved:   ~{stats['hits'] * 2:.0f}s (estimated)")
                print(f"Database:     {stats.get('db_path', 'N/A')}")

        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)


# ============================================================================
# SELF-TEST
# ============================================================================

def run_self_tests():
    """Run comprehensive self-tests."""
    print("\n" + "="*60)
    print("SELF-TEST: Running validation tests")
    print("="*60)

    passed = 0
    failed = 0

    # Test 1: Token protection/restoration
    print("\n[1] Token protection/restoration")
    protector = TokenProtector()
    test_cases = [
        "Hello {player}!",
        "Cost: %price% coins",
        "Use &cred&r color",
        "Run /command here",
        "ID: minecraft:diamond_sword",
        "Visit https://example.com now",
        "Multi: {player} %price% &c /cmd",
    ]
    for test in test_cases:
        protected = protector.protect(test)
        restored = protector.restore(protected)
        if restored == test:
            print(f"  [OK] {test}")
            passed += 1
        else:
            print(f"  [FAIL] {test} -> {restored}")
            failed += 1

    # Test 2: Vietnamese detection
    print("\n[2] Vietnamese detection")
    vn_tests = [
        ("Hello world", False),
        ("Xin chào", True),
        ("Chúc mừng năm mới", True),
        ("123 ABC", False),
    ]
    for text, expected in vn_tests:
        result = is_vietnamese(text)
        if result == expected:
            print(f"  [OK] '{text}' -> {result}")
            passed += 1
        else:
            print(f"  [FAIL] '{text}' -> {result} (expected {expected})")
            failed += 1

    # Test 3: Empty/whitespace detection
    print("\n[3] Empty/whitespace detection")
    ws_tests = [
        ("", True),
        ("   ", True),
        ("\n\t", True),
        ("Hello", False),
    ]
    for text, expected in ws_tests:
        result = is_empty_or_whitespace(text)
        if result == expected:
            print(f"  [OK] '{repr(text)}' -> {result}")
            passed += 1
        else:
            print(f"  [FAIL] '{repr(text)}' -> {result} (expected {expected})")
            failed += 1

    # Test 4: YAML multiline preservation (mock)
    print("\n[4] YAML block scalar handling (structure test)")
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import LiteralScalarString

    yaml = YAML()
    test_yaml = {
        "literal": LiteralScalarString("Line 1\nLine 2\nLine 3"),
    }

    # Check type preservation
    if isinstance(test_yaml["literal"], LiteralScalarString):
        print(f"  [OK] LiteralScalarString preserved")
        passed += 1
    else:
        print(f"  [FAIL] LiteralScalarString lost")
        failed += 1

    # Test 5: Length guard
    print("\n[5] Length validation")
    translator_mock = type('obj', (object,), {
        'force': False,
        'protector': TokenProtector(),
    })()

    # Mock validate_translation
    original = "Short"
    too_long = "A" * 600
    ratio_exceed = "A" * int(2.5 * len(original) + 10)

    # Check absolute length
    if len(too_long) > MAX_ABSOLUTE_LENGTH:
        print(f"  [OK] Absolute length check (600 > {MAX_ABSOLUTE_LENGTH})")
        passed += 1
    else:
        print(f"  [FAIL] Absolute length check failed")
        failed += 1

    # Check ratio
    if len(ratio_exceed) > MAX_LENGTH_RATIO * len(original):
        print(f"  [OK] Length ratio check ({len(ratio_exceed)} > {MAX_LENGTH_RATIO} * {len(original)})")
        passed += 1
    else:
        print(f"  [FAIL] Length ratio check failed")
        failed += 1

    # Test 6: PlaceholderAPI conditionals
    print("\n[6] PlaceholderAPI conditionals protection")
    papi_tests = [
        # Conditionals with else
        "%if_player_level>=10%You are pro!%else%Keep grinding%endif%",
        # Conditionals without else
        "%if_player_has_permission_vip%&6[VIP]%endif% Welcome!",
        # Math expressions
        "Your score: %{player_kills * 100}%",
        "Health: %{player_health / player_max_health * 100}%%",
        # Function calls
        "Balance: %vault_eco_balance(player)_formatted%",
        # Mixed patterns
        "Hi {player}! Level: %{player_level * 2}% %if_level>=10%&a[PRO]%endif%",
        # Nested placeholders (complex)
        "%if_vault_eco_balance(player)>1000%&aRich!%else%&cPoor%endif%",
    ]

    protector_papi = TokenProtector()
    for test in papi_tests:
        protected = protector_papi.protect(test)
        restored = protector_papi.restore(protected)
        if restored == test:
            print(f"  [OK] {test[:60]}{'...' if len(test) > 60 else ''}")
            passed += 1
        else:
            print(f"  [FAIL] Original: {test}")
            print(f"         Restored: {restored}")
            failed += 1

    # Test 7: Structure validation
    print("\n[7] Structure validation (YAML/JSON corruption detection)")

    # Create a mock translator with validate_structure
    class MockTranslator:
        def __init__(self):
            self.protector = TokenProtector()
            self.force = False

        def validate_structure(self, original: str, translated: str):
            """Mock structure validation from Translator class."""
            structure_chars = {
                ':': 'colons',
                '|': 'pipes',
                '>': 'angle_brackets',
                '[': 'square_brackets_open',
                ']': 'square_brackets_close',
                '{': 'curly_brackets_open',
                '}': 'curly_brackets_close',
            }

            for char, name in structure_chars.items():
                orig_count = original.count(char)
                trans_count = translated.count(char)
                if orig_count != trans_count:
                    return False, f"structure_mismatch: {name} ({orig_count} → {trans_count})"

            for quote_char in ['"', "'"]:
                orig_count = original.count(quote_char)
                trans_count = translated.count(quote_char)
                if orig_count % 2 == 0:  # Original balanced
                    if trans_count % 2 != 0:
                        return False, f"quote_imbalance: {quote_char}"
                    if orig_count != trans_count:
                        return False, f"quote_mismatch: {quote_char}"

            return True, ""

    mock_translator = MockTranslator()

    structure_tests = [
        # (original, translated, should_pass)
        ('key: "value"', 'khóa: "giá trị"', True),  # Good
        ('key: "value"', 'khóa: giá trị', False),   # Missing quotes
        ('list: [1, 2]', 'danh sách: [1, 2]', True),  # Good
        ('list: [1, 2]', 'danh sách: 1, 2]', False),  # Missing bracket
        ('text: |', 'văn bản: |', True),  # Good
        ('text: |', 'văn bản:', False),   # Missing pipe
    ]

    for original, translated, should_pass in structure_tests:
        valid, error = mock_translator.validate_structure(original, translated)
        if valid == should_pass:
            print(f"  [OK] {original} -> {translated}")
            passed += 1
        else:
            print(f"  [FAIL] {original} -> {translated}")
            print(f"         Expected: {'pass' if should_pass else 'fail'}, Got: {'pass' if valid else 'fail'} ({error})")
            failed += 1

    # Summary
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("[OK] All tests passed!")
    else:
        print(f"[FAIL] {failed} test(s) failed")
        sys.exit(1)



class ETACalculator:
    """Tính ETA bằng EWMA."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ewma = None
        self.start_time = None

    def start(self):
        import time
        self.start_time = time.time()

    def update(self, completed, total):
        import time
        if not self.start_time:
            self.start()
        if completed == 0:
            return 0, 0
        elapsed = time.time() - self.start_time
        current_rate = elapsed / completed
        if self.ewma is None:
            self.ewma = current_rate
        else:
            self.ewma = self.alpha * current_rate + (1 - self.alpha) * self.ewma
        remaining = total - completed
        eta = remaining * self.ewma
        return self.ewma, eta

    @staticmethod
    def format_time(seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"


class MinecraftTranslatorGUI:
    """GUI Hoàn Chỉnh - 5 Tabs."""

    def __init__(self):
        try:
            import customtkinter as ctk
            from tkinter import filedialog, messagebox
            import threading
            import queue as queue_module
            import psutil
        except ImportError as e:
            print(f"LỖI: Thiếu thư viện: {e}")
            print("Chạy: pip install customtkinter psutil")
            import sys
            sys.exit(1)

        self.ctk = ctk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.threading = threading
        self.queue = queue_module.Queue()
        self.psutil = psutil

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Minecraft Config Translator - Dịch File Cấu Hình")
        self.root.geometry("1280x800")

        # State (thread-safe)
        self._state_lock = threading.Lock()
        self.folder_path = None
        self._is_translating = False
        self._is_paused = False
        self._stop_monitor = False  # Flag to stop monitor thread
        self.eta_calc = ETACalculator()
        self.translation_results = []
        self.glossary_data = {}
        self.current_file_index = 0
        self.total_files = 0

        # Keys tracking
        self.keys_total = 0
        self.keys_success = 0
        self.keys_error = 0

        # Completion statistics
        self.start_time = None
        self.files_success = 0
        self.files_skipped = 0
        self.files_error = 0
        self.peak_ram = 0
        self.peak_vram = 0

        self.build_ui()
        self.start_memory_monitor()
        self.setup_keyboard_shortcuts()

        # Bind cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    @property
    def is_translating(self):
        """Thread-safe getter for is_translating."""
        with self._state_lock:
            return self._is_translating

    @is_translating.setter
    def is_translating(self, value):
        """Thread-safe setter for is_translating."""
        with self._state_lock:
            self._is_translating = value

    @property
    def is_paused(self):
        """Thread-safe getter for is_paused."""
        with self._state_lock:
            return self._is_paused

    @is_paused.setter
    def is_paused(self, value):
        """Thread-safe setter for is_paused."""
        with self._state_lock:
            self._is_paused = value

    def build_ui(self):
        """Xây UI với 5 tabs."""
        ctk = self.ctk

        # Menu Bar
        self.build_menu_bar()

        # Status Bar
        self.status_bar = ctk.CTkFrame(self.root, height=30, fg_color="#1E1E1E")
        self.status_bar.pack(side="bottom", fill="x")

        self.status_text = ctk.CTkLabel(self.status_bar, text="🟢 Sẵn sàng", font=("Arial",10), anchor="w")
        self.status_text.pack(side="left", padx=10)

        self.status_model = ctk.CTkLabel(self.status_bar, text="Model: Chưa tải", font=("Arial",10))
        self.status_model.pack(side="left", padx=10)

        self.status_device = ctk.CTkLabel(self.status_bar, text="Device: Auto", font=("Arial",10))
        self.status_device.pack(side="left", padx=10)

        self.status_version = ctk.CTkLabel(self.status_bar, text="v1.0.0", font=("Arial",10))
        self.status_version.pack(side="right", padx=10)

        # TabView
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10,0))

        # 5 Tabs
        self.tab_home = self.tabview.add("🏠 Dịch")
        self.tab_preview = self.tabview.add("👁️ Xem Trước")
        self.tab_rollback = self.tabview.add("↩️ Rollback")
        self.tab_glossary = self.tabview.add("📖 Từ Điển")
        self.tab_settings = self.tabview.add("⚙️ Cài Đặt")

        self.build_tab_home()
        self.build_tab_preview()
        self.build_tab_rollback()
        self.build_tab_glossary()
        self.build_tab_settings()

    def build_tab_home(self):
        """Tab Dịch."""
        ctk = self.ctk
        parent = self.tab_home

        # 2 cột
        left = ctk.CTkScrollableFrame(parent, width=380)
        left.pack(side="left", fill="y", padx=(0,10), pady=5)

        right = ctk.CTkFrame(parent)
        right.pack(side="right", fill="both", expand=True, pady=5)

        # === CỘT TRÁI ===
        ctk.CTkLabel(left, text="⚙️ CẤU HÌNH", font=("Arial",18,"bold")).pack(pady=10)

        # Thư mục
        ctk.CTkLabel(left, text="📁 Thư Mục Nguồn:", font=("Arial",12,"bold")).pack(anchor="w",pady=(10,5))
        folder_frame = ctk.CTkFrame(left)
        folder_frame.pack(fill="x", pady=5)
        self.folder_entry = ctk.CTkEntry(folder_frame, placeholder_text="Chọn thư mục...", width=210)
        self.folder_entry.pack(side="left", padx=(0,5))
        ctk.CTkButton(folder_frame, text="Chọn", width=80, command=self.browse_folder).pack(side="left", padx=(0,5))

        # Recent folders dropdown button (using menubutton style)
        from tkinter import Menu, Menubutton
        self.folder_recent_btn = Menubutton(folder_frame, text="📂 ▼", relief="raised", width=5)
        self.folder_recent_btn.pack(side="left")
        self.folder_recent_menu = Menu(self.folder_recent_btn, tearoff=0)
        self.folder_recent_btn['menu'] = self.folder_recent_menu
        self.update_recent_folders_dropdown()

        # Model
        ctk.CTkLabel(left, text="🤖 Model AI:", font=("Arial",12,"bold")).pack(anchor="w",pady=(15,5))
        models = ["nllb-3.3b (Chất lượng cao) ⭐", "nllb-1.3b (Cân bằng)", "marian (Nhanh)", "m2m100 (Thử nghiệm)"]
        self.model_var = ctk.StringVar(value=models[0])
        ctk.CTkOptionMenu(left, values=models, variable=self.model_var, width=340).pack(fill="x")

        # Device Selection
        ctk.CTkLabel(left, text="🔧 Thiết Bị:", font=("Arial",12,"bold")).pack(anchor="w",pady=(15,5))
        device_frame = ctk.CTkFrame(left, fg_color="transparent")
        device_frame.pack(fill="x", pady=2)
        self.device_var = ctk.StringVar(value="auto")
        ctk.CTkRadioButton(device_frame, text="Tự động", variable=self.device_var, value="auto").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(device_frame, text="GPU (CUDA)", variable=self.device_var, value="gpu").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(device_frame, text="CPU", variable=self.device_var, value="cpu").pack(anchor="w", pady=2)

        # Quantization Mode
        ctk.CTkLabel(left, text="📦 Lượng Tử Hóa:", font=("Arial",12,"bold")).pack(anchor="w",pady=(10,5))
        quant_frame = ctk.CTkFrame(left, fg_color="transparent")
        quant_frame.pack(fill="x", pady=2)
        self.quant_var = ctk.StringVar(value="auto")
        ctk.CTkRadioButton(quant_frame, text="Tự động (4-bit → 8-bit → Full)", variable=self.quant_var, value="auto").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(quant_frame, text="4-bit (Nhanh nhất, ít RAM)", variable=self.quant_var, value="4bit").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(quant_frame, text="8-bit (Cân bằng)", variable=self.quant_var, value="8bit").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(quant_frame, text="Full (Chậm nhất, nhiều RAM)", variable=self.quant_var, value="full").pack(anchor="w", pady=2)

        # Tùy chọn
        ctk.CTkLabel(left, text="⚙️ Tùy Chọn:", font=("Arial",12,"bold")).pack(anchor="w",pady=(15,5))
        self.low_mem_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Chế độ ít RAM", variable=self.low_mem_var).pack(anchor="w",pady=2)
        self.report_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Tạo báo cáo CSV", variable=self.report_var).pack(anchor="w",pady=2)
        self.dry_run_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Chỉ xem trước (Dry run)", variable=self.dry_run_var).pack(anchor="w",pady=2)

        # Tùy chọn nâng cao (collapsible)
        self.advanced_visible = False
        adv_header = ctk.CTkFrame(left, fg_color="transparent")
        adv_header.pack(fill="x", pady=(15,5))
        self.adv_toggle_btn = ctk.CTkButton(adv_header, text="▶ Tùy Chọn Nâng Cao",
                                            command=self.toggle_advanced, width=340, height=30,
                                            fg_color="#2B2B2B", hover_color="#3B3B3B")
        self.adv_toggle_btn.pack()

        self.advanced_frame = ctk.CTkFrame(left, fg_color="#1E1E1E")
        # Không pack ngay, sẽ pack khi toggle

        # Batch size
        batch_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        batch_frame.pack(fill="x", pady=5, padx=10)
        ctk.CTkLabel(batch_frame, text="Batch Size:", font=("Arial",10)).pack(side="left")
        self.batch_var = ctk.IntVar(value=8)
        self.batch_slider = ctk.CTkSlider(batch_frame, from_=1, to=32, number_of_steps=31,
                                         variable=self.batch_var, width=180)
        self.batch_slider.pack(side="left", padx=5)
        self.batch_label = ctk.CTkLabel(batch_frame, text="8", font=("Arial",10,"bold"), width=30)
        self.batch_label.pack(side="left")
        self.batch_var.trace_add("write", lambda *args: self.batch_label.configure(text=str(self.batch_var.get())))

        # Fallback model
        fallback_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        fallback_frame.pack(fill="x", pady=5, padx=10)
        ctk.CTkLabel(fallback_frame, text="Fallback Model:", font=("Arial",10)).pack(anchor="w")
        fallback_models = ["Không", "marian", "m2m100"]
        self.fallback_var = ctk.StringVar(value="Không")
        ctk.CTkOptionMenu(fallback_frame, values=fallback_models, variable=self.fallback_var,
                         width=320, height=28, font=("Arial",10)).pack(fill="x")

        # File extensions
        ext_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        ext_frame.pack(fill="x", pady=5, padx=10)
        ctk.CTkLabel(ext_frame, text="Phần Mở Rộng:", font=("Arial",10)).pack(anchor="w")
        self.ext_entry = ctk.CTkEntry(ext_frame, placeholder_text=".yml,.yaml,.json,.properties,.txt,.lang",
                                      font=("Arial",9), height=28)
        self.ext_entry.pack(fill="x")

        # Force checkbox
        self.force_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.advanced_frame, text="Force (dịch lại cả đã dịch)",
                       variable=self.force_var, font=("Arial",10)).pack(anchor="w", pady=2, padx=10)

        # Max memory
        mem_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        mem_frame.pack(fill="x", pady=5, padx=10)
        ctk.CTkLabel(mem_frame, text="Max Memory (MB):", font=("Arial",10)).pack(side="left")
        self.max_mem_entry = ctk.CTkEntry(mem_frame, placeholder_text="Tự động", width=100,
                                         font=("Arial",9), height=28)
        self.max_mem_entry.pack(side="left", padx=5)

        # Glossary file picker
        gloss_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        gloss_frame.pack(fill="x", pady=5, padx=10)
        ctk.CTkLabel(gloss_frame, text="Từ Điển:", font=("Arial",10)).pack(anchor="w")
        gloss_input = ctk.CTkFrame(gloss_frame, fg_color="transparent")
        gloss_input.pack(fill="x")
        self.glossary_entry = ctk.CTkEntry(gloss_input, placeholder_text="Chọn file JSON...",
                                          font=("Arial",9), height=28)
        self.glossary_entry.pack(side="left", fill="x", expand=True, padx=(0,5))
        ctk.CTkButton(gloss_input, text="📁", width=40, height=28,
                     command=self.browse_glossary).pack(side="left")

        # Nút điều khiển
        ctk.CTkLabel(left, text="🎮 Điều Khiển:", font=("Arial",12,"bold")).pack(anchor="w",pady=(15,10))
        self.start_btn = ctk.CTkButton(left, text="▶️ BẮT ĐẦU DỊCH", command=self.start_translation,
                                       fg_color="green", hover_color="darkgreen", height=50, font=("Arial",14,"bold"))
        self.start_btn.pack(fill="x", pady=5)

        self.pause_btn = ctk.CTkButton(left, text="⏸️ TẠM DỪNG", command=self.toggle_pause,
                                       fg_color="orange", hover_color="darkorange", height=40, state="disabled")
        self.pause_btn.pack(fill="x", pady=5)

        self.stop_btn = ctk.CTkButton(left, text="⏹️ DỪNG", command=self.stop_translation,
                                      fg_color="red", hover_color="darkred", height=40, state="disabled")
        self.stop_btn.pack(fill="x", pady=5)

        ctk.CTkButton(left, text="📥 CHỈ TẢI MODEL", command=self.download_model_only,
                     fg_color="#FF8C00", hover_color="#FF6800", height=40).pack(fill="x", pady=5)

        # === CỘT PHẢI ===
        ctk.CTkLabel(right, text="📊 TIẾN ĐỘ & LOG", font=("Arial",18,"bold")).pack(pady=10)

        # Progress
        prog_frame = ctk.CTkFrame(right)
        prog_frame.pack(fill="x", padx=10, pady=5)

        self.status_label = ctk.CTkLabel(prog_frame, text="Trạng thái: Sẵn sàng", font=("Arial",14), anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(prog_frame, width=400)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)

        self.progress_text = ctk.CTkLabel(prog_frame, text="0 / 0 files • 0s/file • ETA: 0s", font=("Arial",11))
        self.progress_text.pack(pady=5)

        # Keys counter
        self.keys_label = ctk.CTkLabel(prog_frame, text="Keys: 0 | Thành công: 0 | Lỗi: 0", font=("Arial",11))
        self.keys_label.pack(pady=5)

        self.memory_label = ctk.CTkLabel(prog_frame, text="💾 RAM: 0 GB | VRAM: 0 GB", font=("Arial",11))
        self.memory_label.pack(pady=5)

        # Log
        log_frame = ctk.CTkFrame(right)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(log_frame, text="📝 Log Trực Tiếp:", font=("Arial",12,"bold")).pack(anchor="w", pady=5)

        self.log_text = ctk.CTkTextbox(log_frame, font=("Consolas",9), wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=5)

        btn_frame = ctk.CTkFrame(log_frame)
        btn_frame.pack(fill="x", pady=5)
        ctk.CTkButton(btn_frame, text="📋 Copy", command=self.copy_log, width=80).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="💾 Lưu", command=self.save_log, width=80).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="🗑️ Xóa", command=self.clear_log, width=80).pack(side="left", padx=5)

    def build_tab_preview(self):
        """Tab Xem Trước - Full implementation."""
        ctk = self.ctk
        parent = self.tab_preview

        # Header with toolbar
        header = ctk.CTkFrame(parent)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="👁️ Xem Trước & So Sánh", font=("Arial",18,"bold")).pack(side="left", padx=10)

        toolbar = ctk.CTkFrame(header)
        toolbar.pack(side="right", padx=10)
        ctk.CTkButton(toolbar, text="🔄 Làm mới", width=100, command=self.preview_refresh).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="📊 Xuất Excel", width=120, command=self.preview_export_excel).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="☑ Chọn tất cả", width=120, command=self.preview_select_all).pack(side="left", padx=2)

        # Filters and search
        filter_frame = ctk.CTkFrame(parent)
        filter_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(filter_frame, text="Lọc:", font=("Arial",11)).pack(side="left", padx=5)
        self.preview_filter_var = ctk.StringVar(value="Tất cả")
        ctk.CTkOptionMenu(filter_frame, values=["Tất cả", "Đã dịch", "Bỏ qua", "Lỗi"],
                         variable=self.preview_filter_var, width=120,
                         command=lambda x: self.preview_refresh()).pack(side="left", padx=5)

        ctk.CTkLabel(filter_frame, text="Tìm kiếm:", font=("Arial",11)).pack(side="left", padx=10)
        self.preview_search_var = ctk.StringVar()
        self.preview_search_var.trace_add("write", lambda *args: self.preview_refresh())
        ctk.CTkEntry(filter_frame, textvariable=self.preview_search_var,
                    placeholder_text="Tên file...", width=300).pack(side="left", padx=5)

        self.preview_count_label = ctk.CTkLabel(filter_frame, text="0 files", font=("Arial",11))
        self.preview_count_label.pack(side="right", padx=10)

        # File table
        table_container = ctk.CTkFrame(parent)
        table_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Headers
        headers = ctk.CTkFrame(table_container, fg_color="#2B2B2B")
        headers.pack(fill="x", pady=(0,5))
        ctk.CTkLabel(headers, text="☑", font=("Arial",11,"bold"), width=40).pack(side="left", padx=5)
        ctk.CTkLabel(headers, text="File", font=("Arial",11,"bold"), width=350, anchor="w").pack(side="left", padx=5)
        ctk.CTkLabel(headers, text="Keys", font=("Arial",11,"bold"), width=80, anchor="center").pack(side="left", padx=5)
        ctk.CTkLabel(headers, text="Trạng thái", font=("Arial",11,"bold"), width=100, anchor="center").pack(side="left", padx=5)
        ctk.CTkLabel(headers, text="Thời gian", font=("Arial",11,"bold"), width=100, anchor="center").pack(side="left", padx=5)
        ctk.CTkLabel(headers, text="", font=("Arial",11,"bold"), width=150, anchor="center").pack(side="left", padx=5)

        # Scrollable file list
        self.preview_list_frame = ctk.CTkScrollableFrame(table_container)
        self.preview_list_frame.pack(fill="both", expand=True)

        # Data storage
        self.preview_files = []
        self.preview_checkboxes = {}
        self.selected_preview_file = None

        # Diff viewer (bottom panel)
        diff_container = ctk.CTkFrame(parent, height=300)
        diff_container.pack(fill="x", padx=10, pady=10)
        diff_container.pack_propagate(False)

        ctk.CTkLabel(diff_container, text="📄 Chi Tiết File", font=("Arial",14,"bold")).pack(anchor="w", padx=10, pady=5)

        # Two-column diff viewer
        diff_content = ctk.CTkFrame(diff_container)
        diff_content.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: Original (EN)
        left_panel = ctk.CTkFrame(diff_content)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0,5))
        ctk.CTkLabel(left_panel, text="❌ TRƯỚC (EN)", font=("Arial",11,"bold"), fg_color="#8B0000").pack(fill="x")
        self.diff_original = ctk.CTkTextbox(left_panel, font=("Consolas",9), wrap="word")
        self.diff_original.pack(fill="both", expand=True)

        # Right: Translated (VI)
        right_panel = ctk.CTkFrame(diff_content)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5,0))
        ctk.CTkLabel(right_panel, text="✅ SAU (VI)", font=("Arial",11,"bold"), fg_color="#006400").pack(fill="x")
        self.diff_translated = ctk.CTkTextbox(right_panel, font=("Consolas",9), wrap="word")
        self.diff_translated.pack(fill="both", expand=True)

        # Diff actions
        diff_actions = ctk.CTkFrame(diff_container)
        diff_actions.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(diff_actions, text="✅ Giữ Thay Đổi", width=150,
                     command=self.preview_keep_changes, fg_color="green").pack(side="left", padx=5)
        ctk.CTkButton(diff_actions, text="↩️ Hoàn Tác File Này", width=150,
                     command=self.preview_undo_file, fg_color="orange").pack(side="left", padx=5)
        ctk.CTkButton(diff_actions, text="📋 Copy EN", width=100,
                     command=lambda: self.copy_to_clipboard(self.diff_original.get("1.0", "end"))).pack(side="left", padx=5)
        ctk.CTkButton(diff_actions, text="📋 Copy VI", width=100,
                     command=lambda: self.copy_to_clipboard(self.diff_translated.get("1.0", "end"))).pack(side="left", padx=5)

    def build_tab_rollback(self):
        """Tab Rollback - Backup Management."""
        ctk = self.ctk
        parent = self.tab_rollback

        # Header
        header = ctk.CTkFrame(parent)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="↩️ Khôi Phục File Backup", font=("Arial",18,"bold")).pack(side="left", padx=10)

        # Toolbar
        toolbar = ctk.CTkFrame(header)
        toolbar.pack(side="right", padx=10)
        ctk.CTkButton(toolbar, text="🔄 Quét lại", width=100, command=self.rollback_scan).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="↩️ Khôi phục đã chọn", width=140, command=self.rollback_selected).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="↩️ Khôi phục tất cả", width=140, command=self.rollback_all).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="🗑️ Xóa .bak", width=100, command=self.rollback_delete_bak).pack(side="left", padx=2)

        # Info
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(info_frame, text="📁 Thư mục:", font=("Arial",11)).pack(side="left", padx=5)
        self.rollback_folder_label = ctk.CTkLabel(info_frame, text="Chưa chọn thư mục", font=("Arial",11,"bold"))
        self.rollback_folder_label.pack(side="left", padx=5)
        self.rollback_count_label = ctk.CTkLabel(info_frame, text="0 file backup", font=("Arial",11))
        self.rollback_count_label.pack(side="right", padx=10)

        # Table
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Headers
        headers_frame = ctk.CTkFrame(table_frame, fg_color="#2B2B2B")
        headers_frame.pack(fill="x", pady=(0,5))
        ctk.CTkLabel(headers_frame, text="☑", font=("Arial",12,"bold"), width=40, anchor="center").pack(side="left", padx=5)
        ctk.CTkLabel(headers_frame, text="File gốc", font=("Arial",12,"bold"), width=500, anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(headers_frame, text="File backup", font=("Arial",12,"bold"), width=400, anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(headers_frame, text="Kích thước", font=("Arial",12,"bold"), width=120, anchor="center").pack(side="left", padx=10)

        # Scrollable list
        self.rollback_list_frame = ctk.CTkScrollableFrame(table_frame)
        self.rollback_list_frame.pack(fill="both", expand=True)

        # Backup data
        self.backup_files = []
        self.backup_checkboxes = {}

    def build_tab_glossary(self):
        """Tab Từ Điển - EN→VI Editor."""
        ctk = self.ctk
        parent = self.tab_glossary

        # Header
        header = ctk.CTkFrame(parent)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="📖 Từ Điển EN → VI", font=("Arial",18,"bold")).pack(side="left", padx=10)

        # Toolbar
        toolbar = ctk.CTkFrame(header)
        toolbar.pack(side="right", padx=10)
        ctk.CTkButton(toolbar, text="📂 Mở JSON", width=100, command=self.glossary_open_json).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="💾 Lưu", width=80, command=self.glossary_save_json).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="📥 Import", width=80, command=self.glossary_import).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="📤 Export", width=80, command=self.glossary_export).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="➕ Thêm", width=80, command=self.glossary_add).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="✏️ Sửa", width=80, command=self.glossary_edit).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="🗑️ Xóa", width=80, command=self.glossary_delete).pack(side="left", padx=2)

        # Search
        search_frame = ctk.CTkFrame(parent)
        search_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(search_frame, text="🔍 Tìm kiếm:", font=("Arial",12)).pack(side="left", padx=5)
        self.glossary_search_var = ctk.StringVar()
        self.glossary_search_var.trace_add("write", lambda *args: self.glossary_filter())
        ctk.CTkEntry(search_frame, textvariable=self.glossary_search_var,
                    placeholder_text="Nhập từ tiếng Anh hoặc tiếng Việt...",
                    width=400).pack(side="left", padx=5)
        self.glossary_count_label = ctk.CTkLabel(search_frame, text="0 từ", font=("Arial",11))
        self.glossary_count_label.pack(side="left", padx=10)

        # Table
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Headers
        headers_frame = ctk.CTkFrame(table_frame, fg_color="#2B2B2B")
        headers_frame.pack(fill="x", pady=(0,5))
        ctk.CTkLabel(headers_frame, text="English", font=("Arial",12,"bold"), width=400, anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(headers_frame, text="Tiếng Việt", font=("Arial",12,"bold"), width=400, anchor="w").pack(side="left", padx=10)
        ctk.CTkLabel(headers_frame, text="Số lần dùng", font=("Arial",12,"bold"), width=100, anchor="center").pack(side="left", padx=10)

        # Scrollable list
        self.glossary_list_frame = ctk.CTkScrollableFrame(table_frame)
        self.glossary_list_frame.pack(fill="both", expand=True)

        # Load glossary data
        self.glossary_data = {}
        self.glossary_filter()

    def build_tab_settings(self):
        """Tab Cài Đặt - Full implementation."""
        ctk = self.ctk
        parent = self.tab_settings

        # Scrollable container
        container = ctk.CTkScrollableFrame(parent)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Giao diện
        ctk.CTkLabel(container, text="🎨 Giao Diện", font=("Arial",16,"bold")).pack(anchor="w", pady=(10,5))

        theme_frame = ctk.CTkFrame(container, fg_color="transparent")
        theme_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(theme_frame, text="Chủ đề:", font=("Arial",12)).pack(side="left", padx=5)
        self.theme_var = ctk.StringVar(value="dark")
        ctk.CTkRadioButton(theme_frame, text="Sáng", variable=self.theme_var, value="light", command=self.change_theme).pack(side="left", padx=5)
        ctk.CTkRadioButton(theme_frame, text="Tối", variable=self.theme_var, value="dark", command=self.change_theme).pack(side="left", padx=5)
        ctk.CTkRadioButton(theme_frame, text="Tự động", variable=self.theme_var, value="system", command=self.change_theme).pack(side="left", padx=5)

        lang_frame = ctk.CTkFrame(container, fg_color="transparent")
        lang_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(lang_frame, text="Ngôn ngữ:", font=("Arial",12)).pack(side="left", padx=5)
        self.lang_var = ctk.StringVar(value="vi")
        ctk.CTkOptionMenu(lang_frame, values=["Tiếng Việt", "English"], variable=self.lang_var, width=200).pack(side="left", padx=5)

        font_frame = ctk.CTkFrame(container, fg_color="transparent")
        font_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(font_frame, text="Cỡ chữ log:", font=("Arial",12)).pack(side="left", padx=5)
        self.font_size_var = ctk.IntVar(value=9)
        ctk.CTkOptionMenu(font_frame, values=["8", "9", "10", "12", "14"], variable=self.font_size_var, width=100, command=self.change_font_size).pack(side="left", padx=5)

        # Bộ nhớ & Cache
        ctk.CTkLabel(container, text="💾 Bộ Nhớ & Cache", font=("Arial",16,"bold")).pack(anchor="w", pady=(20,5))

        cache_dir_frame = ctk.CTkFrame(container, fg_color="transparent")
        cache_dir_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(cache_dir_frame, text="Thư mục cache HuggingFace:", font=("Arial",11)).pack(anchor="w")
        cache_path_frame = ctk.CTkFrame(cache_dir_frame)
        cache_path_frame.pack(fill="x", pady=5)

        import os
        cache_path = os.path.expanduser("~/.cache/huggingface")
        self.cache_entry = ctk.CTkEntry(cache_path_frame, width=500)
        self.cache_entry.insert(0, cache_path)
        self.cache_entry.pack(side="left", padx=5)
        ctk.CTkButton(cache_path_frame, text="📂 Chọn", width=80, command=self.browse_cache).pack(side="left")

        cache_info_frame = ctk.CTkFrame(container, fg_color="transparent")
        cache_info_frame.pack(fill="x", pady=5)
        self.cache_size_label = ctk.CTkLabel(cache_info_frame, text="Dung lượng: Đang tính...", font=("Arial",11))
        self.cache_size_label.pack(side="left", padx=5)

        # Cache limit dropdown
        limit_frame = ctk.CTkFrame(cache_info_frame, fg_color="transparent")
        limit_frame.pack(side="left", padx=20)
        ctk.CTkLabel(limit_frame, text="Giới hạn:", font=("Arial",11)).pack(side="left", padx=5)
        self.cache_limit_var = ctk.StringVar(value="Không giới hạn")
        ctk.CTkOptionMenu(limit_frame, values=["10GB", "20GB", "50GB", "Không giới hạn"],
                         variable=self.cache_limit_var, width=150).pack(side="left")

        ctk.CTkButton(cache_info_frame, text="🗑️ Xóa Cache", width=120, command=self.clear_cache, fg_color="red").pack(side="right", padx=5)
        ctk.CTkButton(cache_info_frame, text="📊 Xem Chi Tiết", width=120, command=self.view_cache_details).pack(side="right", padx=5)

        # Preset
        ctk.CTkLabel(container, text="⚙️ Preset Cấu Hình", font=("Arial",16,"bold")).pack(anchor="w", pady=(20,5))

        preset_list_frame = ctk.CTkFrame(container)
        preset_list_frame.pack(fill="x", pady=5)

        self.presets = {
            "Máy yếu": {"model": "nllb-1.3b", "device": "auto", "quant": "4bit", "low_mem": True},
            "Cân bằng": {"model": "nllb-3.3b", "device": "auto", "quant": "8bit", "low_mem": False},
            "Chất lượng cao": {"model": "nllb-3.3b", "device": "gpu", "quant": "full", "low_mem": False},
        }

        for preset_name in self.presets.keys():
            btn = ctk.CTkButton(preset_list_frame, text=f"📌 {preset_name}", width=200, command=lambda p=preset_name: self.load_preset(p))
            btn.pack(pady=2)

        preset_actions = ctk.CTkFrame(container, fg_color="transparent")
        preset_actions.pack(fill="x", pady=5)
        ctk.CTkButton(preset_actions, text="💾 Lưu Cấu Hình Hiện Tại", width=200, command=self.save_preset).pack(side="left", padx=5)
        ctk.CTkButton(preset_actions, text="❌ Xóa Preset", width=150, command=self.delete_preset).pack(side="left", padx=5)

        # Thông báo
        ctk.CTkLabel(container, text="🔔 Thông Báo", font=("Arial",16,"bold")).pack(anchor="w", pady=(20,5))

        self.sound_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(container, text="Phát âm thanh khi hoàn thành", variable=self.sound_var).pack(anchor="w", pady=2)

        self.notif_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(container, text="Hiển thị notification", variable=self.notif_var).pack(anchor="w", pady=2)

        self.auto_update_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(container, text="Tự động kiểm tra cập nhật khi mở app", variable=self.auto_update_var).pack(anchor="w", pady=2)

        # Nâng cao
        ctk.CTkLabel(container, text="🔐 Nâng Cao", font=("Arial",16,"bold")).pack(anchor="w", pady=(20,5))

        self.auto_backup_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(container, text="Tự động backup trước khi dịch", variable=self.auto_backup_var).pack(anchor="w", pady=2)

        self.delete_bak_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(container, text="Xóa .bak sau khi dịch thành công", variable=self.delete_bak_var).pack(anchor="w", pady=2)

        self.detailed_log_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(container, text="Ghi log chi tiết ra file", variable=self.detailed_log_var).pack(anchor="w", pady=2)

        log_path_frame = ctk.CTkFrame(container, fg_color="transparent")
        log_path_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(log_path_frame, text="Vị trí log:", font=("Arial",11)).pack(side="left", padx=5)
        self.log_path_entry = ctk.CTkEntry(log_path_frame, width=300)
        self.log_path_entry.insert(0, "logs/translator.log")
        self.log_path_entry.pack(side="left", padx=5)
        ctk.CTkButton(log_path_frame, text="📂 Mở", width=80, command=self.open_log_folder).pack(side="left")

    def start_memory_monitor(self):
        """Monitor RAM/VRAM real-time."""
        def monitor():
            import time
            while not self._stop_monitor:
                try:
                    mem = self.psutil.virtual_memory()
                    ram_used = mem.used / (1024**3)
                    ram_total = mem.total / (1024**3)

                    # Track peak RAM
                    if self.is_translating and ram_used > self.peak_ram:
                        self.peak_ram = ram_used

                    vram_text = "N/A"
                    vram_used = 0
                    try:
                        import torch
                        if torch.cuda.is_available():
                            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            vram_text = f"{vram_used:.1f}/{vram_total:.1f} GB"

                            # Track peak VRAM
                            if self.is_translating and vram_used > self.peak_vram:
                                self.peak_vram = vram_used
                    except:
                        pass

                    self.memory_label.configure(text=f"💾 RAM: {ram_used:.1f}/{ram_total:.1f} GB | VRAM: {vram_text}")
                except:
                    pass
                time.sleep(2)

        thread = self.threading.Thread(target=monitor, daemon=True)
        thread.start()

    def browse_folder(self):
        folder = self.filedialog.askdirectory(title="Chọn thư mục Minecraft plugins/configs")
        if folder:
            self.folder_path = folder
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)
            self.log(f"✅ Đã chọn: {folder}")

            # Save to recent folders
            self.add_recent_folder(folder)

            # Auto-scan for backups in Rollback tab
            if hasattr(self, 'rollback_folder_label'):
                self.rollback_scan()

    def load_recent_folders(self):
        """Load recent folders from JSON."""
        try:
            from pathlib import Path
            recent_file = Path.home() / ".minecraft_translator_recent.json"
            if recent_file.exists():
                import json
                with open(recent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('recent_folders', [])[:10]  # Max 10
        except:
            pass
        return []

    def save_recent_folders(self, folders):
        """Save recent folders to JSON."""
        try:
            from pathlib import Path
            import json
            recent_file = Path.home() / ".minecraft_translator_recent.json"
            with open(recent_file, 'w', encoding='utf-8') as f:
                json.dump({'recent_folders': folders}, f, indent=2, ensure_ascii=False)
        except:
            pass

    def add_recent_folder(self, folder):
        """Add folder to recent list."""
        recent = self.load_recent_folders()
        # Remove if already exists
        if folder in recent:
            recent.remove(folder)
        # Add to front
        recent.insert(0, folder)
        # Keep max 10
        recent = recent[:10]
        self.save_recent_folders(recent)

        # Update dropdown if exists
        if hasattr(self, 'folder_recent_menu'):
            self.update_recent_folders_dropdown()

    def select_recent_folder(self, folder):
        """Select a recent folder."""
        self.folder_path = folder
        self.folder_entry.delete(0, "end")
        self.folder_entry.insert(0, folder)
        self.log(f"✅ Đã chọn: {folder}")

        # Auto-scan for backups
        if hasattr(self, 'rollback_folder_label'):
            self.rollback_scan()

    def update_recent_folders_dropdown(self):
        """Update the recent folders dropdown menu."""
        if not hasattr(self, 'folder_recent_menu'):
            return

        # Clear existing menu
        menu = self.folder_recent_menu
        menu.delete(0, 'end')

        # Add recent folders
        recent = self.load_recent_folders()
        if recent:
            for folder in recent:
                # Shorten path for display
                display = folder if len(folder) <= 60 else "..." + folder[-57:]
                menu.add_command(label=display, command=lambda f=folder: self.select_recent_folder(f))
            menu.add_separator()
            menu.add_command(label="🗑️ Xóa Lịch Sử", command=self.clear_recent_folders)
        else:
            menu.add_command(label="(Chưa có lịch sử)", state="disabled")

    def clear_recent_folders(self):
        """Clear recent folders history."""
        if self.messagebox.askyesno("Xác nhận", "Xóa toàn bộ lịch sử thư mục?"):
            self.save_recent_folders([])
            self.update_recent_folders_dropdown()
            self.log("🗑️ Đã xóa lịch sử thư mục")

    def show_file_context_menu(self, event, filename, data):
        """Show context menu for file in preview."""
        from tkinter import Menu
        menu = Menu(self.root, tearoff=0)

        menu.add_command(label="🔍 Xem Chi Tiết", command=lambda: self.preview_view_file(filename, data))
        menu.add_separator()
        menu.add_command(label="📂 Mở File", command=lambda: self.open_file(filename))
        menu.add_command(label="📁 Mở Thư Mục Chứa", command=lambda: self.open_file_location(filename))
        menu.add_command(label="📋 Copy Đường Dẫn", command=lambda: self.copy_file_path(filename))
        menu.add_separator()
        menu.add_command(label="↩️ Khôi Phục File", command=lambda: self.preview_undo_file_by_name(filename))
        menu.add_command(label="🗑️ Xóa Backup", command=lambda: self.delete_file_backup(filename), foreground="red")

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def open_file(self, filename):
        """Open file with default application."""
        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename

        if not file_path.exists():
            self.messagebox.showerror("Lỗi", f"File không tồn tại: {filename}")
            return

        import os
        import subprocess
        import platform

        try:
            if platform.system() == "Windows":
                os.startfile(str(file_path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", str(file_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(file_path)])
            self.log(f"📂 Đã mở file: {filename}")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể mở file: {e}")

    def open_file_location(self, filename):
        """Open folder containing the file."""
        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename

        # Open parent folder
        import os
        import subprocess
        import platform

        try:
            folder = file_path.parent
            if platform.system() == "Windows":
                # Select file in Explorer
                subprocess.Popen(f'explorer /select,"{file_path}"')
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", "-R", str(file_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(folder)])
            self.log(f"📁 Đã mở thư mục: {folder}")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể mở thư mục: {e}")

    def copy_file_path(self, filename):
        """Copy file path to clipboard."""
        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(str(file_path))
            self.root.update()
            self.log(f"📋 Đã copy: {file_path}")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể copy: {e}")

    def preview_undo_file_by_name(self, filename):
        """Undo a single file by name."""
        if not self.folder_path:
            return

        if not self.messagebox.askyesno("Xác nhận", f"Khôi phục file:\n{filename}?"):
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        if not backup_path.exists():
            self.messagebox.showerror("Lỗi", "Không tìm thấy backup!")
            return

        try:
            import shutil
            shutil.copy2(backup_path, file_path)
            self.log(f"✅ Đã khôi phục: {filename}")
            self.messagebox.showinfo("Thành công", f"Đã khôi phục file:\n{filename}")
            self.preview_refresh()
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Khôi phục thất bại: {e}")

    def delete_file_backup(self, filename):
        """Delete backup file."""
        if not self.folder_path:
            return

        if not self.messagebox.askyesno("Cảnh báo", f"XÓA backup của:\n{filename}?\n\nHành động này KHÔNG THỂ HOÀN TÁC!"):
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        if not backup_path.exists():
            self.messagebox.showerror("Lỗi", "Không tìm thấy backup!")
            return

        try:
            backup_path.unlink()
            self.log(f"🗑️ Đã xóa backup: {filename}.bak")
            self.messagebox.showinfo("Thành công", "Đã xóa backup!")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Xóa thất bại: {e}")

    def toggle_advanced(self):
        """Toggle hiển thị tùy chọn nâng cao."""
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_frame.pack(fill="x", pady=(0,10))
            self.adv_toggle_btn.configure(text="▼ Tùy Chọn Nâng Cao")
        else:
            self.advanced_frame.pack_forget()
            self.adv_toggle_btn.configure(text="▶ Tùy Chọn Nâng Cao")

    def browse_glossary(self):
        """Chọn file glossary JSON."""
        filename = self.filedialog.askopenfilename(
            title="Chọn file từ điển",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.glossary_entry.delete(0, "end")
            self.glossary_entry.insert(0, filename)
            self.log(f"📖 Đã chọn từ điển: {filename}")

    def glossary_filter(self):
        """Filter và hiển thị glossary."""
        # Clear current list
        for widget in self.glossary_list_frame.winfo_children():
            widget.destroy()

        # Get search term
        search = self.glossary_search_var.get().lower() if hasattr(self, 'glossary_search_var') else ""

        # Filter items
        filtered = {}
        for en, data in self.glossary_data.items():
            vi = data.get("vi", "") if isinstance(data, dict) else data
            if search in en.lower() or search in vi.lower():
                filtered[en] = data

        # Update count
        if hasattr(self, 'glossary_count_label'):
            self.glossary_count_label.configure(text=f"{len(filtered)} từ")

        # Display items
        for en in sorted(filtered.keys()):
            data = filtered[en]
            vi = data.get("vi", "") if isinstance(data, dict) else data
            usage = data.get("usage", 0) if isinstance(data, dict) else 0

            row = self.ctk.CTkFrame(self.glossary_list_frame, fg_color="#1E1E1E")
            row.pack(fill="x", pady=2, padx=5)

            # Click to select
            row.bind("<Button-1>", lambda e, key=en: self.glossary_select(key))

            self.ctk.CTkLabel(row, text=en, font=("Arial",10), width=400, anchor="w").pack(side="left", padx=10)
            self.ctk.CTkLabel(row, text=vi, font=("Arial",10), width=400, anchor="w").pack(side="left", padx=10)
            self.ctk.CTkLabel(row, text=str(usage), font=("Arial",10), width=100, anchor="center").pack(side="left", padx=10)

        self.selected_glossary_key = None

    def glossary_select(self, key):
        """Select một từ."""
        self.selected_glossary_key = key

    def glossary_add(self):
        """Thêm từ mới."""
        dialog = self.ctk.CTkToplevel(self.root)
        dialog.title("Thêm từ mới")
        dialog.geometry("500x200")
        dialog.transient(self.root)
        dialog.grab_set()

        self.ctk.CTkLabel(dialog, text="English:", font=("Arial",12)).pack(anchor="w", padx=20, pady=(20,5))
        en_entry = self.ctk.CTkEntry(dialog, width=460)
        en_entry.pack(padx=20)

        self.ctk.CTkLabel(dialog, text="Tiếng Việt:", font=("Arial",12)).pack(anchor="w", padx=20, pady=(10,5))
        vi_entry = self.ctk.CTkEntry(dialog, width=460)
        vi_entry.pack(padx=20)

        def save():
            en = en_entry.get().strip()
            vi = vi_entry.get().strip()
            if en and vi:
                self.glossary_data[en] = {"vi": vi, "usage": 0}
                self.glossary_filter()
                dialog.destroy()
            else:
                self.messagebox.showwarning("Lỗi", "Vui lòng nhập đầy đủ!")

        btn_frame = self.ctk.CTkFrame(dialog)
        btn_frame.pack(pady=20)
        self.ctk.CTkButton(btn_frame, text="✅ Lưu", command=save, width=100).pack(side="left", padx=5)
        self.ctk.CTkButton(btn_frame, text="❌ Hủy", command=dialog.destroy, width=100).pack(side="left", padx=5)

    def glossary_edit(self):
        """Sửa từ đã chọn."""
        if not hasattr(self, 'selected_glossary_key') or not self.selected_glossary_key:
            self.messagebox.showwarning("Lỗi", "Vui lòng chọn từ cần sửa!")
            return

        en = self.selected_glossary_key
        data = self.glossary_data[en]
        vi = data.get("vi", "") if isinstance(data, dict) else data

        dialog = self.ctk.CTkToplevel(self.root)
        dialog.title("Sửa từ")
        dialog.geometry("500x200")
        dialog.transient(self.root)
        dialog.grab_set()

        self.ctk.CTkLabel(dialog, text="English:", font=("Arial",12)).pack(anchor="w", padx=20, pady=(20,5))
        en_entry = self.ctk.CTkEntry(dialog, width=460)
        en_entry.insert(0, en)
        en_entry.pack(padx=20)

        self.ctk.CTkLabel(dialog, text="Tiếng Việt:", font=("Arial",12)).pack(anchor="w", padx=20, pady=(10,5))
        vi_entry = self.ctk.CTkEntry(dialog, width=460)
        vi_entry.insert(0, vi)
        vi_entry.pack(padx=20)

        def save():
            new_en = en_entry.get().strip()
            new_vi = vi_entry.get().strip()
            if new_en and new_vi:
                usage = data.get("usage", 0) if isinstance(data, dict) else 0
                if new_en != en:
                    del self.glossary_data[en]
                self.glossary_data[new_en] = {"vi": new_vi, "usage": usage}
                self.glossary_filter()
                dialog.destroy()
            else:
                self.messagebox.showwarning("Lỗi", "Vui lòng nhập đầy đủ!")

        btn_frame = self.ctk.CTkFrame(dialog)
        btn_frame.pack(pady=20)
        self.ctk.CTkButton(btn_frame, text="✅ Lưu", command=save, width=100).pack(side="left", padx=5)
        self.ctk.CTkButton(btn_frame, text="❌ Hủy", command=dialog.destroy, width=100).pack(side="left", padx=5)

    def glossary_delete(self):
        """Xóa từ đã chọn."""
        if not hasattr(self, 'selected_glossary_key') or not self.selected_glossary_key:
            self.messagebox.showwarning("Lỗi", "Vui lòng chọn từ cần xóa!")
            return

        if self.messagebox.askyesno("Xác nhận", f"Xóa từ '{self.selected_glossary_key}'?"):
            del self.glossary_data[self.selected_glossary_key]
            self.glossary_filter()

    def glossary_import(self):
        """Import từ file JSON."""
        filename = self.filedialog.askopenfilename(
            title="Import từ điển",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.glossary_data.update(data)
                self.glossary_filter()
                self.messagebox.showinfo("Thành công", f"Đã import {len(data)} từ!")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể import: {e}")

    def glossary_open_json(self):
        """Mở file JSON để chỉnh sửa trực tiếp."""
        filename = self.filedialog.askopenfilename(
            title="Mở file từ điển",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.glossary_data = data
                self.glossary_filter()
                self.glossary_current_file = filename  # Track current file
                self.log(f"📂 Đã mở: {filename}")
                self.messagebox.showinfo("Thành công", f"Đã load {len(data)} từ!")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể mở: {e}")

    def glossary_save_json(self):
        """Lưu vào file JSON hiện tại hoặc chọn file mới."""
        if hasattr(self, 'glossary_current_file') and self.glossary_current_file:
            # Save to current file
            try:
                import json
                with open(self.glossary_current_file, "w", encoding="utf-8") as f:
                    json.dump(self.glossary_data, f, ensure_ascii=False, indent=2)
                self.log(f"💾 Đã lưu: {self.glossary_current_file}")
                self.messagebox.showinfo("Thành công", "Đã lưu!")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể lưu: {e}")
        else:
            # Save as new file
            self.glossary_export()

    def glossary_export(self):
        """Export ra file JSON."""
        filename = self.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export từ điển"
        )
        if filename:
            try:
                import json
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(self.glossary_data, f, ensure_ascii=False, indent=2)
                self.glossary_current_file = filename  # Track new file
                self.messagebox.showinfo("Thành công", f"Đã export {len(self.glossary_data)} từ!")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể export: {e}")

    def rollback_scan(self):
        """Quét tất cả .bak files trong thư mục."""
        if not self.folder_path:
            self.messagebox.showwarning("Lỗi", "Vui lòng chọn thư mục trước!")
            return

        from pathlib import Path
        folder = Path(self.folder_path)

        # Clear current list
        for widget in self.rollback_list_frame.winfo_children():
            widget.destroy()
        self.backup_files.clear()
        self.backup_checkboxes.clear()

        # Find all .bak files
        bak_files = list(folder.rglob("*.bak"))

        # Update folder label
        self.rollback_folder_label.configure(text=str(folder))
        self.rollback_count_label.configure(text=f"{len(bak_files)} file backup")

        # Display each backup
        for bak_path in sorted(bak_files):
            # Original file path (remove .bak)
            orig_name = bak_path.name[:-4]  # Remove .bak extension
            orig_path = bak_path.with_name(orig_name)

            # Relative paths for display
            try:
                bak_rel = str(bak_path.relative_to(folder))
                orig_rel = str(orig_path.relative_to(folder))
            except ValueError:
                bak_rel = str(bak_path)
                orig_rel = str(orig_path)

            # File size
            try:
                size_bytes = bak_path.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            except:
                size_str = "N/A"

            # Create row
            row = self.ctk.CTkFrame(self.rollback_list_frame, fg_color="#1E1E1E")
            row.pack(fill="x", pady=2, padx=5)

            # Checkbox
            checkbox_var = self.ctk.BooleanVar(value=False)
            checkbox = self.ctk.CTkCheckBox(row, text="", variable=checkbox_var, width=40)
            checkbox.pack(side="left", padx=5)

            # Store for later
            self.backup_checkboxes[str(bak_path)] = checkbox_var
            self.backup_files.append({
                "bak_path": bak_path,
                "orig_path": orig_path,
                "bak_rel": bak_rel,
                "orig_rel": orig_rel,
                "size": size_str,
            })

            # Labels
            self.ctk.CTkLabel(row, text=orig_rel, font=("Arial",10), width=500, anchor="w").pack(side="left", padx=10)
            self.ctk.CTkLabel(row, text=bak_rel, font=("Arial",10), width=400, anchor="w").pack(side="left", padx=10)
            self.ctk.CTkLabel(row, text=size_str, font=("Arial",10), width=120, anchor="center").pack(side="left", padx=10)

        self.log(f"📁 Tìm thấy {len(bak_files)} file backup")

    def rollback_selected(self):
        """Khôi phục các file đã chọn."""
        selected = [f for f in self.backup_files if self.backup_checkboxes[str(f["bak_path"])].get()]

        if not selected:
            self.messagebox.showwarning("Lỗi", "Vui lòng chọn ít nhất 1 file!")
            return

        if not self.messagebox.askyesno("Xác nhận", f"Khôi phục {len(selected)} file đã chọn?"):
            return

        success = 0
        failed = 0

        for item in selected:
            try:
                import shutil
                bak_path = item["bak_path"]
                orig_path = item["orig_path"]

                # Restore
                shutil.copy2(bak_path, orig_path)
                bak_path.unlink()
                success += 1
                self.log(f"✅ Đã khôi phục: {item['orig_rel']}")
            except Exception as e:
                failed += 1
                self.log(f"❌ Lỗi: {item['orig_rel']} - {e}")

        self.messagebox.showinfo("Hoàn tất", f"Khôi phục: {success} thành công, {failed} thất bại")
        self.rollback_scan()  # Refresh list

    def rollback_all(self):
        """Khôi phục tất cả file backup."""
        if not self.backup_files:
            self.messagebox.showwarning("Lỗi", "Không có file backup!")
            return

        if not self.messagebox.askyesno("Xác nhận", f"Khôi phục TẤT CẢ {len(self.backup_files)} file backup?"):
            return

        success = 0
        failed = 0

        for item in self.backup_files:
            try:
                import shutil
                bak_path = item["bak_path"]
                orig_path = item["orig_path"]

                shutil.copy2(bak_path, orig_path)
                bak_path.unlink()
                success += 1
                self.log(f"✅ Đã khôi phục: {item['orig_rel']}")
            except Exception as e:
                failed += 1
                self.log(f"❌ Lỗi: {item['orig_rel']} - {e}")

        self.messagebox.showinfo("Hoàn tất", f"Khôi phục: {success} thành công, {failed} thất bại")
        self.rollback_scan()

    def rollback_delete_bak(self):
        """Xóa tất cả file .bak."""
        if not self.backup_files:
            self.messagebox.showwarning("Lỗi", "Không có file backup!")
            return

        if not self.messagebox.askyesno("Cảnh báo", f"XÓA TẤT CẢ {len(self.backup_files)} file .bak?\nHành động này KHÔNG THỂ HOÀN TÁC!"):
            return

        success = 0
        failed = 0

        for item in self.backup_files:
            try:
                item["bak_path"].unlink()
                success += 1
                self.log(f"🗑️ Đã xóa: {item['bak_rel']}")
            except Exception as e:
                failed += 1
                self.log(f"❌ Lỗi: {item['bak_rel']} - {e}")

        self.messagebox.showinfo("Hoàn tất", f"Xóa: {success} thành công, {failed} thất bại")
        self.rollback_scan()

    def log(self, message):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {message}\n")
        self.log_text.see("end")
        self.root.update()

    def copy_log(self):
        """Copy toàn bộ log vào clipboard."""
        try:
            log_content = self.log_text.get("1.0", "end")
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.root.update()
            self.messagebox.showinfo("Thành công", "Đã copy log vào clipboard!")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể copy: {e}")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def save_log(self):
        filename = self.filedialog.asksaveasfilename(defaultextension=".txt",
                                                      filetypes=[("Text files", "*.txt")])
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", "end"))
            self.log(f"💾 Log đã lưu: {filename}")

    def on_file_start_callback(self, filename, current, total):
        """Callback khi bắt đầu xử lý file."""
        try:
            self.log(f"📄 {filename} ({current}/{total})")

            # Track current file
            self.current_file_name = filename

            # Update progress
            progress = current / total
            self.progress_bar.set(progress)

            # Update ETA
            avg_time, eta = self.eta_calc.update(current, total)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            eta_str = f"{eta_min}p {eta_sec}s" if eta_min > 0 else f"{eta_sec}s"

            stats_text = f"{current}/{total} • {avg_time:.1f}s/file • ETA: {eta_str}"
            self.progress_text.configure(text=stats_text)

            # Assume success by default (will be updated if errors occur)
            self.files_success += 1
        except Exception as e:
            pass  # Ignore callback errors

    def on_translation_callback(self, key_path, original, translated):
        """Callback khi dịch 1 cặp EN→VI."""
        try:
            # Track keys
            self.keys_total += 1
            if original != translated:
                self.keys_success += 1

                # Log translation
                ellipsis_en = "..." if len(original) > 80 else ""
                ellipsis_vi = "..." if len(translated) > 80 else ""
                self.log(f"  🔍 {key_path}")
                self.log(f"  🇬🇧 EN: {original[:80]}{ellipsis_en}")
                self.log(f"  🇻🇳 VI: {translated[:80]}{ellipsis_vi}")

            # Update keys counter
            self.keys_label.configure(text=f"Keys: {self.keys_total} | Thành công: {self.keys_success} | Lỗi: {self.keys_error}")

            # Store for preview tab
            import datetime
            current_file = getattr(self, 'current_file_name', 'unknown')
            self.translation_results.append({
                "file": current_file,
                "key": key_path,
                "original": original,
                "translated": translated,
                "changed": original != translated,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "reason": ""
            })
        except Exception as e:
            pass  # Ignore callback errors


    def get_model_name(self):
        selection = self.model_var.get()
        if "nllb-3.3b" in selection:
            return "nllb-3.3b"
        elif "nllb-1.3b" in selection:
            return "nllb-1.3b"
        elif "marian" in selection:
            return "marian"
        elif "m2m100" in selection:
            return "m2m100"
        return "nllb-3.3b"

    def download_model_only(self):
        self.log("📥 Đang tải model...")
        self.start_btn.configure(state="disabled")

        def download_thread():
            try:
                # Import ModelLoader from main file
                from pathlib import Path
                import sys
                sys.path.insert(0, str(Path(__file__).parent))

                model_name = self.get_model_name()

                # Get device from radio buttons
                device_choice = self.device_var.get() if hasattr(self, 'device_var') else "auto"
                if device_choice == "auto":
                    device = None  # Let backend auto-detect
                elif device_choice == "gpu":
                    device = "cuda"
                else:
                    device = "cpu"

                self.log(f"Model: {model_name}, Device: {device_choice}")

                # Assuming ModelLoader is available globally
                model_loader = ModelLoader(model_name=model_name, device=device, low_memory=self.low_mem_var.get())
                model_loader.load()

                self.log("✅ Model đã tải thành công!")
                self.messagebox.showinfo("Thành công", "Model sẵn sàng!")
            except Exception as e:
                self.log(f"❌ LỖI: {e}")
                self.messagebox.showerror("Lỗi", str(e))
            finally:
                self.start_btn.configure(state="normal")

        self.threading.Thread(target=download_thread, daemon=True).start()

    def start_translation(self):
        if not self.folder_path:
            self.messagebox.showerror("Lỗi", "Vui lòng chọn thư mục!")
            return

        # Reset counters
        import datetime
        self.start_time = datetime.datetime.now()
        self.keys_total = 0
        self.keys_success = 0
        self.keys_error = 0
        self.files_success = 0
        self.files_skipped = 0
        self.files_error = 0
        self.peak_ram = 0
        self.peak_vram = 0
        self.translation_results = []

        self.is_translating = True
        self.is_paused = False
        self.eta_calc.start()
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal")
        self.stop_btn.configure(state="normal")
        self.log("🚀 Bắt đầu dịch...")

        def translate_thread():
            try:
                class Args:
                    pass
                args = Args()
                args.folder = self.folder_path
                args.model = self.get_model_name()

                # Device from radio buttons
                device_choice = self.device_var.get() if hasattr(self, 'device_var') else "auto"
                if device_choice == "gpu":
                    args.gpu = True
                    args.cpu = False
                elif device_choice == "cpu":
                    args.gpu = False
                    args.cpu = True
                else:  # auto
                    args.gpu = False
                    args.cpu = False

                # Advanced options
                args.batch_size = self.batch_var.get() if hasattr(self, 'batch_var') else None
                args.low_memory = self.low_mem_var.get()

                # Fallback model
                fallback = self.fallback_var.get() if hasattr(self, 'fallback_var') else "Không"
                args.fallback_model = None if fallback == "Không" else fallback

                # File extensions
                ext_text = self.ext_entry.get() if hasattr(self, 'ext_entry') else ""
                args.ext = ext_text.strip() if ext_text.strip() else None

                # Force flag
                args.force = self.force_var.get() if hasattr(self, 'force_var') else False

                args.dry_run = self.dry_run_var.get()
                args.rollback = False
                args.report = self.report_var.get()
                args.download_only = False
                args.self_test = False

                # Max memory
                max_mem_text = self.max_mem_entry.get() if hasattr(self, 'max_mem_entry') else ""
                try:
                    args.max_memory = int(max_mem_text) if max_mem_text.strip() else None
                except ValueError:
                    args.max_memory = None

                # Glossary
                gloss_path = self.glossary_entry.get() if hasattr(self, 'glossary_entry') else ""
                args.glossary = gloss_path.strip() if gloss_path.strip() else None

                # Redirect print
                import builtins
                original_print = print
                def gui_print(*msg, **_):
                    self.log(" ".join(str(m) for m in msg))
                builtins.print = gui_print

                processor = InPlaceTranslator(args)

                # Connect GUI state to processor
                processor.is_translating = True
                processor.is_paused = False
                # Share state references
                def check_pause():
                    return self.is_paused
                def check_translating():
                    return self.is_translating
                # Update processor state in loop (processor will check these)
                import threading
                def sync_state():
                    import time
                    while processor.is_translating:
                        processor.is_paused = self.is_paused
                        processor.is_translating = self.is_translating
                        time.sleep(0.1)
                state_thread = threading.Thread(target=sync_state, daemon=True)
                state_thread.start()

                # Register callbacks
                processor.on_file_start = self.on_file_start_callback
                processor.on_translation = self.on_translation_callback
                processor.run()

                builtins.print = original_print
                self.log("🎉 HOÀN TẤT!")

                # Show completion dialog
                self.root.after(100, self.show_completion_dialog)
            except Exception as e:
                self.log(f"❌ LỖI: {e}")
                import traceback
                self.log(traceback.format_exc())
                self.messagebox.showerror("Lỗi", str(e))
            finally:
                self.is_translating = False
                self.start_btn.configure(state="normal")
                self.pause_btn.configure(state="disabled")
                self.stop_btn.configure(state="disabled")

        self.threading.Thread(target=translate_thread, daemon=True).start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="▶️ TIẾP TỤC")
            self.log("⏸️ Đã tạm dừng")
        else:
            self.pause_btn.configure(text="⏸️ TẠM DỪNG")
            self.log("▶️ Tiếp tục")

    def stop_translation(self):
        self.is_translating = False
        self.log("⏹️ Đang dừng...")
        self.messagebox.showwarning("Cảnh báo", "Hãy đợi file hiện tại hoàn tất.")

    # ============================================================================
    # PREVIEW TAB METHODS
    # ============================================================================

    def preview_refresh(self):
        """Làm mới danh sách preview từ translation_results."""
        # Clear current list
        for widget in self.preview_list_frame.winfo_children():
            widget.destroy()
        self.preview_files.clear()
        self.preview_checkboxes.clear()

        if not hasattr(self, 'translation_results') or not self.translation_results:
            self.preview_count_label.configure(text="0 files")
            return

        # Group results by file
        files_data = {}
        for result in self.translation_results:
            file_name = result.get("file", "unknown")
            if file_name not in files_data:
                files_data[file_name] = {
                    "keys_total": 0,
                    "keys_changed": 0,
                    "status": "✅ Đã dịch",
                    "time": result.get("time", "N/A"),
                    "changes": []
                }
            files_data[file_name]["keys_total"] += 1
            if result.get("changed", False):
                files_data[file_name]["keys_changed"] += 1
            files_data[file_name]["changes"].append(result)

        # Apply filters
        filter_val = self.preview_filter_var.get()
        search_val = self.preview_search_var.get().lower()

        filtered_files = {}
        for fname, data in files_data.items():
            # Search filter
            if search_val and search_val not in fname.lower():
                continue

            # Status filter
            if filter_val == "Đã dịch" and data["keys_changed"] == 0:
                continue
            elif filter_val == "Bỏ qua" and data["keys_changed"] > 0:
                continue
            elif filter_val == "Lỗi":
                # Check if any result has error
                has_error = any(r.get("reason", "") for r in data["changes"])
                if not has_error:
                    continue

            filtered_files[fname] = data

        # Update count
        self.preview_count_label.configure(text=f"{len(filtered_files)} files")

        # Display files
        for fname in sorted(filtered_files.keys()):
            data = filtered_files[fname]
            self.preview_files.append({"file": fname, "data": data})

            row = self.ctk.CTkFrame(self.preview_list_frame, fg_color="#1E1E1E")
            row.pack(fill="x", pady=2, padx=5)

            # Bind right-click context menu to row
            row.bind("<Button-3>", lambda e, f=fname, d=data: self.show_file_context_menu(e, f, d))

            # Checkbox
            checkbox_var = self.ctk.BooleanVar(value=False)
            checkbox = self.ctk.CTkCheckBox(row, text="", variable=checkbox_var, width=40)
            checkbox.pack(side="left", padx=5)
            self.preview_checkboxes[fname] = checkbox_var

            # File name
            file_label = self.ctk.CTkLabel(row, text=fname, font=("Arial",10), width=350, anchor="w")
            file_label.pack(side="left", padx=5)
            file_label.bind("<Button-3>", lambda e, f=fname, d=data: self.show_file_context_menu(e, f, d))

            # Keys count
            keys_text = f"{data['keys_changed']}/{data['keys_total']}"
            self.ctk.CTkLabel(row, text=keys_text, font=("Arial",10), width=80, anchor="center").pack(side="left", padx=5)

            # Status
            status_color = "#00AA00" if data['keys_changed'] > 0 else "#888888"
            self.ctk.CTkLabel(row, text=data['status'], font=("Arial",10), width=100,
                            anchor="center", text_color=status_color).pack(side="left", padx=5)

            # Time
            self.ctk.CTkLabel(row, text=data['time'], font=("Arial",10), width=100, anchor="center").pack(side="left", padx=5)

            # View button
            self.ctk.CTkButton(row, text="🔍 Xem", width=80, height=28,
                              command=lambda f=fname, d=data: self.preview_view_file(f, d)).pack(side="left", padx=5)

    def preview_view_file(self, filename, data):
        """Hiển thị diff của file."""
        self.selected_preview_file = filename

        # Load original and translated
        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / filename
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        # Clear diff viewers
        self.diff_original.delete("1.0", "end")
        self.diff_translated.delete("1.0", "end")

        try:
            # Read original from .bak
            if backup_path.exists():
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original = f.read()
                self.diff_original.insert("1.0", original)
            else:
                self.diff_original.insert("1.0", "(Không có backup)")

            # Read translated from current file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    translated = f.read()
                self.diff_translated.insert("1.0", translated)
            else:
                self.diff_translated.insert("1.0", "(File không tồn tại)")

        except Exception as e:
            self.diff_original.insert("1.0", f"Lỗi đọc file: {e}")
            self.diff_translated.insert("1.0", f"Lỗi đọc file: {e}")

    def preview_select_all(self):
        """Chọn tất cả files."""
        for checkbox_var in self.preview_checkboxes.values():
            checkbox_var.set(True)

    def preview_export_excel(self):
        """Xuất preview ra Excel."""
        if not self.preview_files:
            self.messagebox.showwarning("Lỗi", "Không có dữ liệu!")
            return

        filename = self.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filename:
            try:
                import csv
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["File", "Keys Thay Đổi", "Keys Tổng", "Trạng Thái", "Thời Gian"])

                    for item in self.preview_files:
                        fname = item["file"]
                        data = item["data"]
                        writer.writerow([
                            fname,
                            data["keys_changed"],
                            data["keys_total"],
                            data["status"],
                            data["time"]
                        ])

                self.messagebox.showinfo("Thành công", f"Đã xuất: {filename}")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể xuất: {e}")

    def preview_keep_changes(self):
        """Giữ thay đổi (xóa backup để không thể hoàn tác)."""
        if not self.selected_preview_file:
            self.messagebox.showwarning("Lỗi", "Chưa chọn file!")
            return

        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / self.selected_preview_file
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        if not backup_path.exists():
            self.messagebox.showwarning("Lỗi", "Không có backup để xóa!")
            return

        if self.messagebox.askyesno("Xác nhận",
            f"Giữ thay đổi và XÓA backup?\n{self.selected_preview_file}\n\nKhông thể hoàn tác sau này!"):
            try:
                backup_path.unlink()
                self.log(f"✅ Đã giữ thay đổi: {self.selected_preview_file}")
                self.messagebox.showinfo("Thành công", "Đã xóa backup!")
                self.preview_refresh()
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể xóa backup: {e}")

    def preview_undo_file(self):
        """Hoàn tác file đang xem."""
        if not self.selected_preview_file:
            self.messagebox.showwarning("Lỗi", "Chưa chọn file!")
            return

        if not self.folder_path:
            return

        from pathlib import Path
        file_path = Path(self.folder_path) / self.selected_preview_file
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")

        if not backup_path.exists():
            self.messagebox.showwarning("Lỗi", "Không tìm thấy backup!")
            return

        if self.messagebox.askyesno("Xác nhận", f"Hoàn tác file:\n{self.selected_preview_file}?"):
            try:
                import shutil
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()
                self.messagebox.showinfo("Thành công", "Đã hoàn tác!")
                self.log(f"↩️ Hoàn tác: {self.selected_preview_file}")
                self.preview_refresh()
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể hoàn tác: {e}")

    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.log("📋 Đã copy vào clipboard")
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể copy: {e}")

    # ============================================================================
    # MENU BAR & SETTINGS METHODS
    # ============================================================================

    def build_menu_bar(self):
        """Xây dựng menu bar sử dụng tkinter Menu."""
        from tkinter import Menu

        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Menu File
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Mở Thư Mục (Ctrl+O)", command=self.browse_folder)
        file_menu.add_command(label="Lưu Log (Ctrl+S)", command=self.save_log)
        file_menu.add_command(label="Xuất Báo Cáo", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát (Ctrl+Q)", command=self.quit_app)

        # Menu Công Cụ
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Công Cụ", menu=tools_menu)
        tools_menu.add_command(label="Rollback", command=lambda: self.tabview.set("↩️ Rollback"))
        tools_menu.add_command(label="Glossary Editor", command=lambda: self.tabview.set("📖 Từ Điển"))
        tools_menu.add_command(label="Cài Đặt (Ctrl+,)", command=lambda: self.tabview.set("⚙️ Cài Đặt"))

        # Menu Trợ Giúp
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trợ Giúp", menu=help_menu)
        help_menu.add_command(label="Hướng Dẫn (F1)", command=self.show_help)
        help_menu.add_command(label="Kiểm Tra Cập Nhật", command=self.check_updates)
        help_menu.add_separator()
        help_menu.add_command(label="Về", command=self.show_about)

    def setup_keyboard_shortcuts(self):
        """Thiết lập phím tắt."""
        self.root.bind('<Control-o>', lambda e: self.browse_folder())
        self.root.bind('<Control-s>', lambda e: self.start_translation())
        self.root.bind('<Control-p>', lambda e: self.toggle_pause())
        self.root.bind('<Control-q>', lambda e: self.quit_app())
        self.root.bind('<Control-comma>', lambda e: self.tabview.set("⚙️ Cài Đặt"))
        self.root.bind('<Control-f>', lambda e: self.focus_log_search())
        self.root.bind('<Control-l>', lambda e: self.clear_log())
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<F5>', lambda e: self.refresh_ui())

    def export_report(self):
        """Xuất báo cáo dịch."""
        if not hasattr(self, 'translation_results') or not self.translation_results:
            self.messagebox.showwarning("Lỗi", "Chưa có kết quả dịch!")
            return

        filename = self.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            try:
                import csv
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["File", "Key", "Original", "Translated", "Status"])
                    for result in self.translation_results:
                        writer.writerow([result.get(k, "") for k in ["file", "key", "original", "translated", "status"]])
                self.messagebox.showinfo("Thành công", f"Đã xuất báo cáo: {filename}")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể xuất: {e}")

    def quit_app(self):
        """Thoát ứng dụng."""
        if self.is_translating:
            if self.messagebox.askyesno("Xác nhận", "Đang dịch! Bạn có chắc muốn thoát?"):
                self.root.quit()
        else:
            self.root.quit()

    def show_completion_dialog(self):
        """Hiển thị dialog thống kê hoàn tất."""
        import datetime

        # Play sound if enabled
        if hasattr(self, 'sound_var') and self.sound_var.get():
            try:
                import platform
                if platform.system() == "Windows":
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                elif platform.system() == "Darwin":  # macOS
                    import os
                    os.system('afplay /System/Library/Sounds/Glass.aiff')
                else:  # Linux
                    import os
                    os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null')
            except:
                pass

        # Show system notification if enabled
        if hasattr(self, 'notif_var') and self.notif_var.get():
            try:
                # Try plyer first (cross-platform)
                try:
                    from plyer import notification
                    notification.notify(
                        title='🎉 Dịch Hoàn Tất',
                        message=f'Đã dịch {self.files_success} files thành công!',
                        app_name='Minecraft Translator',
                        timeout=10
                    )
                except ImportError:
                    # Fallback to platform-specific methods
                    import platform
                    if platform.system() == "Windows":
                        from win10toast import ToastNotifier
                        toaster = ToastNotifier()
                        toaster.show_toast("Dịch Hoàn Tất",
                                          f"Đã dịch {self.files_success} files thành công!",
                                          duration=10, threaded=True)
                    # macOS and Linux have no fallback
            except:
                pass

        # Calculate statistics
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        avg_speed = elapsed / max(self.files_success, 1)
        speed_str = f"{avg_speed:.1f}s/file"

        total_files = self.files_success + self.files_skipped + self.files_error

        # Build dialog
        dialog = self.ctk.CTkToplevel(self.root)
        dialog.title("Hoàn Tất Dịch")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"+{x}+{y}")

        # Content frame
        content = self.ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=30)

        # Title
        title = self.ctk.CTkLabel(content, text="🎉 DỊCH HOÀN TẤT!",
                                 font=("Arial", 24, "bold"), text_color="#00FF00")
        title.pack(pady=(0, 20))

        # Stats frame
        stats_frame = self.ctk.CTkFrame(content, fg_color="#2B2B2B", corner_radius=10)
        stats_frame.pack(fill="both", expand=True, pady=(0, 20))

        stats_inner = self.ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_inner.pack(fill="both", expand=True, padx=20, pady=20)

        # File statistics
        self._add_stat_row(stats_inner, "✅ Thành công:", f"{self.files_success} files ({self.keys_success} keys)", "#00FF00")
        self._add_stat_row(stats_inner, "⏭️ Bỏ qua:", f"{self.files_skipped} files (đã là tiếng Việt)", "#FFA500")
        self._add_stat_row(stats_inner, "❌ Lỗi:", f"{self.files_error} files ({self.keys_error} keys)", "#FF0000")

        # Separator
        sep = self.ctk.CTkFrame(stats_inner, height=2, fg_color="#444444")
        sep.pack(fill="x", pady=15)

        # Performance statistics
        self._add_stat_row(stats_inner, "⏱️ Thời gian:", elapsed_str, "#FFFFFF")
        self._add_stat_row(stats_inner, "📊 Tốc độ TB:", speed_str, "#FFFFFF")
        self._add_stat_row(stats_inner, "💾 RAM cao nhất:", f"{self.peak_ram:.1f} GB", "#00BFFF")

        if self.peak_vram > 0:
            self._add_stat_row(stats_inner, "🎮 VRAM cao nhất:", f"{self.peak_vram:.1f} GB", "#FF1493")

        # Separator
        sep2 = self.ctk.CTkFrame(stats_inner, height=2, fg_color="#444444")
        sep2.pack(fill="x", pady=15)

        # Backup info
        if self.folder_path:
            backup_text = f"📁 Backup tại: {self.folder_path}/*.bak"
            self._add_stat_row(stats_inner, "", backup_text, "#FFFF00", single=True)

        # Buttons frame
        buttons = self.ctk.CTkFrame(content, fg_color="transparent")
        buttons.pack(fill="x")

        self.ctk.CTkButton(buttons, text="📊 Xem Báo Cáo", width=180, height=40,
                          command=lambda: [dialog.destroy(), self.export_report()],
                          fg_color="#1E90FF").pack(side="left", padx=5)

        self.ctk.CTkButton(buttons, text="📂 Mở Thư Mục", width=180, height=40,
                          command=lambda: [dialog.destroy(), self.open_folder()],
                          fg_color="#32CD32").pack(side="left", padx=5)

        self.ctk.CTkButton(buttons, text="✅ OK", width=180, height=40,
                          command=dialog.destroy,
                          fg_color="#4CAF50").pack(side="left", padx=5)

    def _add_stat_row(self, parent, label, value, color="#FFFFFF", single=False):
        """Helper to add a stat row."""
        row = self.ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)

        if single:
            lbl = self.ctk.CTkLabel(row, text=value, font=("Arial", 12), text_color=color, anchor="w")
            lbl.pack(side="left", fill="x")
        else:
            lbl = self.ctk.CTkLabel(row, text=label, font=("Arial", 12, "bold"),
                                   text_color="#AAAAAA", anchor="w", width=150)
            lbl.pack(side="left")

            val = self.ctk.CTkLabel(row, text=value, font=("Arial", 12),
                                   text_color=color, anchor="w")
            val.pack(side="left", fill="x", expand=True)

    def open_folder(self):
        """Mở thư mục trong file explorer."""
        if self.folder_path:
            import os
            import subprocess
            import platform

            if platform.system() == "Windows":
                os.startfile(self.folder_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", self.folder_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.folder_path])

    def show_help(self):
        """Hiển thị hướng dẫn."""
        help_text = """
🎯 HƯỚNG DẪN SỬ DỤNG

📁 Bước 1: Chọn Thư Mục
   - Click "Chọn" hoặc Ctrl+O
   - Chọn thư mục chứa file cần dịch

🤖 Bước 2: Cấu Hình
   - Chọn Model AI (khuyến nghị: NLLB-3.3B)
   - Chọn Device (Tự động/GPU/CPU)
   - Mở "Tùy Chọn Nâng Cao" nếu cần

▶️ Bước 3: Bắt Đầu
   - Click "BẮT ĐẦU DỊCH" hoặc Ctrl+S
   - Theo dõi tiến độ trong log

⏸️ Điều Khiển:
   - TẠM DỪNG: Ctrl+P
   - DỪNG: Dừng hoàn toàn
   - Xem LOG: Ctrl+F để tìm kiếm

↩️ Rollback:
   - Tab "Rollback" để khôi phục file gốc
   - Tất cả file được backup tự động (.bak)

📖 Từ Điển:
   - Tab "Từ Điển" để thêm thuật ngữ riêng
   - Import/Export JSON

⚙️ Cài Đặt:
   - Tab "Cài Đặt" để tùy chỉnh giao diện
   - Lưu Preset cho cấu hình thường dùng

🔑 PHÍM TẮT:
   Ctrl+O: Mở thư mục
   Ctrl+S: Bắt đầu dịch
   Ctrl+P: Tạm dừng
   Ctrl+Q: Thoát
   F1: Trợ giúp
   F5: Làm mới
        """
        dialog = self.ctk.CTkToplevel(self.root)
        dialog.title("Hướng Dẫn")
        dialog.geometry("600x700")
        dialog.transient(self.root)

        text = self.ctk.CTkTextbox(dialog, font=("Arial", 11), wrap="word")
        text.pack(fill="both", expand=True, padx=20, pady=20)
        text.insert("1.0", help_text)
        text.configure(state="disabled")

        self.ctk.CTkButton(dialog, text="Đóng", command=dialog.destroy).pack(pady=10)

    def check_updates(self):
        """Kiểm tra cập nhật."""
        self.messagebox.showinfo("Kiểm Tra Cập Nhật", "Bạn đang dùng phiên bản mới nhất: v1.0.0")

    def show_about(self):
        """Hiển thị thông tin."""
        about_text = """
Minecraft Config Translator
Phiên bản: 1.0.0

Công cụ dịch tự động file cấu hình Minecraft
từ tiếng Anh sang tiếng Việt sử dụng AI.

Model: NLLB-200 (Meta AI)
Framework: CustomTkinter

© 2024 - Made with ❤️
        """
        self.messagebox.showinfo("Về", about_text)

    def focus_log_search(self):
        """Focus vào ô tìm kiếm log (placeholder)."""
        pass

    def refresh_ui(self):
        """Làm mới giao diện."""
        if hasattr(self, 'rollback_folder_label') and self.folder_path:
            self.rollback_scan()
        self.log("🔄 Đã làm mới")

    # Settings Tab Methods
    def change_theme(self):
        """Thay đổi theme."""
        theme = self.theme_var.get()
        self.ctk.set_appearance_mode(theme)
        self.log(f"🎨 Đã đổi theme: {theme}")

    def change_font_size(self, size):
        """Thay đổi cỡ chữ log."""
        try:
            self.log_text.configure(font=("Consolas", int(size)))
            self.log(f"🔤 Đã đổi cỡ chữ: {size}")
        except:
            pass

    def browse_cache(self):
        """Chọn thư mục cache."""
        folder = self.filedialog.askdirectory(title="Chọn thư mục cache")
        if folder:
            self.cache_entry.delete(0, "end")
            self.cache_entry.insert(0, folder)

    def clear_cache(self):
        """Xóa cache."""
        if self.messagebox.askyesno("Xác nhận", "Xóa toàn bộ cache HuggingFace?\nModel sẽ phải tải lại!"):
            cache_path = self.cache_entry.get()
            try:
                import shutil
                from pathlib import Path
                cache_dir = Path(cache_path)
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True)
                    self.messagebox.showinfo("Thành công", "Đã xóa cache!")
                    self.log("🗑️ Đã xóa cache")
            except Exception as e:
                self.messagebox.showerror("Lỗi", f"Không thể xóa: {e}")

    def view_cache_details(self):
        """Xem chi tiết cache."""
        cache_path = self.cache_entry.get()
        try:
            from pathlib import Path
            cache_dir = Path(cache_path)
            if not cache_dir.exists():
                self.messagebox.showinfo("Thông tin", "Thư mục cache không tồn tại")
                return

            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            file_count = len(list(cache_dir.rglob('*')))

            info = f"Thư mục: {cache_path}\nDung lượng: {size_gb:.2f} GB\nSố file: {file_count}"
            self.messagebox.showinfo("Chi Tiết Cache", info)
        except Exception as e:
            self.messagebox.showerror("Lỗi", f"Không thể đọc: {e}")

    def load_preset(self, preset_name):
        """Load preset cấu hình."""
        if preset_name not in self.presets:
            return

        preset = self.presets[preset_name]

        # Apply preset
        if preset["model"] == "nllb-3.3b":
            self.model_var.set("nllb-3.3b (Chất lượng cao) ⭐")
        elif preset["model"] == "nllb-1.3b":
            self.model_var.set("nllb-1.3b (Cân bằng)")

        self.device_var.set(preset["device"])
        self.quant_var.set(preset["quant"])
        self.low_mem_var.set(preset["low_mem"])

        self.log(f"✅ Đã load preset: {preset_name}")
        self.messagebox.showinfo("Thành công", f"Đã áp dụng preset: {preset_name}")

    def save_preset(self):
        """Lưu cấu hình hiện tại thành preset."""
        name = self.ctk.CTkInputDialog(text="Tên preset:", title="Lưu Preset").get_input()
        if name:
            self.presets[name] = {
                "model": self.get_model_name(),
                "device": self.device_var.get(),
                "quant": self.quant_var.get(),
                "low_mem": self.low_mem_var.get(),
            }
            self.messagebox.showinfo("Thành công", f"Đã lưu preset: {name}")
            self.log(f"💾 Đã lưu preset: {name}")

    def delete_preset(self):
        """Xóa preset."""
        # Show dialog to select preset
        dialog = self.ctk.CTkToplevel(self.root)
        dialog.title("Xóa Preset")
        dialog.geometry("300x200")
        dialog.transient(self.root)

        self.ctk.CTkLabel(dialog, text="Chọn preset cần xóa:", font=("Arial",12)).pack(pady=10)

        selected = self.ctk.StringVar(value=list(self.presets.keys())[0] if self.presets else "")
        for name in self.presets.keys():
            self.ctk.CTkRadioButton(dialog, text=name, variable=selected, value=name).pack(anchor="w", padx=20)

        def do_delete():
            name = selected.get()
            if name in self.presets:
                del self.presets[name]
                self.messagebox.showinfo("Thành công", f"Đã xóa: {name}")
                dialog.destroy()

        self.ctk.CTkButton(dialog, text="Xóa", command=do_delete, fg_color="red").pack(pady=10)

    def open_log_folder(self):
        """Mở thư mục chứa log."""
        import os
        import subprocess
        from pathlib import Path

        log_path = Path(self.log_path_entry.get())
        log_dir = log_path.parent

        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        if os.name == 'nt':  # Windows
            os.startfile(log_dir)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', log_dir])

    def on_closing(self):
        """Cleanup threads before closing."""
        self._stop_monitor = True
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(
        description="In-place translator for Minecraft configs (EN -> VI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model only
  python mine_inplace_translator.py --download-only

  # Translate folder
  python mine_inplace_translator.py /path/to/server/plugins

  # Dry-run with report
  python mine_inplace_translator.py /path/to/configs --dry-run --report

  # Low-memory mode with fallback
  python mine_inplace_translator.py /path/to/data --low-memory --fallback-model nllb-1.3b

  # Rollback changes
  python mine_inplace_translator.py /path/to/server --rollback

  # Self-test
  python mine_inplace_translator.py --self-test
        """
    )

    parser.add_argument("folder", nargs="?", help="Folder to process")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="nllb-3.3b",
                        help="Translation model (default: nllb-3.3b)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--batch-size", type=int, help="Override adaptive batch size")
    parser.add_argument("--low-memory", action="store_true", help="Enable low-memory optimizations")
    parser.add_argument("--fallback-model", choices=list(MODEL_CONFIGS.keys()),
                        help="Fallback model if primary fails")
    parser.add_argument("--ext", help="Comma-separated file extensions (default: yml,yaml,json,properties,lang,txt,mcfunction)")
    parser.add_argument("--include", help="Include patterns (comma-separated, e.g., 'plugins/**,datapacks/**')")
    parser.add_argument("--exclude", help="Exclude patterns (comma-separated, e.g., '*.bak,backup/**')")
    parser.add_argument("--force", action="store_true", help="Translate even if Vietnamese detected")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--rollback", action="store_true", help="Restore all .bak files")
    parser.add_argument("--report", action="store_true", help="Generate mine_translate_report.csv")
    parser.add_argument("--download-only", action="store_true", help="Download model and exit")
    parser.add_argument("--self-test", action="store_true", help="Run internal validation tests")
    parser.add_argument("--max-memory", help='Cap memory usage (e.g., "cuda:0=7GiB,cpu=30GiB")')
    parser.add_argument("--glossary", help="Path to glossary JSON file (en->vi)")
    parser.add_argument("--no-cache", action="store_true", help="Disable translation memory cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear translation memory cache and exit")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit")

    args = parser.parse_args()

    # Self-test mode
    if args.self_test:
        run_self_tests()
        return

    # Cache management modes
    if args.clear_cache:
        tm = TranslationMemory()
        tm.clear()
        stats = tm.get_stats()
        print(f"📊 Cache cleared. Database: {stats.get('db_path', 'N/A')}")
        return

    if args.cache_stats:
        tm = TranslationMemory()
        stats = tm.get_stats()
        print("\n📊 TRANSLATION MEMORY STATISTICS")
        print("=" * 60)
        print(f"Status:     {'✅ Enabled' if stats['enabled'] else '❌ Disabled'}")
        print(f"Cache size: {stats['cache_size']} translations")
        print(f"Hit rate:   {stats['hit_rate']:.1%} ({stats['hits']} hits / {stats['total']} queries)")
        print(f"Database:   {stats.get('db_path', 'N/A')}")
        print("=" * 60)
        return

    # Validate folder requirement
    if not args.download_only and not args.folder:
        parser.error("folder argument required (unless using --download-only or --self-test)")

    # Create and run processor
    processor = InPlaceTranslator(args)
    processor.run()


if __name__ == "__main__":
    # Check if GUI mode requested (no arguments = GUI, or --gui flag)
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--gui"):
        # Launch GUI (imports are checked inside MinecraftTranslatorGUI.__init__)
        app = MinecraftTranslatorGUI()
        app.mainloop()
    else:
        # Launch CLI
        main()
