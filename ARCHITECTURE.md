# 🏗️ KIẾN TRÚC TOOL DỊCH - PHÂN TÍCH CHI TIẾT

## 📋 MỤC LỤC
1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [So sánh với mô hình đề xuất](#2-so-sánh-với-mô-hình-đề-xuất)
3. [Chi tiết từng module](#3-chi-tiết-từng-module)
4. [Luồng xử lý](#4-luồng-xử-lý)
5. [Các tính năng nâng cao](#5-các-tính-năng-nâng-cao)

---

## 1. KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN ENTRY POINT                              │
│                    main() function                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   InPlaceTranslator                              │
│              (Orchestrator - điều phối chính)                    │
│  - scan_directory(): Scanner                                     │
│  - translate_file(): Coordinator                                 │
│  - generate_report(): Reporter                                   │
└─────────┬───────────────┬─────────────┬─────────────────────────┘
          │               │             │
          ↓               ↓             ↓
    ┌─────────┐   ┌──────────────┐   ┌──────────────┐
    │ Scanner │   │   Handlers   │   │  Translator  │
    │ Module  │   │   (Parser)   │   │   Engine     │
    └─────────┘   └──────────────┘   └──────────────┘
```

---

## 2. SO SÁNH VỚI MÔ HÌNH ĐỀ XUẤT

### 🎯 **MÔ HÌNH ĐỀ XUẤT (Bạn)**

| Module | Chức năng |
|--------|-----------|
| **Scanner** | Quét thư mục tìm file |
| **Extractor/Parser** | Đọc file, tách text |
| **Translator Engine** | Dịch với AI local |
| **Rebuilder & Validator** | Ghép lại, validate |
| **Translation Memory** | Cache, tránh dịch lại |
| **Glossary** | Thuật ngữ cố định |

### ✅ **MÔ HÌNH THỰC TẾ (Tool hiện tại)**

| Module Đề xuất | Implement Thực tế | Class/Function | Trạng thái |
|----------------|-------------------|----------------|------------|
| **Scanner** | ✅ Có | `InPlaceTranslator.scan_directory()` | ✅ Hoàn chỉnh |
| **Parser** | ✅ Có | `FileHandler` + subclasses | ✅ Hoàn chỉnh |
| **Translator Engine** | ✅ Có | `Translator` + `ModelLoader` | ✅ Hoàn chỉnh |
| **Token Protection** | ✅ Có | `TokenProtector` | ✅ Hoàn chỉnh |
| **Rebuilder** | ✅ Có | `FileHandler.write_file()` | ✅ Hoàn chỉnh |
| **Validator** | ✅ Có | `Translator.validate_*()` | ✅ Hoàn chỉnh |
| **Translation Memory** | ❌ Chưa | N/A | ⚠️ Chưa implement |
| **Glossary** | ✅ Có | `Translator.glossary` | ✅ Hoàn chỉnh |
| **Context-Aware** | ✅ Có | `is_code_context()`, `looks_like_code()` | ✅ Mới thêm |

---

## 3. CHI TIẾT TỪNG MODULE

### 📁 **Module 1: SCANNER**

**Class:** `InPlaceTranslator`
**Function:** `scan_directory()`

**Chức năng:**
```python
def scan_directory(self, folder: Path) -> List[Path]:
    """
    Quét thư mục tìm file cần dịch
    """
    # 1. Walk through directory tree
    # 2. Filter by extensions (.yml, .json, .lang, .properties, .txt)
    # 3. Skip backup files (.bak)
    # 4. Return list of file paths
```

**Tính năng:**
- ✅ Quét recursive toàn bộ thư mục
- ✅ Filter theo extension
- ✅ Skip backup files
- ✅ Memory-efficient (generator for large folders)
- ✅ Progress bar với tqdm

**Code thực tế:** `mine_inplace_translator.py:2201-2500`

---

### 📝 **Module 2: EXTRACTOR/PARSER**

**Base Class:** `FileHandler`
**Subclasses:**
- `YAMLHandler` - Parse YAML files
- `JSONHandler` - Parse JSON files
- `PropertiesHandler` - Parse .properties files
- `TextHandler` - Parse .lang/.txt files

**Kiến trúc:**

```
FileHandler (Abstract Base)
├── read_file() → (data, encoding)
├── write_file() → Atomic write với backup
├── translate_file() → Entry point
└── translate_recursive() → Duyệt cấu trúc

YAMLHandler (YAML-specific)
├── Preserves: comments, anchors, structure
├── Uses: ruamel.yaml (not PyYAML)
├── Handles: multiline strings (|, >)
└── Context-aware: checks YAML path

JSONHandler (JSON-specific)
├── Preserves: indentation, structure
├── Uses: json.loads/dumps
└── Handles: nested objects/arrays

PropertiesHandler (.properties)
├── Format: key=value
├── Preserves: comments (#)
└── Handles: escaped characters

TextHandler (.lang/.txt)
├── Line-by-line translation
├── Preserves: empty lines
└── Auto-detects format
```

**Điểm mạnh:**
- ✅ **Preserve structure:** Giữ nguyên cấu trúc file
- ✅ **Context-aware:** Dùng YAML path để quyết định dịch
- ✅ **Atomic writes:** Ghi file atomic (temp file → rename)
- ✅ **Auto backup:** Tự động tạo .bak trước khi ghi

**Code thực tế:** `mine_inplace_translator.py:1739-2200`

---

### 🤖 **Module 3: TRANSLATOR ENGINE**

**Architecture:**

```
┌──────────────────────────────────────────────────────┐
│                  ModelLoader                          │
│  - Load NLLB/Marian/M2M100/Argos                    │
│  - Quantization: 4-bit → 8-bit → full fallback      │
│  - Device detection: CUDA/CPU/macOS/ROCm            │
└────────────────┬─────────────────────────────────────┘
                 │
                 ↓
┌──────────────────────────────────────────────────────┐
│                  TokenProtector                       │
│  - Protect: placeholders, colors, commands           │
│  - Replace: %player% → __P001__, &c → __C001__      │
│  - Restore: After translation                        │
│  - Validate: Check all tokens present                │
└────────────────┬─────────────────────────────────────┘
                 │
                 ↓
┌──────────────────────────────────────────────────────┐
│                   Translator                          │
│  ┌────────────────────────────────────────────────┐ │
│  │ translate_single(text) → (translated, reason) │ │
│  │  1. Check: should_skip()?                     │ │
│  │  2. Detect: language (en/vi/zh/ja/ko/th)     │ │
│  │  3. Protect: tokens                           │ │
│  │  4. Apply: glossary                           │ │
│  │  5. Translate: batch                          │ │
│  │  6. Restore: tokens                           │ │
│  │  7. Validate: length, structure               │ │
│  └────────────────────────────────────────────────┘ │
│                                                       │
│  ┌────────────────────────────────────────────────┐ │
│  │ translate_adjacent(texts[]) → results[]       │ │
│  │  - Group adjacent texts for context            │ │
│  │  - Join with delimiters                        │ │
│  │  - Translate batch, split back                │ │
│  └────────────────────────────────────────────────┘ │
│                                                       │
│  ┌────────────────────────────────────────────────┐ │
│  │ translate_long_text(text) → translated        │ │
│  │  - Chunk by sentences                          │ │
│  │  - Translate chunks                            │ │
│  │  - Join back                                   │ │
│  └────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

**Tính năng:**

#### **3.1. Token Protection System**

```python
PROTECTED_PATTERNS = [
    # PlaceholderAPI conditionals
    r"%if_[^%]+%[^%]*%else%[^%]*%endif%",

    # Math expressions
    r"%\{[^}]+\}%",

    # Standard placeholders
    r"\{[^}]+\}",      # {player}
    r"%[^%\s]+%",      # %player%

    # Color codes
    r"&[0-9a-fk-orA-FK-OR]",  # &c, &l
    r"§[0-9a-fk-orA-FK-OR]",  # §c, §l

    # MiniMessage
    r"<[A-Za-z0-9:_= ,./'\"-]+>",  # <red>, <click:run_command:/cmd>

    # Commands & permissions
    r"/[A-Za-z0-9_\-]+",      # /spawn
    r"[A-Za-z]+\.[A-Za-z0-9_.\-]+",  # permission.node

    # URLs, IPs
    r"https?://[^\s]+",
    r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b",

    # Namespace IDs
    r"[a-z0-9_]+:[a-z0-9_/]+",  # minecraft:diamond_sword
]
```

**Process:**
1. **Protect:** `%player%` → `__PLACEHOLDER_001__`
2. **Translate:** Text với placeholders được thay
3. **Restore:** `__PLACEHOLDER_001__` → `%player%`
4. **Validate:** Check tất cả tokens còn nguyên

#### **3.2. Context-Aware Detection** (MỚI)

```python
# YAML Key-based context
CODE_CONTEXT_KEYS = {
    'permission', 'command', 'node', 'id', 'type',
    'path', 'file', 'material', 'namespace', ...
}

TEXT_CONTEXT_KEYS = {
    'message', 'description', 'title', 'help',
    'lore', 'tooltip', 'error', 'success', ...
}

def is_code_context(path: str) -> bool:
    """Check YAML path: permissions.admin.node → CODE"""

def looks_like_code(text: str) -> bool:
    """Check patterns:
    - player.admin.ban → permission node
    - /spawn → command
    - minecraft:item → namespace ID
    - camelCase, snake_case → code
    """
```

#### **3.3. Language Detection**

```python
def detect_language(text: str) -> (lang_code, confidence):
    """
    1. Check English indicators first
       - "the", "you", "your", "please" → ALWAYS en

    2. Use langdetect (with caution)
       - Latin languages (ro/fr/it) → FALSE POSITIVES HIGH
       - Only skip if 99.5%+ confident

    3. CJK/Thai/Arabic detection
       - High confidence (>99.5%) → Skip translation
    """
```

**Tránh false positives:**
- ❌ "are you sure" bị nhầm là Romanian → FIXED (always translate)
- ❌ "Ban" bị nhầm là Vietnamese → FIXED (require 3+ words)

#### **3.4. Validation System**

```python
def validate_translation(original, translated):
    """
    1. Token validation (all placeholders present?)
    2. Length validation (4x ratio, 1500 chars max)
    3. Structure validation (brackets, quotes, newlines)
    4. Empty check
    """

def validate_structure(original, translated):
    """
    FIXED: Relaxed validation
    - Only check critical: [] {}
    - Allow ±1 brackets, ±2 quotes
    - Allow ±2 newlines or 50% change
    - NEVER check apostrophes (EN "it's" ≠ VI "nó")
    """
```

**Code thực tế:** `mine_inplace_translator.py:924-1738`

---

### 🔄 **Module 4: REBUILDER & VALIDATOR**

**Location:** Inside each `FileHandler` subclass

**Process:**

```python
def write_file(file_path, data, encoding):
    """
    1. Create backup (.bak)
    2. Write to temp file
    3. Atomic rename (temp → original)
    4. Validate (try parse again)
    5. On error: restore from backup
    """
```

**Features:**
- ✅ **Atomic writes:** Không corrupt file giữa chừng
- ✅ **Auto backup:** Luôn có .bak để rollback
- ✅ **Validation:** Parse lại để check YAML/JSON valid
- ✅ **Error recovery:** Tự động restore nếu fail

**Code thực tế:** `mine_inplace_translator.py:1800-1850` (YAMLHandler.write_file)

---

### 📖 **Module 5: GLOSSARY**

**Implementation:** `Translator.glossary` (Dict[str, str])

**Chức năng:**
```python
def apply_glossary_pre(text: str) -> (text_with_placeholders, map):
    """
    Before translation:
    - Replace glossary terms with placeholders
    - "server" → __GLOSS_001__
    """

def apply_glossary_post(text: str, map) -> text:
    """
    After translation:
    - Restore glossary terms with Vietnamese
    - __GLOSS_001__ → "máy chủ"
    """
```

**Usage:**
```bash
python3 mine_inplace_translator.py ./plugins --glossary terms.json
```

**terms.json:**
```json
{
  "server": "máy chủ",
  "admin": "quản trị viên",
  "player": "người chơi",
  "inventory": "túi đồ",
  "spawn": "điểm hồi sinh"
}
```

**Features:**
- ✅ Thuật ngữ dịch nhất quán
- ✅ Longest-match first (sort by length)
- ✅ Case-sensitive matching

**Code thực tế:** `mine_inplace_translator.py:1142-1162`

---

### 💾 **Module 6: TRANSLATION MEMORY** (Chưa có)

**Trạng thái:** ❌ **CHƯA IMPLEMENT**

**Đề xuất implementation:**

```python
class TranslationMemory:
    """SQLite-based translation cache"""

    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """
        CREATE TABLE tm (
            source_text TEXT PRIMARY KEY,
            target_text TEXT,
            context TEXT,
            timestamp INTEGER
        )
        """

    def get(self, source: str, context: str = None) -> Optional[str]:
        """Lookup translation from cache"""

    def put(self, source: str, target: str, context: str = None):
        """Store translation to cache"""
```

**Benefits:**
- ✅ Tránh dịch lại text giống nhau
- ✅ Nhất quán khi dịch nhiều files
- ✅ Nhanh hơn (cache hit = instant)

**TODO:** Cần implement trong version tiếp theo

---

## 4. LUỒNG XỬ LÝ

### 🔄 **Flow Diagram**

```
START
  │
  ├─→ 1. SCAN DIRECTORY
  │   └─→ Collect all .yml, .json, .lang, .properties, .txt files
  │
  ├─→ 2. FOR EACH FILE:
  │   │
  │   ├─→ 2.1. READ & PARSE
  │   │    └─→ YAMLHandler/JSONHandler/etc. reads file
  │   │
  │   ├─→ 2.2. TRANSLATE RECURSIVE
  │   │    │
  │   │    ├─→ 2.2.1. Check context (is_code_context?)
  │   │    │    └─→ YES: Skip (permission, command, id...)
  │   │    │    └─→ NO: Continue
  │   │    │
  │   │    ├─→ 2.2.2. Check if looks like code
  │   │    │    └─→ YES: Skip (permission.node, /command, camelCase...)
  │   │    │    └─→ NO: Continue
  │   │    │
  │   │    ├─→ 2.2.3. Should skip? (empty, Vietnamese)
  │   │    │    └─→ YES: Skip
  │   │    │    └─→ NO: Continue
  │   │    │
  │   │    ├─→ 2.2.4. Detect language
  │   │    │    └─→ Non-EN (CJK/Thai): Skip
  │   │    │    └─→ EN: Continue
  │   │    │
  │   │    ├─→ 2.2.5. PROTECT TOKENS
  │   │    │    └─→ %player% → __P001__
  │   │    │    └─→ &c → __C001__
  │   │    │    └─→ {balance} → __PH001__
  │   │    │
  │   │    ├─→ 2.2.6. APPLY GLOSSARY
  │   │    │    └─→ "server" → __GLOSS_001__
  │   │    │
  │   │    ├─→ 2.2.7. TRANSLATE (NLLB/Marian/etc.)
  │   │    │    └─→ Batch processing for speed
  │   │    │
  │   │    ├─→ 2.2.8. RESTORE GLOSSARY
  │   │    │    └─→ __GLOSS_001__ → "máy chủ"
  │   │    │
  │   │    ├─→ 2.2.9. RESTORE TOKENS
  │   │    │    └─→ __P001__ → %player%
  │   │    │    └─→ __C001__ → &c
  │   │    │
  │   │    └─→ 2.2.10. VALIDATE
  │   │         ├─→ Check tokens present?
  │   │         ├─→ Check length ratio?
  │   │         ├─→ Check structure?
  │   │         └─→ PASS: Use translation
  │   │              FAIL: Keep original
  │   │
  │   ├─→ 2.3. REBUILD & WRITE
  │   │    ├─→ Create backup (.bak)
  │   │    ├─→ Write to temp file
  │   │    ├─→ Atomic rename
  │   │    └─→ Validate parse
  │   │
  │   └─→ 2.4. LOG CHANGES
  │        └─→ Store in report.csv
  │
  └─→ 3. GENERATE REPORT
      └─→ mine_translate_report.csv

END
```

---

## 5. CÁC TÍNH NĂNG NÂNG CAO

### 🎯 **5.1. Smart Quantization Fallback**

```
Try: 4-bit quantization (fastest, 48 batch size)
  │
  ├─→ Success: Use 4-bit
  │
  └─→ Timeout/Error
      │
      Try: 8-bit quantization (slower, 24 batch size)
        │
        ├─→ Success: Use 8-bit
        │
        └─→ Timeout/Error
            │
            Try: Full precision (slowest, 12 batch size)
              │
              └─→ Success: Use full
```

### 🎯 **5.2. Adjacent Context Translation**

```python
# Instead of:
texts = ["Welcome", "to", "our", "server"]
# Translate individually → lose context

# Tool does:
joined = "Welcome⟨UNIT⟩to⟨UNIT⟩our⟨UNIT⟩server"
translated = translate(joined)  # Better context!
results = split(translated, "⟨UNIT⟩")
```

**Benefits:**
- ✅ Better context for AI
- ✅ More natural translations
- ✅ Faster (batch processing)

### 🎯 **5.3. Long Text Chunking**

```python
# Text > 1500 chars:
text = "Very long text with multiple sentences..."

# Chunk by sentences:
chunks = ["Sentence 1.", "Sentence 2.", ...]

# Translate each chunk:
translated_chunks = [translate(c) for c in chunks]

# Join back:
result = " ".join(translated_chunks)
```

**Avoids:** 512 token truncation limit

### 🎯 **5.4. Memory Management**

```python
# After each batch:
del inputs, outputs
torch.cuda.empty_cache()
gc.collect()

# Low memory mode:
--low-memory
  → Halve batch size
  → Disable adjacent grouping
  → Aggressive cache clearing
```

### 🎯 **5.5. Progress Tracking**

```python
# Real-time ETA calculation
ETACalculator:
  - Files processed: 45/100
  - Translation speed: 120 strings/min
  - ETA: 15 minutes remaining

# Progress bar with tqdm:
[████████████░░░░░░░░] 45% | 45/100 files | ETA: 15:32
```

---

## 6. SO SÁNH FEATURE-BY-FEATURE

| Feature | Đề xuất | Thực tế | Notes |
|---------|---------|---------|-------|
| **Scanner** | Quét thư mục | ✅ | `scan_directory()` |
| **Parser** | Tách text | ✅ | 4 handlers (YAML/JSON/Props/Text) |
| **Structure preservation** | Giữ cấu trúc | ✅ | ruamel.yaml, atomic writes |
| **Token protection** | Bảo vệ placeholder | ✅ | TokenProtector with 20+ patterns |
| **Color code protection** | Giữ màu | ✅ | &c, §e, MiniMessage |
| **AI Offline** | Model local | ✅ | NLLB/Marian/M2M100/Argos |
| **Quantization** | Tiết kiệm RAM | ✅ | 4-bit/8-bit/full fallback |
| **Validation** | Check output | ✅ | Token/length/structure validation |
| **Backup** | Backup file | ✅ | .bak auto-created |
| **Atomic write** | Không corrupt | ✅ | temp file + rename |
| **Translation Memory** | Cache dịch | ❌ | **TODO** |
| **Glossary** | Thuật ngữ | ✅ | JSON glossary support |
| **Context-aware** | Phân biệt code/text | ✅ | Key-based + pattern-based |
| **Language detection** | Phát hiện ngôn ngữ | ✅ | langdetect + manual rules |
| **Progress tracking** | Hiển thị tiến độ | ✅ | tqdm + ETA calculator |
| **Dry-run mode** | Preview | ✅ | --dry-run flag |
| **Rollback** | Khôi phục | ✅ | --rollback flag |
| **Report** | Log changes | ✅ | CSV report |

**Score: 17/18 features implemented (94%)**

---

## 7. ĐIỂM MẠNH CỦA TOOL HIỆN TẠI

✅ **Comprehensive:** Bao phủ 17/18 features
✅ **Production-ready:** Atomic writes, backups, validation
✅ **Smart:** Context-aware detection, không dịch code
✅ **Robust:** Fallback mechanisms, error handling
✅ **Fast:** Batch processing, quantization, caching
✅ **Offline:** Không cần internet sau khi download model
✅ **User-friendly:** Progress bars, dry-run, rollback
✅ **Extensible:** Easy to add new file handlers

---

## 8. ĐIỂM CẦN CẢI THIỆN

### ❌ **Translation Memory (Thiếu)**

**Đề xuất:**
```python
class TranslationMemory:
    """SQLite cache for translations"""

    # Usage:
    tm = TranslationMemory("cache.db")

    # Before translating:
    cached = tm.get(text, context="messages.welcome")
    if cached:
        return cached

    # After translating:
    tm.put(text, translated, context="messages.welcome")
```

**Benefits:**
- Tránh dịch lại text giống nhau
- Nhất quán giữa các files
- Nhanh hơn nhiều (cache hit = instant)

### ⚠️ **Sentence Segmentation (Có thể cải thiện)**

Hiện tại chunking dùng simple split, có thể cải thiện:

```python
# Current:
chunks = text.split(". ")

# Better:
import nltk
chunks = nltk.sent_tokenize(text)
```

### ⚠️ **Parallel Processing (Có thể thêm)**

Hiện tại xử lý tuần tự, có thể song song:

```python
from multiprocessing import Pool

# Translate multiple files in parallel
with Pool(4) as pool:
    results = pool.map(translate_file, files)
```

---

## 9. KẾT LUẬN

### 📊 **So sánh tổng quan**

```
MÔ HÌNH ĐỀ XUẤT:
  Scanner ───→ Parser ───→ Translator ───→ Rebuilder
              ↓                              ↓
         Glossary                       Validator
              ↓
      Translation Memory

TOOL THỰC TẾ:
  InPlaceTranslator ───→ FileHandler ───→ Translator ───→ Write + Validate
         (Scanner)          (Parser)     (Engine + Token Protection)
                              ↓                ↓
                         Glossary        Context-Aware
                                              ↓
                                      Validation System
```

### ✅ **Tool hiện tại có:**

1. ✅ **Scanner** - Quét thư mục đầy đủ
2. ✅ **Parser** - 4 loại file handlers
3. ✅ **Translator Engine** - NLLB with quantization
4. ✅ **Token Protection** - 20+ patterns
5. ✅ **Context-Aware** - Key + pattern detection
6. ✅ **Validator** - Token/length/structure
7. ✅ **Rebuilder** - Atomic writes + backup
8. ✅ **Glossary** - JSON support
9. ✅ **Progress** - tqdm + ETA
10. ✅ **Safety** - Dry-run, rollback, backup

### ❌ **Còn thiếu:**

1. ❌ **Translation Memory** - SQLite cache
2. ⚠️ **Parallel processing** - Multi-file parallel
3. ⚠️ **Better sentence segmentation** - NLTK

---

## 10. ROADMAP

### **Version hiện tại: v2.5** (94% complete)
- ✅ All core features
- ✅ Context-aware detection
- ✅ Smart validation

### **Version tiếp theo: v3.0**
- [ ] Translation Memory (SQLite)
- [ ] Parallel file processing
- [ ] Better sentence segmentation (NLTK)
- [ ] Web UI (optional)
- [ ] Plugin system for custom handlers

---

**Tác giả:** Claude + User collaboration
**Ngày cập nhật:** 2025-01-10
**License:** Same as main tool
