"""模型系列推断。根据模型名推断厂商/系列，供打标和过滤使用。"""

# 前缀 → (厂商, 系列)
_MODEL_PREFIX_MAP: list[tuple[str, str, str]] = [
    # OpenAI
    ("gpt-", "openai", "gpt"),
    ("o1", "openai", "o1"),
    ("o3", "openai", "o3"),
    ("o4", "openai", "o4"),
    # Anthropic
    ("claude-", "anthropic", "claude"),
    # Google
    ("gemini-", "google", "gemini"),
    ("gemma-", "google", "gemma"),
    # Meta
    ("llama-", "meta", "llama"),
    # Mistral
    ("mistral-", "mistral", "mistral"),
    ("codestral-", "mistral", "codestral"),
    # DeepSeek
    ("deepseek-", "deepseek", "deepseek"),
    # Qwen
    ("qwen-", "qwen", "qwen"),
    ("qvq-", "qwen", "qvq"),
    # GLM
    ("glm-", "zhipu", "glm"),
    # Yi
    ("yi-", "01ai", "yi"),
    # MiniMax
    ("abab-", "minimax", "abab"),
    ("abab", "minimax", "abab"),
    ("minimax-", "minimax", "minimax"),
    # Moonshot / Kimi
    ("moonshot-", "moonshot", "moonshot"),
    ("kimi-", "moonshot", "kimi"),
    # ByteDance
    ("doubao-", "bytedance", "doubao"),
    # Step
    ("step-", "stepfun", "step"),
    # xAI
    ("grok-", "xai", "grok"),
    # Amazon
    ("nova-", "amazon", "nova"),
    # Cohere
    ("command-", "cohere", "command"),
    # AI21
    ("jamba-", "ai21", "jamba"),
    ("jurassic-", "ai21", "jurassic"),
]


def detect_family(model: str) -> tuple[str, str]:
    """根据模型名推断 (厂商, 系列)。匹配不到返回 ("unknown", "unknown")。"""
    lower = model.lower()
    for prefix, vendor, family in _MODEL_PREFIX_MAP:
        if lower.startswith(prefix):
            return vendor, family
    return "unknown", "unknown"
