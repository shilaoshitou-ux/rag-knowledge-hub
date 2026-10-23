"""Model service choices and validation of OpenAI-compatible API endpoints."""

from urllib.parse import urlsplit

# Default service catalog (public pricing list, 2026-10-22); availability depends on the key.
CHAT_MODELS = (
    "gpt-6-sol",
    "gpt-6.1-sol",
    "gpt-6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)
COMPANY_EMBEDDING_MODELS = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-v4",
)
OPENAI_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
LOCAL_EMBEDDING_MODELS = ("bge-m3", "nomic-embed-text")
LOCAL_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CUSTOM_MODEL = "自定义（填写模型 ID）"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
COMPANY_BASE_URL = "https://llm-proxy.yuwan-game.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SERVICE = "默认服务"
OPENAI_SERVICE = "OpenAI"
CUSTOM_SERVICE = "自定义接口"
SERVICE_OPTIONS = (DEFAULT_SERVICE, OPENAI_SERVICE, CUSTOM_SERVICE)


def clean_model_id(value: str) -> str:
    """Validate user input without claiming this service offers the model."""
    model = value.strip()
    if not model or len(model) > 128 or any(char.isspace() for char in model):
        raise ValueError("模型 ID 不能为空、包含空格或超过 128 个字符。")
    return model


def validate_base_url(value: str) -> str:
    """Only HTTPS remotes and loopback HTTP; reject keys and parameters in URLs."""
    url = value.strip().rstrip("/")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("接口地址格式不正确。") from exc
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query
            or parsed.fragment or any(char.isspace() for char in url)):
        raise ValueError("接口地址不能包含密钥、空格、查询参数或片段。")
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return url
    raise ValueError("远程接口必须使用 HTTPS；仅本机接口可以使用 HTTP。")


def service_base_url(service: str, custom_url: str = "") -> str:
    """Resolve a selected service without looking up credentials from another service."""
    if service == DEFAULT_SERVICE:
        return COMPANY_BASE_URL
    if service == OPENAI_SERVICE:
        return OPENAI_BASE_URL
    if service == CUSTOM_SERVICE:
        return validate_base_url(custom_url)
    raise ValueError("请选择有效的模型服务。")


def safe_error(error: Exception, api_key: str) -> str:
    """Do not echo the active key even if a gateway includes it in its error."""
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[已隐藏密钥]")
    return message[:600]
