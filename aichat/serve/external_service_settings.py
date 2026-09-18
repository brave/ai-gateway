from pydantic_settings import BaseSettings


class ExternalServiceSettings(BaseSettings):
    ai_chat_premium_host: str = ""
    near_api_key: str = ""
    near_bucket_name: str = ""
    near_bucket_region: str = ""
    analytics_enabled: bool = False
    analytics_model_address: str = ""
    bedrock_retry_delay_seconds: float = 2.0
    bedrock_max_retries: int = 3
    bedrock_image_limit: int = 20
    bedrock_document_limit: int = 5
    max_pdf_file_size_mb: int = 20
    max_stt_upload_size_mb: int = 25
    max_stt_audio_duration_seconds: int = 600
    global_image_limit: int = 20
    share_s3_bucket: str = ""
    share_viewer_origin: str = ""
    share_max_ciphertext_bytes: int = 1048576  # 1 MB


external_service_settings = ExternalServiceSettings()
