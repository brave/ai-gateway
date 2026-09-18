PRODUCTION = "production"
DEPRECATED_API_RESPONSE = "I'm unable to answer your query as your version of Brave is too out of date. Please update your browser."

# Best-effort localizations of the /conversation deprecation notice, keyed by
# the primary ISO 639-1 language subtag. English is always available as a
# fallback so the notice stays understandable even without a localization.
CONVERSATION_API_DEPRECATION_MESSAGES = {
    "en": "Please update your browser to a newer version to continue using Leo.",
    "es": "Actualiza tu navegador a una versión más reciente para seguir usando Leo.",
    "fr": "Merci de mettre à jour votre navigateur vers une version plus récente pour continuer à utiliser Leo.",
    "de": "Bitte aktualisiere deinen Browser auf eine neuere Version, um Leo weiterhin nutzen zu können.",
    "pt": "Atualize seu navegador para uma versão mais recente para continuar usando o Leo.",
    "it": "Aggiorna il tuo browser a una versione più recente per continuare a usare Leo.",
    "ja": "Leo をご利用いただくには、ブラウザを最新バージョンに更新してください。",
    "zh": "请将您的浏览器更新到最新版本以继续使用 Leo。",
    "ko": "Leo를 계속 사용하려면 브라우저를 최신 버전으로 업데이트해 주세요.",
    "ru": "Обновите браузер до новой версии, чтобы продолжить использовать Leo.",
    "nl": "Update je browser naar een nieuwere versie om Leo te kunnen blijven gebruiken.",
}
CONTENT_FILTER_MESSAGE = (
    "The model is having difficulty responding to the content in the conversation. "
    "Please try again with different content."
)
