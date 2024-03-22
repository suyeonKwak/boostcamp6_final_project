from deep_translator import GoogleTranslator


translated = GoogleTranslator(source="ko", target="en").translate("이건 테스트 입니다")
print(translated)
