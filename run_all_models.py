from TTS.api import TTS


for model_name in TTS.list_models():
    tts_model = TTS(model_name, gpu=True)

    kwargs = {
        "text": "This is a test! This is also a test!!",
        "emotion": "Happy",
        "speed": 1.5,
    }

    if tts_model.is_multi_speaker:
        speaker = tts_model.speakers[0]
        kwargs["speaker"] = speaker

    if tts_model.is_multi_lingual:
        language = tts_model.languages[0]
        kwargs["language"] = language
    

    tts_model.tts_to_file(**kwargs)