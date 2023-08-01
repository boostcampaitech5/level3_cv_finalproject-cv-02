import asyncio
import yaml
import os

class GoogleTranslateClient:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.language = self.conf["translate"]["language"]

        from gpytranslate import Translator
        self.translator = Translator()
    
    async def translate(self, text: str):
        result = await self.translator.translate(text, targetlang=self.language)
        return result.text
