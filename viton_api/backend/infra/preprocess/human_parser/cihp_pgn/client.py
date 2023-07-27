import httpx


class HumanParsingClient:
    API_URL = "http://localhost:9002/human_parser"

    async def human_parsing(self, storage_root: str, img_name: str):
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'img_name': img_name
            }
            response = await client.post(
                self.API_URL,
                data=data,
                timeout=None
            )

            return response.text