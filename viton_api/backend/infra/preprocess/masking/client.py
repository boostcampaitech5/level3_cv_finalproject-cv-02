import httpx


# cloth_mask 프로그램에 inference 요청
class MaskingClient:
    API_URL = "http://localhost:9000/masking"

    async def predict_mask(self, storage_root: str, img_name: str, category: str = 'cloth') -> str:
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'img_name': img_name,
                'category': category
            }
            response = await client.post(
                self.API_URL,
                data = data,
                timeout=None
            )

            return response.text