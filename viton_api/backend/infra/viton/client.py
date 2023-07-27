import httpx


# viton 프로그램에 inference 요청
class VitOnClient:
    API_URL = "http://localhost:10000/viton"

    async def hr_viton(self, storage_root: str, p_img_name: str, c_img_name: str):
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'p_img_name': p_img_name,
                'c_img_name': c_img_name
            }
            response = await client.post(
                f"{self.API_URL}/hr_viton",
                data=data,
                timeout=None
                )

            return response.text
        
    async def ladi_viton(self, storage_root: str, p_img_name: str, c_img_name: str, category: str):
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'p_img_name': p_img_name,
                'c_img_name': c_img_name,
                'category': category
            }
            response = await client.post(
                f"{self.API_URL}/ladi_viton",
                data=data,
                timeout=None
            )

            return response.content