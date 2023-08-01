import httpx


# cloth_mask 프로그램에 inference 요청
class DensePoseClient:
    API_URL = "http://localhost:9003/densepose"

    async def predict_dense_pose_map(self, storage_root: str, img_name: str):
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'img_name': img_name
            }
            response = await client.post(
                f"{self.API_URL}?storage_root={storage_root}&img_name={img_name}",
                timeout=None
            )

            return response.text