import httpx


# pose_estimation 프로그램에 inference 요청
class PoseEstimationClient:
    API_URL = "http://localhost:9001/pose_estimation"

    async def predict_pose_kpts(self, storage_root: str, img_name: str, model_name: str) -> str:
        async with httpx.AsyncClient() as client:
            data = {
                'storage_root': storage_root,
                'img_name': img_name,
                'model_name': model_name
            }
            response = await client.post(
                self.API_URL,
                data = data,
                timeout=None
            )

            return response.text