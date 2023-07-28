import httpx
from fastapi import HTTPException, status

# --embedding server에 embedding 요청
class EmbeddingClient:
    # --embedding server api
    API_URL = "http://localhost:9000/embedding"
    
    async def get_image_embedding(self, rid):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.API_URL}/image/{rid}",
                timeout=None
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="EmbeddingClient - get_image_embedding is error!"
                )
            
            embedding = response.json()["embedding"]
            return embedding
        
    
    async def get_text_embedding(self, text: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_URL}/text",
                timeout=None,
                data = {
                    "text": text,
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="EmbeddingClient - get_text_embedding is error!"
                )
            
            embedding = response.json()["embedding"]
            return embedding            