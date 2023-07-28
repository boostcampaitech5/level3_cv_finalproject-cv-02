import httpx
import pandas as pd
from fastapi import HTTPException, status, UploadFile, File

class BuildClient:
    API_URL = "http://localhost:8003/build"
    
    async def make_pickle(self, keys: list[int], paths: list[str]):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_URL}/make-pickle",
                timeout=None,
                json = {
                    "keys": keys,
                    "paths": paths
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="make pickle client error!"
                )
                
            response = response.json()
            
            return response["pickles"]
