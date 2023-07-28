import httpx

# --search server에 search 요청
class SearchClient:
    ## -- search server api url
    API_URL = "http://localhost:9001/search"
    print("여기 밑으로 못 내려가는건가?")
    
    
    async def search(self, embedding: list[float], thresh: float) -> tuple[list[float], list[int]]:
        async with httpx.AsyncClient() as client:
            print("단일한 embedding 요청 들어왔나?")
            response = await client.post(
                f"{self.API_URL}",
                json={
                    "thresh": thresh,
                    "embedding": embedding,
                    },
                timeout=None
            )
        
            print("단일 embedding의 response의 Status_code는 여기에 있다!", response.status_code)
            
            dists = response.json()["dists"]
            ids = response.json()["ids"]
        
        return dists, ids
    
    
    async def search_with_filter(self, embedding: list[float], filter_embedding: list[float], thresh: float) -> tuple[list[float], list[int]]:
        print("여기는 client의 search_with_filter야! 잘 들어왔는지 확인하는 곳임@-@!")
        async with httpx.AsyncClient() as client:
            print("embedding들 요청 들어왔나?")
            response = await client.post(
                f"{self.API_URL}/with-filter",
                json={
                    "thresh": thresh,
                    "embedding": embedding,
                    "filter_embedding": filter_embedding,
                },
                timeout=None
            )
            
            print("embedding들의 response의 Status_code는 여기에 있다!", response.status_code)

            dists = response.json()["dists"]
            ids = response.json()["ids"]
        
        return dists, ids