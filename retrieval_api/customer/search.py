from fastapi import FastAPI, Request
import uvicorn
import time

app = FastAPI()

from routes.search import search_router

app.include_router(search_router, prefix="/search")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    print(f"response time: {elapsed_time:.3f} sec")
    return response


@app.get("/")
async def servercheck() -> dict:
    return {
        "message": "Search server is OK!"
    }
    

if __name__ == "__main__":
    uvicorn.run("search:app", port=9001, host="0.0.0.0", reload=False)