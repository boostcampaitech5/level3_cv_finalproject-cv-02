from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.proxy import proxy_router

app.include_router(proxy_router, prefix="/proxy")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    print(f"response time: {elapsed_time:.3f} sec")
    return response


app.mount("/storage", StaticFiles(directory="../../storage"), name="storage")

@app.get("/")
async def servercheck() -> dict:
    return {
        "message": "Proxy server is OK!"
    }


if __name__ == "__main__":
    uvicorn.run("proxy:app", port=30007, host="0.0.0.0", reload=False)
