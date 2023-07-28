from fastapi import FastAPI
import uvicorn

app = FastAPI()

from routes.build import build_router

app.include_router(build_router, prefix="/build")

@app.get("/")
async def servercheck() -> dict:
    return {
            "message": "build server is OK!"
        }


if __name__ == "__main__":
    uvicorn.run("build:app", port=8003, host="0.0.0.0", reload=True)
