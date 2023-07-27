from fastapi import FastAPI
import uvicorn

from routers.densepose import router


app = FastAPI()
app.include_router(router)


@app.get("/")
def check():
    return {"message": "ready to use a densepose program"}


if __name__ == "__main__":
    uvicorn.run("densepose:app", host="0.0.0.0", port=9003, reload=False)