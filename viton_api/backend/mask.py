from fastapi import FastAPI
import uvicorn

from routers.mask import router

app = FastAPI()
app.include_router(router)


@app.get("/")
def check():
    return {"message": "ready to use a mask program"}


if __name__ == "__main__":
    uvicorn.run("mask:app", host="0.0.0.0", port=9000, reload=False)