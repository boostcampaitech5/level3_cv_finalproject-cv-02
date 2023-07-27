from fastapi import FastAPI
import uvicorn

from routers.masking import router

app = FastAPI()
app.include_router(router)


@app.get("/")
def check():
    return {"message": "ready to use a cloth mask program"}


if __name__ == "__main__":
    uvicorn.run("masking:app", host="0.0.0.0", port=9000, reload=False)