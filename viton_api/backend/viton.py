from fastapi import FastAPI
import uvicorn

from routers.viton import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def check():
    return {"message": "ready to use a virtual try-on program"}


if __name__ == "__main__":
    uvicorn.run("viton:app", host="0.0.0.0", port=10000, reload=False)