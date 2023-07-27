from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers.proxy import router


app = FastAPI()
app.include_router(router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def check():
    return {"message": "ready to use a proxy program"}


if __name__ == "__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=30008, reload=False)