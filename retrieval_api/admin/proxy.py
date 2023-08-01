from fastapi import FastAPI
import uvicorn

from dotenv import dotenv_values
from pymongo import MongoClient

from fastapi.middleware.cors import CORSMiddleware

# config = dotenv_values(".env")

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

# client = MongoClient(config["ATLAS_URI"],connect=False)


@app.get("/")
async def servercheck() -> dict:
    return {
        "message": "Proxy server is OK!"
    }


# @app.on_event("startup")
# def startup_db_client():
#     app.mongodb_client = MongoClient(config["ATLAS_URI"])
#     app.database = app.mongodb_client[config["DB_NAME"]]
#     print("Connected to the MongoDB database!")

# @app.on_event("shutdown")
# def shutdown_db_client():
#     app.mongodb_client.close()
#     print("Disconnected to the MongoDB database!")


if __name__ == "__main__":
    uvicorn.run("proxy:app", port=30007, host="0.0.0.0", reload=True)
