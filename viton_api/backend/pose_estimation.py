# fastapi
from fastapi import FastAPI
import uvicorn

# custom-library
from routers.pose_estimation import router

app = FastAPI()
app.include_router(router)


@app.get("/")
def check():
    return {"message": "ready to use a pose estimation program"}


if __name__ == "__main__":
    uvicorn.run("pose_estimation:app", host="0.0.0.0", port=9001, reload=False)