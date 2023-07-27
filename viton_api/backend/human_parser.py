from fastapi import FastAPI
import uvicorn

from routers.human_parser import router


app = FastAPI()
app.include_router(router)


@app.get("/")
def check():
    return {"message": "ready to use a human parsing program"}


if __name__ == "__main__":
    uvicorn.run("human_parser:app", host="0.0.0.0", port=9002, reload=False)