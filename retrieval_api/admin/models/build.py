from pydantic import BaseModel

class BuildParam(BaseModel):
    keys: list[int]
    paths: list[str]
    