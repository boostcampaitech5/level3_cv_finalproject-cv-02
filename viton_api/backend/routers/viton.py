# fastapi
from fastapi import Form
from fastapi import APIRouter

# built-in library
import io
import base64

# custom-library
# from infra.viton.hr_viton.test_generator import inference
from infra.viton.ladi_viton.src.ladi_vton import LadiVton


router = APIRouter(
    prefix="/viton",
    tags=["viton"]
)

ladi_vton = LadiVton()


# @router.post("/hr_viton")
# def hr_inference(storage_root: str, p_img_name: str, c_img_name: str):
#     img_name = inference(storage_root, p_img_name, c_img_name)

#     return img_name

@router.post("/ladi_viton")
def ladi_inference(storage_root: str = Form(...), p_img_name: str = Form(...),
                   c_img_name: str = Form(...), category: str = Form(...)):
    
    tryon_img = ladi_vton.inference(storage_root, p_img_name, c_img_name, category)

    tryon_img_bytes = io.BytesIO()
    tryon_img.save(tryon_img_bytes, format="PNG")
    
    tryon_img_base64 = base64.b64encode(tryon_img_bytes.getvalue()).decode("utf-8")

    return tryon_img_base64
