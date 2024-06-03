from pathlib import Path


def get_cached_belief_filename(x: float, a: float):
    return f"src/cached_belief_store/paths_beliefs_x{format(x,".3g").replace(".","")}_a{format(a,".3g").replace(".","")}.pt"


def get_jpg_filename(model_path: str, x: float, a: float):
    return f"src/images/{str(model_path).split("/")[-2]}_x{format(x,".3g").replace(".","")}_a{format(x,".3g").replace(".","")}.jpg"


# TODO: these should really be env vars but oh well
MODEL_PATH_015_06 = Path(
    "/root/compmech-bowman-lu/src/models/f6gnm1we-mess3-0.15-0.6/"
)
MODEL_PATH_005_085 = Path(
    "/root/compmech-bowman-lu/src/models/54qc0vyb_mess3-0.05-0.85"
)

