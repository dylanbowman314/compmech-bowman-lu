from pathlib import Path


def get_cached_belief_filename(x: float, a: float):
    return f"src/cached_belief_store/paths_beliefs_x{str(x).replace(".","")}_a{str(a).replace(".","")}.pt"


def get_jpg_filename(x1: float, a1: float, x2: float, a2: float):
    return f"src/images/x{str(x1).replace(".","")}_a{str(a1).replace(".","")}_to_x{str(x2).replace(".","")}_a{str(a2).replace(".","")}.jpg"


# TODO: these should really be env vars but oh well
MODEL_PATH_015_06 = Path(
    "/workspaces/cure/compmech-models/models/f6gnm1we-mess3-0.15-0.6/"
)
MODEL_PATH_005_085 = Path(
    "/workspaces/cure/compmech-models/models/54qc0vyb_mess3-0.05-0.85"
)
