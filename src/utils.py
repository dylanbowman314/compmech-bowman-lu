from pathlib import Path


def get_cached_belief_filename(x: float, a: float):
    return f"src/cached_belief_store/paths_beliefs_x{format(x,".3g").replace(".","")}_a{format(a,".3g").replace(".","")}.pt"


def get_jpg_filename(model_path: str, x: float, a: float):
    return f"src/images/{str(model_path).split("/")[-2]}_x{format(x,".3g").replace(".","")}_a{format(x,".3g").replace(".","")}.jpg"


# TODO: these should really be env vars but oh well
MODEL_PATH_015_06 = Path.cwd().resolve() / "pretrained_hmm_transformers/015_06"
MODEL_PATH_005_085 = Path.cwd().resolve() / "pretrained_hmm_transformers/005_085"
