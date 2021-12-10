from pathlib import Path
import fire

def link_body_models(smplh_dir: Path):
    if type(smplh_dir) is not Path:
        smplh_dir = Path(smplh_dir)
    if (smplh_dir / "SMPLH_NEUTRAL.npz").exists():
        body_model_locs = {
            "male": smplh_dir/"SMPLH_MALE.npz",
            "female": smplh_dir/"SMPLH_FEMALE.npz",
            "neutral": smplh_dir/"SMPLH_NEUTRAL.npz"
        }
    elif (smplh_dir / "neutral").exists():
        body_model_locs = {
            "male": smplh_dir / "male/model.npz",
            "female": smplh_dir / "female/model.npz",
            "neutral": smplh_dir / "neutral/model.npz"
        }
    else:
        raise ValueError(f"No valid SMPLH directory structures found at {smplh_dir}")

    target_dir = Path("./body_models/smplh")

    for gender, loc in body_model_locs.items():
        t = target_dir / gender
        t.mkdir(exist_ok=True, parents=True)
        t = t / "model.npz"
        if t.exists():
            print(f"{t.resolve()} -> {t} exists")
        else:
            t.symlink_to(loc)
            print(f"{loc} -> {t}")

if __name__ == "__main__":
    fire.Fire(link_body_models)
