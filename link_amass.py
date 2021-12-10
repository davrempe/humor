from pathlib import Path
import fire

amass_subdirs = [
    "ACCAD",
    "BioMotionLab_NTroje",
    "BMLhandball",
    "BMLmovi",
    "CMU",
    "DanceDB",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "KIT",
    "MPI_Limits",
    "TCD_handMocap",
    "TotalCapture",
    "HumanEva",
    "MPI_HDM05",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "Transitions_mocap",
]

def link_amass(amass_dir: Path):
    if type(amass_dir) is not Path:
        amass_dir = Path(amass_dir)
    amass_subdir_locs = {}
    for p in amass_dir.glob("**/*"):
        if p.is_dir():
            if p.stem in amass_subdirs:
                amass_subdir_locs[p.stem] = p
    target_dir = Path("./data/amass_raw")
    target_dir.mkdir(exist_ok=True, parents=True)
    
    for subdir, loc in amass_subdir_locs.items():
        t = target_dir / subdir
        if t.exists():
            print(f"{t.resolve()} -> {t} exists")
        else:
            t.symlink_to(loc)
            print(f"{loc} -> {t} created")

if __name__ == "__main__":
    fire.Fire(link_amass)


