from os.path import join
from model.hidden1 import LitHidden1
from config.const import PROJECT_ROOT
from config import get_config

def train(experimental_setup_path):
    cfg = get_config()
    cfg.merge_from_file(experimental_setup_path)
    cfg.freeze()
    print(cfg)

    model = LitHidden1(args=cfg)

if __name__ == "__main__":
    train(join(PROJECT_ROOT, "config", "toy.yaml"))

