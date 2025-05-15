import yaml
import ml_collections


class CFG:

    @classmethod
    def read_from_yaml(cls, cfg_path):
        cfg = ml_collections.ConfigDict()
        with open(cfg_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        cls._read_from_yaml_dfs(cfg, params)
        return cfg

    @classmethod
    def _read_from_yaml_dfs(cls, cfg, param):
        for k, v in param.items():
            if isinstance(v, dict):
                inner_cfg = ml_collections.ConfigDict()
                cls._read_from_yaml_dfs(inner_cfg, v)
                cfg[k] = inner_cfg
            else:
                cfg[k] = v

    @classmethod
    def read_from_command(cls, args):
        pass
