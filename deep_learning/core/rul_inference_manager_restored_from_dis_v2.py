class RULInferenceManager:
    def __init__(self, cfg, ckpt, precomputed_loader):
        self.cfg = cfg
        self.ckpt = ckpt
        self.loader = precomputed_loader
        self.last_record = None
        self.last_r_ratio = None
        self.last_battery = None
        self.source = None

    def load_or_infer(self, battery: str, r_ratio: float, force_recompute: bool):
        """
        force_recompute = False → precomputed 우선 로드
        force_recompute = True → 무조건 BMAML 재적응 수행
        """
        if (not force_recompute) and self.loader.has_precomputed(battery, r_ratio):
            rec = self.loader.load(battery, r_ratio)
            self._update_state(rec, battery, r_ratio, source="precomputed")
            return rec

        from deep_learning.core.prefix_inference_viz_meta import run_bmaml_once

        rec = run_bmaml_once(battery=battery, cfg=self.cfg, ckpt=self.ckpt, r_ratio=r_ratio)
        self.loader.save(battery, r_ratio, rec)
        self._update_state(rec, battery, r_ratio, source="bmaml")
        return rec

    def _update_state(self, rec, battery, r_ratio, source):
        self.last_record = rec
        self.last_battery = battery
        self.last_r_ratio = r_ratio
        self.source = source

    def get_last(self):
        return (self.last_record, self.source)
