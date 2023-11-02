from dataclasses import field, dataclass


@dataclass
class Flops:
    # float-point operators
    flops: float = field(compare=True)

    def convert_gflops(self):
        # 1 GFLOPS (gigaFLOPS) is equal to 10^9 times float-point operators，
        return self.flops / 10e9

    pass