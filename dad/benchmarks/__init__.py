# from .benchmark import Benchmark as Benchmark
from .num_inliers import NumInliersBenchmark as NumInliersBenchmark
from .megadepth import Mega1500 as Mega1500
from .megadepth import Mega1500_F as Mega1500_F
from .megadepth import MegaIMCPT as MegaIMCPT
from .megadepth import MegaIMCPT_F as MegaIMCPT_F
from .scannet import ScanNet1500 as ScanNet1500
from .scannet import ScanNet1500_F as ScanNet1500_F
from .hpatches import HPatchesViewpoint as HPatchesViewpoint
from .hpatches import HPatchesIllum as HPatchesIllum

all_benchmarks = [
    Mega1500.__name__,
    Mega1500_F.__name__,
    MegaIMCPT.__name__,
    MegaIMCPT_F.__name__,
    ScanNet1500.__name__,
    ScanNet1500_F.__name__,
    HPatchesViewpoint.__name__,
    HPatchesIllum.__name__,
]
