"""C1 단계 스크립트에서 C2 단계(5단계 과적합 대응)를 불러오기 위한 연결."""
import os
import sys

C2_FLOW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "C2", "flow"))
if C2_FLOW not in sys.path:
    sys.path.insert(0, C2_FLOW)

from c2_step05_overfit_fix import compute as step05_compute  # noqa: E402,F401
