#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from collections.abc import Callable

from quark.experimental.torch.quant_perf import landing
from quark.experimental.torch.quant_perf.repair.request import build_repair_request
from quark.experimental.torch.quant_perf.repair.service import RepairService
from quark.experimental.torch.quant_perf.runtime.recovery import classify_failure
from quark.experimental.torch.quant_perf.session.spec import ServerHandle, Spec, StageError


class LandingStage:
    def __init__(
        self,
        repair_service: RepairService,
        loader: Callable[..., ServerHandle] | None = None,
    ) -> None:
        self.repair_service = repair_service
        self.loader = loader or landing.load

    def load_model_with_repair(
        self,
        spec: Spec,
        model_dir: str,
        *,
        profiler_dir: str | None = None,
    ) -> ServerHandle:
        try:
            return self.loader(
                model_dir,
                spec,
                profiler_dir=profiler_dir,
            )
        except StageError as first_error:
            diagnosis = classify_failure(
                first_error,
                framework_repo=spec.active_framework_repo,
                kernel_repo=spec.active_kernel_repo,
            )
            request = build_repair_request(
                spec,
                failure_class="load_run",
                error=first_error.diagnostic,
                quant_ckpt_dir=model_dir,
                verifier_profile="load_inference",
                diagnosis=diagnosis,
            )
            if not diagnosis.repair_eligible or not self.repair_service.can_repair(request):
                raise
            repair = self.repair_service.repair(request)
            if repair.status != "fixed":
                raise
            return self.loader(
                model_dir,
                spec,
                profiler_dir=profiler_dir,
            )
